# -*- coding: utf-8 -*-
"""X2 资源 → mjswan 资产：参考动作 npz + 循环版 ONNX + 带站立 keyframe 的 MJCF。

X2 契约（官方 SDK extra/x2_rl_deploy）：
- 输入 obs[1,151] + time_step[1,1]；输出 actions/joint_pos/joint_vel/body_*_w[/motion_phase]
- obs = command_dof_pos(29, 上一步参考) + command_dof_vel(29) + projected_gravity(3)
        + base_ang_vel(3) + dof_pos_rel(29) + dof_vel*0.05(29) + actions(29)
- 参考分支只由 time_step 驱动；time_step 在 [phase.start, phase.end] 间推进，到 end 后
  部署端重放（本脚本用 Mod 包装成浏览器可无限循环的 time_step）。

产物（默认 scratch/oss_survey/poc_x2/assets/<资源名>/）：
- motion.npz: joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w/body_names/fps
- policy_loop.onnx: 原始策略 + time_step 循环包装
- meta.json: 关节/body/default/scale/kp/kd/phase/文件路径
- x2_mjswan.xml: 修好 meshdir、补了站立 keyframe 的 X2 MJCF（写到 models/x2/mjcf/ 下）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(os.environ.get("ONNX_PLAYER_ROOT") or str(Path(__file__).resolve().parents[1]))
X2_XML = ROOT / "models/x2/mjcf/x2_ultra.xml"
CACHE_DIR = ROOT / "mjswan/cache/x2"
FPS = 50.0


def floats(text):
    return [float(x) for x in str(text).replace("\n", ",").split(",") if x.strip()]


def load_meta(onnx_path: Path) -> dict:
    m = onnx.load(str(onnx_path), load_external_data=False)
    md = {kv.key: kv.value for kv in m.metadata_props}
    return {
        "joint_names": [x.strip() for x in md["joint_names"].split(",") if x.strip()],
        "body_names": [x.strip() for x in md["body_names"].split(",") if x.strip()],
        "anchor": md["anchor_body_name"],
        "default": floats(md["default_joint_pos"]),
        "scale": floats(md["action_scale"]),
        "kp": floats(md["joint_stiffness"]),
        "kd": floats(md["joint_damping"]),
        "time_step_total": int(float(md.get("time_step_total", "0") or 0)),
        "obs_dim": int(m.graph.input[0].type.tensor_type.shape.dim[1].dim_value),
    }


def extract_reference(onnx_path: Path, meta: dict) -> dict:
    """按部署端语义（上一步参考 + 零状态）跑参考分支，产出参考动作与全量输出。"""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    n = len(meta["joint_names"])
    probe_obs = np.zeros((1, 5 * n + 6), dtype=np.float32)
    probe_obs[0, 2 * n + 2] = -1.0
    feeds = {"obs": probe_obs, "time_step": np.array([[0.0]], dtype=np.float32)}
    start, end = 0, max(meta["time_step_total"] - 1, 0)
    if "motion_phase" in outs:
        phase = sess.run(["motion_phase"], feeds)[0][0]
        start, end = int(round(float(phase[0]))), int(round(float(phase[1])))
    total = end - start + 1
    if total <= 0:
        raise RuntimeError(f"无法确定动作步范围：start={start} end={end}")
    print(f"phase=[{start},{end}] total={total} steps ({total / FPS:.2f}s)", flush=True)

    want = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    missing = [k for k in want if k not in outs]
    if missing:
        raise RuntimeError(f"ONNX 缺少输出：{missing}")

    prev_jp = np.zeros(n, dtype=np.float32)
    prev_jv = np.zeros(n, dtype=np.float32)
    prev_a = np.zeros(n, dtype=np.float32)
    frames = {k: [] for k in want}
    for i, step in enumerate(range(start, end + 1)):
        obs = np.concatenate([
            prev_jp, prev_jv, np.array([0, 0, -1], np.float32), np.zeros(3, np.float32),
            np.zeros(n, np.float32), np.zeros(n, np.float32), prev_a,
        ]).astype(np.float32).reshape(1, -1)
        res = sess.run(want + ["actions"], {"obs": obs, "time_step": np.array([[float(step)]], np.float32)})
        d = dict(zip(want + ["actions"], res))
        prev_jp = np.asarray(d["joint_pos"][0], np.float32)
        prev_jv = np.asarray(d["joint_vel"][0], np.float32)
        prev_a = np.asarray(d["actions"][0], np.float32)
        for k in want:
            frames[k].append(np.asarray(d[k][0], np.float32))
        if i % 200 == 0:
            print(f"  ref {i}/{total}", flush=True)

    data = {k: np.ascontiguousarray(np.stack(v)) for k, v in frames.items()}
    return {"start": start, "end": end, "total": total, **data}


def remap_body_names(body_names: list[str], model: mujoco.MjModel) -> list[str]:
    """X2 训练数据里根 body 叫 base_link，MJCF 里叫 pelvis；其余同名。"""
    available = {model.body(i).name for i in range(model.nbody)}
    mapped = []
    for name in body_names:
        if name in available:
            mapped.append(name)
        elif name == "base_link" and "pelvis" in available:
            mapped.append("pelvis")
        else:
            raise RuntimeError(f"body {name!r} 在 X2 MJCF 中不存在")
    return mapped


def build_loop_onnx(src: Path, out: Path, start: int, total: int, obs_dim: int) -> None:
    """把 time_step 包装成 start + (t mod total)，让策略在浏览器里无限循环。"""
    model = onnx.load(str(src), load_external_data=False)
    g = model.graph
    if len(g.input) != 2 or g.input[1].name != "time_step":
        raise RuntimeError(f"期望两个输入 (obs, time_step)，实际 {[i.name for i in g.input]}")

    init_names = {i.name for i in g.initializer}
    consts = [
        numpy_helper.from_array(np.array([[float(total)]], np.float32), name="__loop_total"),
        numpy_helper.from_array(np.array([[float(start)]], np.float32), name="__loop_start"),
    ]
    assert not ({"__loop_total", "__loop_start"} & init_names)
    # 只用常见的 float 算子做取模（Mod 在部分 ORT Web 构建里支持不稳定）：
    #   ts_mod = ts - total * floor(ts / total)
    nodes = [
        helper.make_node("Div", ["time_step", "__loop_total"], ["__ts_div"]),
        helper.make_node("Floor", ["__ts_div"], ["__ts_floor"]),
        helper.make_node("Mul", ["__ts_floor", "__loop_total"], ["__ts_off"]),
        helper.make_node("Sub", ["time_step", "__ts_off"], ["__ts_mod"]),
        helper.make_node("Add", ["__ts_mod", "__loop_start"], ["__ts_inner"]),
    ]
    for node in g.node:
        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)
        for idx, name in enumerate(new_node.input):
            if name == "time_step":
                new_node.input[idx] = "__ts_inner"
        nodes.append(new_node)
    # 只保留动作输出（第一个输出），其余运行时不需要
    first_out = g.output[0]
    out_vi = helper.make_tensor_value_info(first_out.name, first_out.type.tensor_type.elem_type,
                                           [d.dim_value for d in first_out.type.tensor_type.shape.dim])
    inputs = [
        helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, int(g.input[0].type.tensor_type.shape.dim[1].dim_value)]),
        helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1]),
    ]
    new_graph = helper.make_graph(nodes, g.name + "_loop", inputs, [out_vi],
                                  list(g.initializer) + consts)
    new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version
    onnx.checker.check_model(new_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(new_model, str(out))

    # 校验循环语义
    nj = (obs_dim - 6) // 5

    def make_obs():
        obs = np.zeros((1, obs_dim), np.float32)
        obs[0, 2 * nj + 2] = -1.0
        return obs

    def run_wrapper(sess, t):
        return sess.run(None, {"obs": make_obs(), "time_step": np.array([[t]], np.float32)})[0]

    w = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    orig = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])

    def run_orig(t):
        return orig.run(["actions"], {"obs": make_obs(), "time_step": np.array([[t]], np.float32)})[0]

    checks = [(0, start), (total - 1, start + total - 1), (total, start), (total + 5, start + 5)]
    for t, expect in checks:
        got = run_wrapper(w, t)
        ref = run_orig(expect)
        err = float(np.max(np.abs(got - ref)))
        print(f"  loop check t={t} -> {expect}: max_err={err:.2e}", flush=True)
        if err > 1e-5:
            raise RuntimeError("循环包装 ONNX 校验失败")
    print("wrapper ok ->", out, flush=True)


def ensure_mjcf(default_joints: dict[str, float], ref_root: tuple, ref_quat: tuple,
                out_dir: Path) -> Path:
    """为该资源生成专属 MJCF：绝对 meshdir + 首帧参考姿态 keyframe + 地面/光源。"""
    src = X2_XML.read_text(encoding="utf-8")
    model = mujoco.MjSpec.from_file(str(X2_XML)).compile()
    qpos = np.array(model.qpos0, dtype=np.float64)
    qpos[0:3] = ref_root
    qpos[3:7] = ref_quat
    for name, value in default_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MJCF 缺少关节 {name}")
        qpos[int(model.jnt_qposadr[jid])] = value

    meshdir = str((X2_XML.parent.parent / "meshes").resolve()).replace("\\", "/")
    text = src.replace('meshdir="./meshes"', f'meshdir="{meshdir}"')
    # X2 原 MJCF 没有地面与主光源（实机不需要）；mjswan 场景需要，否则渲染全黑。
    scene_extra = (
        "\n    <geom name=\"floor\" type=\"plane\" size=\"0 0 0.05\" pos=\"0 0 0\" "
        "rgba=\"0.34 0.34 0.34 1\" conaffinity=\"1\" condim=\"3\"/>\n"
        "    <light name=\"sun\" pos=\"0 0 4\" dir=\"0 0 -1\" directional=\"true\" "
        "castshadow=\"true\" diffuse=\"0.8 0.8 0.8\" ambient=\"0.25 0.25 0.25\"/>\n"
    )
    text = text.replace("<worldbody>", "<worldbody>" + scene_extra, 1)
    key = "<keyframe>\n    <key name=\"stand\" qpos=\"" + " ".join(f"{v:.6f}" for v in qpos) + "\"/>\n  </keyframe>\n"
    text = text.replace("</mujoco>", key + "</mujoco>")
    out = out_dir / "x2_mjswan.xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    # 自检：编译后 keyframe 存在、脚底接近地面
    spec2 = mujoco.MjSpec.from_file(str(out))
    m2 = spec2.compile()
    assert m2.nkey >= 1, "keyframe 未写入"
    d = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d, 0)
    mujoco.mj_forward(m2, d)
    feet = []
    for bname in ("left_ankle_roll_link", "right_ankle_roll_link"):
        b = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, bname)
        for g in range(m2.ngeom):
            if m2.geom_bodyid[g] == b and m2.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                feet.append(float(d.geom_xpos[g][2] - m2.geom_size[g][0]))
    print(f"mjcf -> {out}; foot spheres bottom z={min(feet):.4f} (keyframe root z={qpos[2]:.4f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?", default=str(ROOT / "assets/samples/x2/特技连踢/policy.onnx"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    onnx_path = Path(args.onnx)
    out_dir = Path(args.out) if args.out else CACHE_DIR / onnx_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(onnx_path)
    print("joints", len(meta["joint_names"]), "bodies", len(meta["body_names"]),
          "obs", meta["obs_dim"], "anchor", meta["anchor"], flush=True)
    ref = extract_reference(onnx_path, meta)
    model = mujoco.MjSpec.from_file(str(X2_XML)).compile()
    mapped_bodies = remap_body_names(meta["body_names"], model)

    root_pos = ref["body_pos_w"][0, 0].astype(np.float64)
    root_quat = ref["body_quat_w"][0, 0].astype(np.float64)
    root_quat = root_quat / (np.linalg.norm(root_quat) or 1.0)
    print("ref start root:", np.round(root_pos, 3), "quat:", np.round(root_quat, 3), flush=True)

    mjcf = ensure_mjcf(dict(zip(meta["joint_names"], ref["joint_pos"][0])), tuple(root_pos), tuple(root_quat), out_dir)
    npz_path = out_dir / "motion.npz"
    np.savez(
        npz_path,
        joint_pos=np.ascontiguousarray(ref["joint_pos"]),
        joint_vel=np.ascontiguousarray(ref["joint_vel"]),
        body_pos_w=np.ascontiguousarray(ref["body_pos_w"]),
        body_quat_w=np.ascontiguousarray(ref["body_quat_w"]),
        body_lin_vel_w=np.ascontiguousarray(ref["body_lin_vel_w"]),
        body_ang_vel_w=np.ascontiguousarray(ref["body_ang_vel_w"]),
        body_names=np.array(mapped_bodies),
        fps=np.array([FPS], dtype=np.float32),
    )
    wrapper = out_dir / "policy_loop.onnx"
    build_loop_onnx(onnx_path, wrapper, ref["start"], ref["total"], meta["obs_dim"])

    out_meta = {
        **meta,
        "kind": "x2",
        "name": onnx_path.parent.name,
        "body_names": mapped_bodies,
        "mjcf": str(mjcf),
        "motion_npz": str(npz_path),
        "policy_loop_onnx": str(wrapper),
        "start": ref["start"],
        "end": ref["end"],
        "total": ref["total"],
        "fps": FPS,
        "source_onnx": str(onnx_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("assets ->", out_dir, flush=True)
    print("  motion frames:", ref["joint_pos"].shape[0], flush=True)
    print("  npz size:", npz_path.stat().st_size, "wrapper size:", wrapper.stat().st_size, flush=True)


if __name__ == "__main__":
    sys.exit(main())
