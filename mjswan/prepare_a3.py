# -*- coding: utf-8 -*-
"""A3 动作资源准备：ONNX + 配套 CSV → mjswan 用的 motion npz。

口径与 A3 PoC 一致：
- 参考轨迹 = CSV（root xyz + quat xyzw + 29 关节角，50Hz）
- npz 存全模型 body 的 xpos/xquat 与有限差分线/角速度（mjswan 按 body_names 选取 16 个跟踪 body）
- meta.json 汇总策略关节顺序、default/scale/kp/kd、anchor、body_names
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnx

import os

ROOT = Path(os.environ.get("ONNX_PLAYER_ROOT") or str(Path(__file__).resolve().parents[1]))
A3_XML = ROOT / "models/a3/a3_t2d5/mjcf/a3_t2d5.xml"
FPS = 50.0

LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
CSV_ORDER = ([f"left_{j}_joint" for j in LEG] + [f"right_{j}_joint" for j in LEG]
             + WAIST
             + [f"left_{j}_joint" for j in ARM] + [f"right_{j}_joint" for j in ARM])


def floats(text):
    return [float(x) for x in str(text).replace("\n", ",").split(",") if x.strip()]


def load_meta(onnx_path: Path) -> dict:
    m = onnx.load(str(onnx_path), load_external_data=False)
    md = {kv.key: kv.value for kv in m.metadata_props}
    obs_dim = int(m.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    return {
        "kind": "a3",
        "obs_dim": obs_dim,
        "joint_names": [x.strip() for x in md["joint_names"].split(",") if x.strip()],
        "body_names": [x.strip() for x in md["body_names"].split(",") if x.strip()],
        "anchor": md.get("anchor_body_name", "pelvis_link"),
        "default": floats(md["default_joint_pos"]),
        "scale": floats(md["action_scale"]),
        "kp": floats(md["joint_stiffness"]),
        "kd": floats(md["joint_damping"]),
        "mjcf": str(A3_XML),
        "fps": FPS,
    }


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def prepare(onnx_path: Path, csv_path: Path, out_dir: Path) -> dict:
    meta = load_meta(onnx_path)
    joint_names = meta["joint_names"]
    model = mujoco.MjModel.from_xml_path(str(A3_XML))
    nbody = model.nbody
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
                  for i in range(nbody)]

    raw = np.loadtxt(str(csv_path), delimiter=",", dtype=np.float64, ndmin=2)
    T = raw.shape[0]
    # CSV 列顺序 = 标准分组顺序（左右腿 12 + 腰 3 + 左右臂），25-DOF 时按策略关节名过滤
    canonical = []
    for side in ("left", "right"):
        canonical += [f"{side}_{j}_joint" for j in LEG]
    canonical += WAIST
    for side in ("left", "right"):
        canonical += [f"{side}_{j}_joint" for j in ARM]
    usable = [n for n in canonical if n in joint_names]
    if len(usable) != len(joint_names):
        missing = [n for n in joint_names if n not in usable]
        raise RuntimeError(f"CSV 参考顺序里缺少关节：{missing}")
    if len(usable) != raw.shape[1] - 7:
        raise RuntimeError(f"CSV 列数 {raw.shape[1] - 7} 与策略关节数 {len(usable)} 不一致")
    csv_idx = [usable.index(name) for name in joint_names]
    joint_pos = raw[:, 7:][:, csv_idx].astype(np.float32)
    joint_vel = np.gradient(joint_pos.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)

    data = mujoco.MjData(model)
    qpos0 = np.array(model.qpos0, dtype=np.float64)
    jadr = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jadr[name] = int(model.jnt_qposadr[jid])

    body_pos = np.zeros((T, nbody, 3), dtype=np.float64)
    body_quat = np.zeros((T, nbody, 4), dtype=np.float64)
    for t in range(T):
        data.qpos[:] = qpos0
        data.qpos[0:3] = raw[t, 0:3]
        data.qpos[3:7] = quat_xyzw_to_wxyz(raw[t, 3:7])
        for i, name in enumerate(joint_names):
            data.qpos[jadr[name]] = joint_pos[t, i]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos
        body_quat[t] = data.xquat

    dt = 1.0 / FPS
    body_lin_vel = np.gradient(body_pos, dt, axis=0)
    body_ang_vel = np.zeros_like(body_pos)
    for t in range(T):
        t0, t1 = max(0, t - 1), min(T - 1, t + 1)
        span = (t1 - t0) * dt
        for b in range(nbody):
            dq = quat_mul(body_quat[t1, b], quat_conj(body_quat[t0, b]))
            w = dq[0]
            if w < 0:
                dq, w = -dq, -w
            w = min(1.0, max(-1.0, w))
            angle = 2.0 * np.arccos(w)
            s = np.sqrt(max(1e-12, 1.0 - w * w))
            axis = dq[1:] / s if s > 1e-9 else np.zeros(3)
            body_ang_vel[t, b] = axis * angle / span

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "motion.npz"
    np.savez(
        npz_path,
        joint_pos=np.ascontiguousarray(joint_pos),
        joint_vel=np.ascontiguousarray(joint_vel),
        body_pos_w=np.ascontiguousarray(body_pos.astype(np.float32)),
        body_quat_w=np.ascontiguousarray(body_quat.astype(np.float32)),
        body_lin_vel_w=np.ascontiguousarray(body_lin_vel.astype(np.float32)),
        body_ang_vel_w=np.ascontiguousarray(body_ang_vel.astype(np.float32)),
        body_names=np.array(body_names),
        fps=np.array([FPS], dtype=np.float32),
    )
    meta.update({
        "motion_npz": str(npz_path),
        "source_onnx": str(onnx_path),
        "source_csv": str(csv_path),
        "frames": int(T),
        "duration": round(T / FPS, 3),
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = prepare(Path(args.onnx), Path(args.csv), Path(args.out))
    print(f"A3 prepared: {args.out} frames={meta['frames']} duration={meta['duration']}s")


if __name__ == "__main__":
    sys.exit(main())
