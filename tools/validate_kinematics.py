# -*- coding: utf-8 -*-
"""运动学一致性校验：把播放轨迹的 root 位姿 + 关节角喂给 MuJoCo 正运动学，
检查脚落地是否合理（验证 root 位姿坐标系/四元数约定/关节名映射是否正确）。

用法（在 onnx-player 目录）：
    set PYTHONPATH=.pylibs
    python -X utf8 tools/validate_kinematics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import mujoco  # noqa: E402

from backend import motion_a3, motion_x2, onnx_meta as om  # noqa: E402

MODELS = os.path.join(PROJECT, "models")


def _quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def foot_heights(model, data, body_names):
    out = {}
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            out[name] = float(data.xpos[bid][2])
    return out


def check(model_path, track_frames, joint_names, root_names, robot):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    joint_qpos_adr = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        joint_qpos_adr[name] = (int(model.jnt_qposadr[j]), int(model.jnt_type[j]))
    dof_bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
    feet = [b for b in dof_bodies if b and ("ankle_roll" in b or "ankle_pitch" in b)]
    free_name = next((n for n in joint_qpos_adr if n and ("floating" in n or "free" in n)), None)
    if free_name is None:
        # X2/A3 的 free joint 名可能为 floating_base_joint / free_joint
        free_name = next((n for n in joint_qpos_adr if n is not None and joint_qpos_adr[n][1] == 0), None)
    print(f"  model={os.path.basename(model_path)} nq={model.nq} nv={model.nv} free={free_name} feet={feet}")
    adr_free = joint_qpos_adr.get(free_name, (0, 0))[0]

    n_frames = len(track_frames["rootPos"])
    step = max(1, n_frames // 200)
    min_feet, max_feet = [], []
    min_all = 1e9
    for i in range(0, n_frames, step):
        data.qpos[:] = 0.0
        rp = track_frames["rootPos"][i]
        rq = track_frames["rootQuat"][i]
        data.qpos[adr_free:adr_free + 3] = rp
        data.qpos[adr_free + 3:adr_free + 7] = _quat_wxyz_to_xyzw(rq)
        for k, jname in enumerate(joint_names):
            if jname in joint_qpos_adr:
                data.qpos[joint_qpos_adr[jname][0]] = track_frames["joints"][i][k]
        mujoco.mj_forward(model, data)
        fh = foot_heights(model, data, feet)
        if fh:
            vals = list(fh.values())
            min_feet.append(min(vals))
            max_feet.append(max(vals))
            min_all = min(min_all, min(vals))
    print(f"  采样 {len(min_feet)} 帧：脚部最低点 z ∈ [{min(min_feet):.3f}, {max(min_feet):.3f}]，"
          f"全局最低 {min_all:.3f} m")
    return min(min_feet), max(min_feet)


def main():
    ok = True
    # X2 示例
    x2_onnx = os.path.join(PROJECT, "assets", "samples", "x2", "特技连踢", "policy.onnx")
    if os.path.isfile(x2_onnx):
        meta = om.read_meta(x2_onnx)
        res = motion_x2.build_x2_tracks(x2_onnx, meta)
        tr = res["tracks"][0]
        print("[X2] 特技连踢")
        lo, hi = check(os.path.join(MODELS, "x2", "mjcf", "x2_ultra.xml"),
                       {"rootPos": tr["rootPos"], "rootQuat": tr["rootQuat"], "joints": tr["joints"]},
                       res["jointNames"], ("base_link", "pelvis"), "x2")
        if lo < -0.25 or lo > 0.25:
            print("  [WARN] X2 脚部最低点离地超过 ±0.25m，检查 root 坐标系约定")
            ok = False
        else:
            print("  [PASS] X2 运动学落地合理")
    # A3 示例
    a3_onnx = os.path.join(PROJECT, "assets", "samples", "a3", "太极4-v03", "data.onnx")
    if os.path.isfile(a3_onnx):
        meta = om.read_meta(a3_onnx)
        res = motion_a3.build_a3_tracks(a3_onnx, meta, include_policy=False)
        tr = res["tracks"][0]
        print("[A3] 太极4-v03")
        lo, hi = check(os.path.join(MODELS, "a3", "a3_t2d5", "mjcf", "a3_t2d5.xml"),
                       {"rootPos": tr["rootPos"], "rootQuat": tr["rootQuat"], "joints": tr["joints"]},
                       res["jointNames"], ("pelvis_link",), "a3")
        if lo < -0.25 or lo > 0.25:
            print("  [WARN] A3 脚部最低点离地超过 ±0.25m，检查 root/CSV 映射")
            ok = False
        else:
            print("  [PASS] A3 运动学落地合理")
    print("结论：", "通过" if ok else "有警告，请检查")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
