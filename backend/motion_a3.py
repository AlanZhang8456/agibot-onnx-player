# -*- coding: utf-8 -*-
"""A3（远征 A3）ONNX 动作资源回放轨迹生成。

A3 的 ONNX 是"动作跟踪策略"：输入只有 obs[1,157]（29 DOF）或 obs[1,137]（25 DOF），
输出 actions → target_joint = action * action_scale + default_joint_pos。
参考动作不在 ONNX 内部，而在资源目录的配套 CSV（data.csv / 同名 csv）里：
    每行 = root_xyz(3) + root_quat_xyzw(4) + N 个关节角（50Hz）
    关节列顺序（按 default.yaml 组顺序）：
      左腿6、右腿6、腰3(yaw,roll,pitch)、左臂7、右臂7（=ONNX joint_names 的规范子集）
本模块生成两条轨迹：
  - ref    ：CSV 参考动作（默认显示）
  - policy ：把 CSV 参考喂给 ONNX 策略（开环近似，实验性，默认不显示）
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from . import onnx_meta as om
from .paths import load_config

# A3 关节的规范顺序（= default.yaml 的 limb 组顺序；CSV 列顺序）
_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]

# 超过该帧数（80s）不再生成开环策略轨迹（耗时/偏差考虑）
POLICY_TRACK_MAX_FRAMES = 4000


class UnsupportedResource(Exception):
    pass


def a3_reference_order(joint_names: List[str]) -> List[str]:
    """按 A3 CSV/default.yaml 的组顺序，返回过滤后的关节名列表。"""
    names = set(joint_names)
    order: List[str] = []
    for side in ("left", "right"):
        order += [f"{side}_{j}_joint" for j in _LEG]
    order += _WAIST
    for side in ("left", "right"):
        order += [f"{side}_{j}_joint" for j in _ARM]
    filtered = [n for n in order if n in names]
    return filtered


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _mat_to_quat_wxyz(m: np.ndarray) -> List[float]:
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return [w / n, x / n, y / n, z / n]


def load_csv_reference(csv_path: str, joint_names: List[str]) -> Dict[str, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
    n = len(joint_names)
    if raw.shape[1] not in (7 + n,):
        raise UnsupportedResource(
            f"CSV 列数 {raw.shape[1]} 与关节数 {n} 不匹配（期望 {7 + n}）"
        )
    order = a3_reference_order(joint_names)
    if len(order) != n:
        missing = [x for x in joint_names if x not in order]
        raise UnsupportedResource(f"无法把 CSV 关节列映射到 ONNX 关节顺序，缺少：{missing[:5]}")
    idx = [order.index(name) for name in joint_names]
    root_pos = raw[:, 0:3]
    root_quat_xyzw = raw[:, 3:7]
    joints_onnx_order = raw[:, 7:][:, idx]
    return {
        "root_pos": root_pos,
        "root_quat_xyzw": root_quat_xyzw,
        "joints": joints_onnx_order,
        "frames": raw.shape[0],
    }


def _build_ref_track(ref: Dict[str, np.ndarray]) -> Dict[str, Any]:
    root_pos = np.round(ref["root_pos"].astype(np.float64), 5).tolist()
    quats = np.array([_mat_to_quat_wxyz(_quat_xyzw_to_mat(np.asarray(q)))
                      for q in ref["root_quat_xyzw"]], dtype=np.float64)
    root_quat = np.round(quats, 5).tolist()
    joints = np.round(ref["joints"].astype(np.float64), 5).tolist()
    return {
        "id": "ref",
        "label": "参考动作（CSV）",
        "defaultVisible": True,
        "rootPos": root_pos,
        "rootQuat": root_quat,
        "joints": joints,
    }


def _build_policy_track(
    path: str,
    meta: om.OnnxMeta,
    ref: Dict[str, np.ndarray],
    ref_track: Dict[str, Any],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """按 unitree_rl_lab mimic deploy 的观测构造做开环推理（实验性）。"""
    n = len(meta.joint_names)
    default = np.asarray(meta.default_joint_pos, dtype=np.float32)
    scale = np.asarray(meta.action_scale, dtype=np.float32)
    if len(default) != n or len(scale) != n:
        warnings.append("ONNX metadata 缺少 default_joint_pos / action_scale，跳过策略轨迹")
        return None
    try:
        sess = om.get_session(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"ONNX 会话创建失败：{exc}")
        return None

    ref_pos = ref["joints"].astype(np.float32)
    root_quat = ref["root_quat_xyzw"].astype(np.float64)
    dt = om.POLICY_DT
    vel = np.gradient(ref_pos.astype(np.float64), dt, axis=0).astype(np.float32)

    prev_a = np.zeros(n, dtype=np.float32)
    anchor = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    targets = np.empty((ref_pos.shape[0], n), dtype=np.float32)
    total = ref_pos.shape[0]
    for k in range(total):
        R = _quat_xyzw_to_mat(root_quat[k])
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        if k + 1 < total:
            R2 = _quat_xyzw_to_mat(root_quat[k + 1])
            dR = R2 @ R.T
            cos_t = max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))
            theta = math.acos(cos_t)
            if theta > 1e-9:
                w_w = theta / (2.0 * math.sin(theta)) * np.array(
                    [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]
                )
            else:
                w_w = np.zeros(3)
            ang_vel = R.T @ w_w / dt
        else:
            ang_vel = np.zeros(3)
        obs = np.concatenate([
            ref_pos[k],
            vel[k],
            gravity.astype(np.float32),
            (ang_vel * 0.25).astype(np.float32),
            ref_pos[k] - default,
            vel[k] * 0.05,
            prev_a,
            anchor,
        ]).astype(np.float32).reshape(1, -1)
        act = sess.run(["actions"], {"obs": obs})[0][0].astype(np.float32)
        prev_a = act
        targets[k] = np.clip(act * scale + default, -4.0, 4.0)

    return {
        "id": "policy",
        "label": "ONNX 策略输出（开环近似）",
        "defaultVisible": False,
        "rootPos": ref_track["rootPos"],
        "rootQuat": ref_track["rootQuat"],
        "joints": np.round(targets.astype(np.float64), 5).tolist(),
    }


def build_a3_tracks(
    path: str,
    meta: om.OnnxMeta,
    include_policy: bool = True,
) -> Dict[str, Any]:
    if meta.robot != "a3" or meta.kind != "motion":
        raise UnsupportedResource("不是可回放的 A3 动作 ONNX 资源")
    if not meta.joint_names:
        raise UnsupportedResource("ONNX 缺少 joint_names metadata")
    csv_path = meta.companion.get("csv")
    if not csv_path:
        raise UnsupportedResource("找不到配套参考动作 CSV（data.csv / 同名 csv）")

    warnings: List[str] = []
    ref = load_csv_reference(csv_path, meta.joint_names)
    frames = int(ref["frames"])
    ref_track = _build_ref_track(ref)
    tracks = [ref_track]
    # 过长资源（>80s）不生成策略轨迹：开环推理耗时随帧数线性增长，且长序列偏差更大
    if include_policy and frames > POLICY_TRACK_MAX_FRAMES:
        warnings.append(
            f"资源较长（{frames} 帧 / {frames / om.FPS:.0f}s），已跳过 ONNX 策略轨迹（仅播放参考动作）"
        )
    elif include_policy:
        cfg = load_config()
        sim_track = None
        sim_max = int(cfg.get("a3SimMaxFrames", 2200))
        if cfg.get("a3Sim", True) and frames <= sim_max:
            try:
                from .motion_a3_sim import build_sim_track
                sim_track = build_sim_track(path, meta, ref_track, warnings, sim_max)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"MuJoCo 闭环仿真不可用：{exc}")
        if sim_track is not None:
            tracks.append(sim_track)
        else:
            policy = _build_policy_track(path, meta, ref, ref_track, warnings)
            if policy is not None:
                tracks.append(policy)
                warnings.append(
                    "策略轨迹为开环推理（用参考动作作为理想反馈），与真机/仿真闭环存在偏差，仅供预览"
                )
    if meta.time_step_total and meta.time_step_total != frames:
        warnings.append(f"ONNX time_step_total={meta.time_step_total} 与 CSV 帧数 {frames} 不一致，以 CSV 为准")

    return {
        "tracks": tracks,
        "fps": om.FPS,
        "duration": round(frames / om.FPS, 4),
        "frameCount": frames,
        "jointNames": list(meta.joint_names),
        "meta": {
            "obs_dim": meta.obs_dim,
            "num_actions": meta.num_actions,
            "time_step_total": meta.time_step_total,
            "csv": meta.companion.get("csv"),
            "anchor_body": meta.anchor_body,
            "task": meta.metadata.get("task", ""),
        },
        "warnings": warnings,
    }
