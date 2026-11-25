# -*- coding: utf-8 -*-
"""X2（灵犀 X2）ONNX 动作资源回放轨迹生成。

契约来源：官方 SDK `extra/x2_rl_deploy`（C++ 部署示例）
- 输入：obs[1,151]（29 关节版）+ time_step[1,1]（策略步计数，float）
- 输出：actions / joint_pos / joint_vel / body_pos_w[1,14,3] / body_quat_w[1,14,4] ……
  新导出额外有 motion_phase[1,2] = [起始步, 结束步]。
- obs 组成（与 motion_control_node.cc 的 RlDefault 一致）：
  command_dof_pos(29, 上一步参考 joint_pos) + command_dof_vel(29, 上一步参考 joint_vel)
  + projected_gravity(3) + base_ang_vel(3) + dof_pos_rel(29) + dof_vel(29)*0.05 + last_action(29)
- 参考动作分支由 time_step 驱动，不依赖反馈；因此可以直接读取 ONNX 输出的
  joint_pos + body_pos_w/body_quat_w 得到"ONNX 生成的动作"。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from . import onnx_meta as om

GRAVITY_UP = np.array([0.0, 0.0, -1.0], dtype=np.float32)


class UnsupportedResource(Exception):
    pass


def _collect(name: str, outs: List[str], res: List[np.ndarray]) -> np.ndarray:
    return res[outs.index(name)]


def build_x2_tracks(path: str, meta: om.OnnxMeta, include_policy: bool = True) -> Dict[str, Any]:
    if meta.robot != "x2" or meta.kind != "motion":
        raise UnsupportedResource("不是可回放的 X2 动作 ONNX 资源")
    if not meta.joint_names:
        raise UnsupportedResource("ONNX 缺少 joint_names metadata")
    n = len(meta.joint_names)
    if n not in (25, 29):
        raise UnsupportedResource(f"暂不支持的关节数：{n}")

    sess = om.get_session(path)
    outs = [o.name for o in sess.get_outputs()]
    need = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]
    missing = [x for x in need if x not in outs]
    if missing:
        raise UnsupportedResource(f"ONNX 输出缺少：{', '.join(missing)}")

    # probe：取 motion_phase（新导出才有），确定参考动作的步范围
    probe_obs = np.zeros((1, 5 * n + 6), dtype=np.float32)
    probe_obs[0, 2 * n + 2] = -1.0  # projected gravity (0,0,-1)
    feeds = {"obs": probe_obs}
    has_ts = any(i["name"] == "time_step" for i in meta.inputs)
    if has_ts:
        feeds["time_step"] = np.array([[0.0]], dtype=np.float32)
    start, end = 0, max(meta.time_step_total - 1, 0)
    phase = None
    if "motion_phase" in outs:
        try:
            res = sess.run(["motion_phase", "joint_pos"], feeds)
            phase = res[0][0]
            start = int(round(float(phase[0])))
            end = int(round(float(phase[1])))
        except Exception:  # noqa: BLE001
            phase = None
    if end <= start:
        if meta.time_step_total:
            end = meta.time_step_total - 1
        else:
            raise UnsupportedResource("无法确定动作起止步（缺少 motion_phase 与 time_step_total）")
    total_steps = end - start + 1

    prev_jp = np.zeros(n, dtype=np.float32)
    prev_jv = np.zeros(n, dtype=np.float32)
    prev_a = np.zeros(n, dtype=np.float32)

    root_pos_arr = np.empty((total_steps, 3), dtype=np.float32)
    root_quat_arr = np.empty((total_steps, 4), dtype=np.float32)
    joints_arr = np.empty((total_steps, n), dtype=np.float32)
    warnings: List[str] = []

    out_names = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]
    get_phase = "motion_phase" in outs

    for idx, step in enumerate(range(start, end + 1)):
        obs = np.concatenate([
            prev_jp,
            prev_jv,
            GRAVITY_UP,
            np.zeros(3, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            prev_a,
        ]).astype(np.float32).reshape(1, -1)
        feeds = {"obs": obs}
        if has_ts:
            feeds["time_step"] = np.array([[float(step)]], dtype=np.float32)
        names = list(out_names)
        if get_phase:
            names.append("motion_phase")
        try:
            res = sess.run(names, feeds)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedResource(f"ONNX 推理失败：{exc}") from exc

        d = dict(zip(names, res))
        prev_a = np.asarray(d["actions"][0], dtype=np.float32)
        jp = np.asarray(d["joint_pos"][0], dtype=np.float32)
        jv = np.asarray(d["joint_vel"][0], dtype=np.float32)
        bp = np.asarray(d["body_pos_w"][0], dtype=np.float32)
        bq = np.asarray(d["body_quat_w"][0], dtype=np.float32)
        prev_jp, prev_jv = jp.copy(), jv.copy()

        root_pos_arr[idx] = bp[0] if bp.ndim == 2 else bp[:3]
        # ONNX/IsaacLab 四元数为 wxyz，归一化后输出
        q = np.asarray(bq[0] if bq.ndim == 2 else bq[:4], dtype=np.float64)
        norm = np.linalg.norm(q)
        root_quat_arr[idx] = q / norm if norm > 1e-8 else np.array([1, 0, 0, 0])
        joints_arr[idx] = jp

    root_pos = np.round(root_pos_arr.astype(np.float64), 5).tolist()
    root_quat = np.round(root_quat_arr.astype(np.float64), 5).tolist()
    joints = np.round(joints_arr.astype(np.float64), 5).tolist()

    if not joints:
        raise UnsupportedResource("没有生成任何帧")

    track = {
        "id": "ref",
        "label": "ONNX 参考动作" if "motion_phase" in outs else "ONNX 动作",
        "defaultVisible": True,
        "rootPos": root_pos,
        "rootQuat": root_quat,
        "joints": joints,
    }

    # 合理性检查：root 高度、关节角幅度
    zs = np.array([p[2] for p in root_pos])
    js = np.array(joints)
    if zs.max() - zs.min() > 1.2 or js.max() > 3.6 or js.min() < -3.6:
        warnings.append("参考轨迹幅度异常，请核对 robot_type 与模型版本")
    if phase is not None:
        warnings.append(f"ONNX 动作步范围 [{start}, {end}]（共 {total_steps} 步 / {total_steps / om.FPS:.2f}s）")

    return {
        "tracks": [track],
        "fps": om.FPS,
        "duration": round(total_steps / om.FPS, 4),
        "frameCount": total_steps,
        "jointNames": list(meta.joint_names),
        "meta": {
            "obs_dim": meta.obs_dim,
            "num_actions": n,
            "time_step_total": meta.time_step_total,
            "phase": [start, end] if phase is not None else None,
            "anchor_body": meta.anchor_body,
        },
        "warnings": warnings,
    }
