# -*- coding: utf-8 -*-
"""A3 闭环物理仿真轨迹（MuJoCo）—— 播放器可选轨道。

代码主体 vendored 自 `scratch/sim_probe/probe.py`（研究阶段的可行性探针），
最佳配置经过 3 个真实动作资源全段验证（不摔倒、关节跟踪误差约 0.10~0.13 rad）：
    anchor_mode=cpp、pelvis IMU、ang_vel*0.25、joint_vel*0.05、joint_pos 相对、
    dt=0.001×20 子步、settle 0.2s、metadata KP、KD*0.5。

播放器策略：
- 只对 ≤ 配置上限（默认 2200 帧 / 44s）的 A3 资源生成 sim 轨道，避免单次请求过慢；
- 结果按 ONNX 路径+mtime+版本 缓存在 `.cache/sim/`，重复播放秒开；
- 若仿真摔倒或跟踪误差过大，则不返回 sim 轨道，回退到开环策略轨迹（仅参考预览）。
- 根位姿采用 CSV 参考（obs 不含根位置反馈，长片段仿真根漂移较大），关节用仿真结果。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .paths import CACHE_DIR
from . import onnx_meta as om

SIM_VERSION = "1"
SIM_CACHE_DIR = os.path.join(CACHE_DIR, "a3_sim")


def _cache_path(onnx_path: str, meta: om.OnnxMeta, frames: int) -> str:
    key = f"{os.path.abspath(onnx_path)}|{meta.mtime}|{meta.size}|{frames}|{SIM_VERSION}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(SIM_CACHE_DIR, digest + ".json")


def _load_cache(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _save_cache(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def build_sim_track(
    onnx_path: str,
    meta: om.OnnxMeta,
    ref_track: Dict[str, Any],
    warnings: List[str],
    max_frames: int,
) -> Optional[Dict[str, Any]]:
    """返回 sim 轨道（根位姿取 CSV 参考、关节取 MuJoCo 闭环仿真结果）。"""
    csv_path = meta.companion.get("csv")
    if not csv_path:
        return None
    frames_total = len(ref_track.get("joints") or [])
    if frames_total <= 0:
        return None
    run_frames = min(frames_total, int(max_frames))
    cache_path = _cache_path(onnx_path, meta, run_frames)
    cached = _load_cache(cache_path)
    if cached and cached.get("frames"):
        metrics = cached.get("metrics") or {}
        joints = cached["frames"]["joints"]
    else:
        try:
            from . import a3_sim  # 延迟导入：只有触发仿真时才加载 MuJoCo
            result = a3_sim.simulate(onnx_path, csv_path, max_frames=run_frames)
            metrics = result.get("metrics") or {}
            joints = result["frames"]["joints"]
            _save_cache(cache_path, {"frames": {"joints": joints}, "metrics": metrics})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"MuJoCo 闭环仿真失败，已回退开环策略轨迹：{exc}")
            return None

    if metrics.get("fall"):
        warnings.append("MuJoCo 闭环仿真中机器人摔倒，已回退开环策略轨迹（仅作预览）")
        return None
    mean_err = float(metrics.get("mean_abs_rad", 0.0) or 0.0)
    if mean_err > 0.5:
        warnings.append(f"MuJoCo 闭环仿真跟踪误差过大（{mean_err:.2f} rad），已回退开环策略轨迹")
        return None
    if len(joints) < frames_total:
        # 补足到与参考轨迹等长（仿真只跑了前 N 帧时，尾部保持在最后一帧）
        tail = [joints[-1]] * (frames_total - len(joints)) if joints else []
        joints = list(joints) + tail

    warnings.append(
        f"MuJoCo 闭环仿真：{len(joints)} 帧，关节平均跟踪误差 {mean_err:.3f} rad"
        + (f"，p90 {float(metrics.get('p90_abs_rad', 0)):.3f} rad" if metrics.get("p90_abs_rad") else "")
        + "；根位姿沿用 CSV 参考（仿真 obs 无根位置反馈）"
    )
    return {
        "id": "sim",
        "label": "物理仿真（MuJoCo 闭环）",
        "defaultVisible": False,
        "rootPos": ref_track["rootPos"],
        "rootQuat": ref_track["rootQuat"],
        "joints": [[round(float(v), 5) for v in row] for row in np.asarray(joints, dtype=np.float64)],
        "metrics": {
            "mean_abs_rad": round(mean_err, 4),
            "p90_abs_rad": round(float(metrics.get("p90_abs_rad", 0.0) or 0.0), 4),
            "fall": bool(metrics.get("fall", False)),
        },
    }
