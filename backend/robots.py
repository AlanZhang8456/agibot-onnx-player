# -*- coding: utf-8 -*-
"""机器人 3D 模型清单（来自 models/robots.json，由 tools/asset_prep 生成）。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .paths import MODELS_DIR


def _fallback_robots() -> List[Dict[str, Any]]:
    robots: List[Dict[str, Any]] = []
    candidates = [
        ("x2_ultra", "灵犀 X2 Ultra", os.path.join(MODELS_DIR, "x2", "x2_ultra.urdf")),
        ("a3_t2d5", "远征 A3 T2.5", os.path.join(MODELS_DIR, "a3", "a3_t2d5", "urdf", "model.urdf")),
    ]
    for rid, name, urdf in candidates:
        if os.path.isfile(urdf):
            rel = os.path.relpath(urdf, MODELS_DIR).replace(os.sep, "/")
            robots.append({
                "id": rid,
                "name": name,
                "urdf": "/models/" + rel,
                "rootLink": "base_link" if rid.startswith("x2") else "pelvis_link",
                "joints": [],
                "upAxis": "z",
            })
    return robots


def get_robots() -> List[Dict[str, Any]]:
    path = os.path.join(MODELS_DIR, "robots.json")
    robots: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            robots = data.get("robots", [])
        except Exception:  # noqa: BLE001
            robots = []
    if not robots:
        robots = _fallback_robots()
    # 保证 urdf 路径以 /models/ 开头
    for r in robots:
        urdf = r.get("urdf", "")
        if urdf and not urdf.startswith("/models/"):
            r["urdf"] = "/models/" + urdf.lstrip("/").replace(os.sep, "/")
    return robots


def model_for_robot(robot: str) -> str:
    return {"x2": "x2_ultra", "a3": "a3_t2d5"}.get(robot, "")
