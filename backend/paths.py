# -*- coding: utf-8 -*-
"""项目路径与默认配置。"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
WEB_DIR = os.path.join(PROJECT_DIR, "web")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
VERSION = "0.1.0"


def _default_roots() -> List[Dict[str, str]]:
    candidates = [
        (os.path.join(ASSETS_DIR, "samples"), "内置示例（如有）"),
    ]
    roots = []
    for path, label in candidates:
        if os.path.isdir(path):
            roots.append({"path": path, "label": label})
    return roots


def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:  # noqa: BLE001
            cfg = {}
    roots = cfg.get("roots")
    if not roots:
        roots = _default_roots()
    cfg["roots"] = roots
    cfg.setdefault("port", 8766)
    cfg.setdefault("a3Sim", True)
    cfg.setdefault("a3SimMaxFrames", 2200)
    # 环境变量开关：批量自检/低配机器可关掉 MuJoCo 仿真轨道
    if os.environ.get("ONNX_PLAYER_NO_SIM", "").lower() in ("1", "true", "yes"):
        cfg["a3Sim"] = False
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def set_stdout_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
