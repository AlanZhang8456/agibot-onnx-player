# -*- coding: utf-8 -*-
"""资源库扫描：在配置的根目录里找 ONNX 动作资源，识别机器人/类型/配套文件。"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from . import onnx_meta as om
from .paths import load_config

MAX_DEPTH = 6
MAX_FILES = 2000
SCAN_CACHE_TTL = 120.0

_scan_cache: Dict[str, Dict[str, Any]] = {}
_scan_lock = threading.Lock()


def _skippable(dir_name: str) -> bool:
    name = dir_name.lower()
    if name.startswith(".") and name not in (".", ".."):
        return True
    if name in om.SKIP_DIRS:
        return True
    if name.startswith("$"):  # installers 临时目录
        return True
    return False


def _find_onnx(root: str) -> List[str]:
    found: List[str] = []
    root = os.path.abspath(root)
    root_depth = root.rstrip("\\/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip("\\/").count(os.sep) - root_depth
        if depth >= MAX_DEPTH:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if not _skippable(d)]
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(".onnx"):
                found.append(os.path.join(dirpath, name))
                if len(found) >= MAX_FILES:
                    return found
    return found


def scan_root(root: str, force: bool = False) -> Dict[str, Any]:
    """扫描单个根目录，返回 {root, entries, errors, scan_ms}。"""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return {"root": root, "entries": [], "errors": ["目录不存在"], "scan_ms": 0}
    now = time.time()
    with _scan_lock:
        cached = _scan_cache.get(root)
    if cached and not force and now - cached["time"] < SCAN_CACHE_TTL:
        return cached["data"]

    t0 = time.time()
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    paths = _find_onnx(root)

    def build(path: str) -> Dict[str, Any]:
        try:
            return om.library_entry(path, root)
        except om.MetaError as exc:
            return {
                "path": path,
                "root": root,
                "relPath": os.path.relpath(path, root),
                "name": os.path.basename(path),
                "dir": os.path.dirname(path),
                "size": os.path.getsize(path) if os.path.isfile(path) else 0,
                "mtime": os.path.getmtime(path) if os.path.isfile(path) else 0,
                "robot": "unknown",
                "kind": "unknown",
                "playable": False,
                "note": str(exc),
                "companion": {},
                "meta": {},
            }

    workers = min(8, max(1, len(paths)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(build, paths):
            entries.append(result)
    om.flush_disk_cache()

    # 排序：可播放 > 机器人 x2/a3 > 名称
    def sort_key(e: Dict[str, Any]):
        return (0 if e.get("playable") else 1, {"x2": 0, "a3": 1}.get(e.get("robot", "unknown"), 2),
                e.get("name", ""))

    entries.sort(key=sort_key)
    data = {
        "root": root,
        "entries": entries,
        "errors": errors,
        "scan_ms": int((time.time() - t0) * 1000),
    }
    with _scan_lock:
        if len(_scan_cache) > 32:
            _scan_cache.clear()
        _scan_cache[root] = {"time": now, "data": data}
    return data


def get_roots() -> List[Dict[str, str]]:
    cfg = load_config()
    out = []
    for item in cfg.get("roots", []):
        if isinstance(item, str):
            out.append({"path": item, "label": item})
        else:
            out.append({"path": item.get("path", ""), "label": item.get("label") or item.get("path", "")})
    return [r for r in out if r["path"]]


def scan_library(root: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    roots = get_roots()
    if root:
        targets = [os.path.abspath(root)]
    else:
        targets = [r["path"] for r in roots]
    all_entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    root_infos = []
    for target in targets:
        res = scan_root(target, force=force)
        root_infos.append({"path": target, "scan_ms": res["scan_ms"], "count": len(res["entries"])})
        all_entries.extend(res["entries"])
        errors.extend(res["errors"])

    def sort_key(e: Dict[str, Any]):
        return (0 if e.get("playable") else 1, {"x2": 0, "a3": 1}.get(e.get("robot", "unknown"), 2),
                e.get("name", ""))

    all_entries.sort(key=sort_key)
    return {
        "ok": True,
        "roots": roots,
        "rootScans": root_infos,
        "entries": all_entries,
        "errors": errors,
    }
