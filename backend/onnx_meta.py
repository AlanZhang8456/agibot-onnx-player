# -*- coding: utf-8 -*-
"""ONNX 元数据读取、资源分类与配套文件发现。

分类依据（详见 docs/SDK_AND_RESOURCES.md）：
- X2 灵创动作资源：输入 obs[1,151]/[1,131] + time_step[1,1]，
  输出含 actions/joint_pos/joint_vel/body_pos_w/body_quat_w（新导出另有 motion_phase 等）。
- A3 动作资源：输入 obs[1,157]/[1,137]，输出 actions（29/25），参考动作在配套 CSV。
- 其它（locomotion 速度策略等）：只展示元数据，不生成播放轨迹。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort

from .paths import CACHE_DIR, VERSION

FPS = 50.0
POLICY_DT = 0.02

MEDIA_EXTS = (".mp4", ".wav", ".mp3", ".gif", ".avi", ".mov", ".png", ".jpg", ".jpeg")
SKIP_DIRS = {
    "node_modules", "__pycache__", "__MACOSX", ".git", ".venv", "site-packages",
    ".deps", ".pylibs", "dist", "build", ".dart_tool", ".idea",
}


class MetaError(Exception):
    """无法读取/识别 ONNX 文件。"""


@dataclass
class OnnxMeta:
    path: str
    size: int
    mtime: float
    robot: str = "unknown"          # x2 | a3 | unknown
    kind: str = "unknown"           # motion | locomotion | unknown
    playable: bool = False
    note: str = ""
    obs_dim: int = 0
    num_actions: int = 0
    time_step_total: int = 0
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    joint_names: List[str] = field(default_factory=list)
    default_joint_pos: List[float] = field(default_factory=list)
    action_scale: List[float] = field(default_factory=list)
    joint_stiffness: List[float] = field(default_factory=list)
    joint_damping: List[float] = field(default_factory=list)
    body_names: List[str] = field(default_factory=list)
    anchor_body: str = ""
    companion: Dict[str, Any] = field(default_factory=dict)
    resource_config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.time_step_total:
            return self.time_step_total / FPS
        csv = self.companion.get("csv")
        if csv:
            rows = csv_rows(csv)
            if rows:
                return rows / FPS
        return 0.0

    def summary(self) -> Dict[str, Any]:
        out = {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "time_step_total": self.time_step_total,
            "duration": round(self.duration, 3),
            "task": self.metadata.get("task", ""),
        }
        rc = self.resource_config or {}
        if "robot_type" in rc:
            out["robot_type"] = rc["robot_type"]
        if "total_time" in rc:
            out["yaml_total_time"] = rc["total_time"]
        if "common.policy_name" in rc:
            out["policy_name"] = rc["common.policy_name"]
        return out


_meta_cache: Dict[str, tuple] = {}
_meta_lock = threading.Lock()

# 磁盘缓存：跨进程复用 ONNX 解析结果（扫描 200+ 文件时避免重复解析）
_disk_cache: Optional[Dict[str, Dict[str, Any]]] = None
_disk_dirty = False
_disk_lock = threading.Lock()
_DISK_PATH = os.path.join(CACHE_DIR, "onnx_meta.json")

_session_cache: Dict[str, ort.InferenceSession] = {}
_session_lock = threading.Lock()
_SESSION_LIMIT = 4


def _load_disk_cache() -> Dict[str, Dict[str, Any]]:
    global _disk_cache
    with _disk_lock:
        if _disk_cache is not None:
            return _disk_cache
        data: Dict[str, Dict[str, Any]] = {}
        try:
            with open(_DISK_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("version") == VERSION:
                data = raw.get("items", {})
        except Exception:  # noqa: BLE001
            data = {}
        _disk_cache = data
        return data


def _mark_dirty() -> None:
    global _disk_dirty
    _disk_dirty = True


def flush_disk_cache() -> None:
    global _disk_dirty
    with _disk_lock:
        if not _disk_dirty or _disk_cache is None:
            return
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = _DISK_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": VERSION, "items": _disk_cache}, f, ensure_ascii=False)
            os.replace(tmp, _DISK_PATH)
            _disk_dirty = False
        except Exception:  # noqa: BLE001
            pass



def _shape_of(value_info) -> List[int]:
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append(-1)
    return dims


def _dtype_of(value_info) -> str:
    return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type).lower()


def _floats(text: str) -> List[float]:
    out = []
    for part in str(text).replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def _ints(text: str) -> List[int]:
    return [int(round(v)) for v in _floats(text)]


def _split_names(text: str) -> List[str]:
    return [p.strip() for p in str(text).replace("\n", ",").split(",") if p.strip()]


def _parse_resource_yaml(path: Optional[str]) -> Dict[str, Any]:
    """轻量解析资源配套 yaml 的关键字段（不依赖 PyYAML）。

    - X2 x2a.yaml: robot_type / total_time
    - A3 default.yaml: common.policy_name / obs.num_obs / actions.num_actions / common.decimation
    """
    out: Dict[str, Any] = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return out
    stack: List[tuple] = []  # (indent, key)
    for raw in lines:
        line = raw.rstrip("\n\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.split(" #")[0].strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = ".".join([k for _, k in stack] + [key])
        if val:
            if val.startswith("[") or val.startswith("{"):
                continue
            try:
                out[full] = int(val)
            except ValueError:
                try:
                    out[full] = float(val)
                except ValueError:
                    out[full] = val.strip("'\"")
        else:
            stack.append((indent, key))
    keep = {}
    for k, v in out.items():
        if k in ("robot_type", "total_time", "common.policy_name", "common.decimation",
                 "obs.num_obs", "obs.num_obs_history", "actions.num_actions", "actions.num_actions") \
                or k.endswith(".num_obs") or k.endswith(".num_actions"):
            keep[k] = v
    return keep


def _find_csv(onnx_path: str) -> Optional[str]:
    """为 A3 资源寻找配套参考动作 CSV。"""
    d = os.path.dirname(onnx_path)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    candidates: List[str] = []
    # 1) 同名 CSV（taiji4_v03-hb-0508-v1.onnx -> .csv）
    p = os.path.join(d, stem + ".csv")
    if os.path.isfile(p):
        candidates.append(p)
    # 2) data.csv（linkcraft/A3动作包）
    p = os.path.join(d, "data.csv")
    if os.path.isfile(p):
        candidates.append(p)
    # 3) 目录里其它 csv（按体积从大到小，宁大勿小）
    try:
        others = [
            os.path.join(d, n) for n in os.listdir(d)
            if n.lower().endswith(".csv") and os.path.isfile(os.path.join(d, n))
        ]
        others.sort(key=lambda x: -os.path.getsize(x))
        candidates.extend(others)
    except OSError:
        pass
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if _csv_looks_like_reference(c):
            return c
    return None


def _csv_looks_like_reference(path: str) -> bool:
    """粗略校验 CSV 是否为 'root(7)+N 关节角' 的参考动作。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline()
        parts = [p for p in first.replace(";", ",").split(",") if p.strip()]
        if len(parts) < 8:
            return False
        float(parts[0])
        float(parts[-1])
        return True
    except Exception:
        return False


def csv_rows(path: str) -> int:
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header_skipped = False
            for line in f:
                if not line.strip():
                    continue
                if not header_skipped:
                    try:
                        float(line.split(",")[0])
                    except ValueError:
                        header_skipped = True
                        continue
                    header_skipped = True
                n += 1
    except OSError:
        return 0
    return n


def _find_yaml(onnx_path: str) -> Optional[str]:
    d = os.path.dirname(onnx_path)
    for name in ("x2a.yaml", "default.yaml", "a3a.yaml", "config.yaml"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    try:
        for n in sorted(os.listdir(d)):
            if n.lower().endswith((".yaml", ".yml")):
                return os.path.join(d, n)
    except OSError:
        pass
    return None


def _find_media(onnx_path: str) -> List[str]:
    d = os.path.dirname(onnx_path)
    out = []
    try:
        for n in sorted(os.listdir(d)):
            if n.lower().endswith(MEDIA_EXTS) and os.path.isfile(os.path.join(d, n)):
                out.append(n)
    except OSError:
        pass
    return out


def guess_display_name(onnx_path: str) -> str:
    d = os.path.dirname(os.path.abspath(onnx_path))
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    base = os.path.basename(d)
    # 资源目录常见层级：.../<名字>__linkcraft_resource_onnx_<id>/<版本>/unzip/xxx.onnx
    #                    .../A3动作包/<动作名>/data.onnx
    #                    .../dances/<作品>/09/xxx.onnx
    junk = {"unzip", "dances", "motion", "mjcf", "urdf", "meshes", "data", "policy", "resource",
            "assets", "config", "output", "results", ""}
    _version_re = __import__("re").compile(r"^(\d+(\.\d+)+|\d{1,2})$")
    depth = 0
    while base.lower() in junk or _version_re.match(base):
        d = os.path.dirname(d)
        new_base = os.path.basename(d)
        if not new_base or new_base == d or depth > 5:
            break
        base = new_base
        depth += 1
    name = base
    import re as _re
    linkcraft = _re.match(
        r"^(.*?)linkcraft_resource(?:_onnx)?_([0-9A-Za-z]{26})-?(.*)$", name)
    if linkcraft:
        prefix, _, suffix = linkcraft.groups()
        name = prefix.strip(" _-.") or suffix.strip(" _-.")
    else:
        for token in ("__linkcraft_resource_onnx_", "__linkcraft_resource_"):
            if token in name:
                prefix, suffix = name.split(token, 1)
                name = prefix.strip(" _-.") or suffix.strip(" _-.")
                break
    # 去掉 "UUID-名字" 前缀
    m = _re.match(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}-(.+)$",
        name)
    if m:
        name = m.group(1)
    name = name.strip(" _-.")
    if not name or name.lower() in junk or _version_re.match(name):
        name = stem
    # 容器目录名（如 A3 的 510c）+ 具体文件 stem，避免一堆同名条目
    if stem.lower() not in ("policy", "data", "motion") and name != stem:
        name = f"{name} · {stem}"
    return name


def read_meta(path: str) -> OnnxMeta:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise MetaError(f"文件不存在：{path}")
    st = os.stat(path)
    key = f"{path}|{st.st_mtime_ns}|{st.st_size}"
    with _meta_lock:
        hit = _meta_cache.get(key)
        if hit is not None:
            return hit
    disk = _load_disk_cache()
    if key in disk:
        try:
            meta = OnnxMeta(**disk[key])
            with _meta_lock:
                _meta_cache[key] = meta
            return meta
        except Exception:  # noqa: BLE001
            pass

    meta = OnnxMeta(path=path, size=st.st_size, mtime=st.st_mtime)
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as exc:  # noqa: BLE001
        raise MetaError(f"ONNX 解析失败：{exc}") from exc

    meta.inputs = [
        {"name": v.name, "shape": _shape_of(v), "dtype": _dtype_of(v)}
        for v in model.graph.input
    ]
    meta.outputs = [
        {"name": v.name, "shape": _shape_of(v), "dtype": _dtype_of(v)}
        for v in model.graph.output
    ]
    meta.metadata = {kv.key: kv.value for kv in model.metadata_props}

    md = meta.metadata
    if "joint_names" in md:
        meta.joint_names = _split_names(md["joint_names"])
    if "default_joint_pos" in md:
        meta.default_joint_pos = _floats(md["default_joint_pos"])
    if "action_scale" in md:
        meta.action_scale = _floats(md["action_scale"])
    if "joint_stiffness" in md:
        meta.joint_stiffness = _floats(md["joint_stiffness"])
    if "joint_damping" in md:
        meta.joint_damping = _floats(md["joint_damping"])
    if "body_names" in md:
        meta.body_names = _split_names(md["body_names"])
    meta.anchor_body = md.get("anchor_body_name", "")
    if "time_step_total" in md:
        try:
            meta.time_step_total = int(round(float(md["time_step_total"])))
        except ValueError:
            pass

    input_names = {i["name"] for i in meta.inputs}
    output_names = {o["name"] for o in meta.outputs}
    obs_shape = next((i["shape"] for i in meta.inputs if i["name"] == "obs"), [])
    meta.obs_dim = obs_shape[-1] if obs_shape else 0
    action_shape = next((o["shape"] for o in meta.outputs if o["name"] == "actions"), [])
    num_actions = action_shape[-1] if action_shape else 0
    meta.num_actions = max(num_actions, 0)

    meta.companion = {
        "yaml": _find_yaml(path),
        "csv": None,
        "media": _find_media(path),
    }
    meta.resource_config = _parse_resource_yaml(meta.companion.get("yaml"))
    if str(meta.resource_config.get("robot_type", "")).lower() == "x2_ultra_new":
        meta.warnings.append(
            "资源标注为 X2 Ultra 焕新版（x2_ultra_new），当前 3D 模型为 X2 Ultra v1.3.0，外形/关节零点可能有细微差异"
        )

    # ---------- 分类 ----------
    has_ts = "time_step" in input_names
    n_from_obs = (meta.obs_dim - 12) / 5.0
    n_a3 = int(round(n_from_obs)) if abs(n_from_obs - round(n_from_obs)) < 1e-6 else 0
    if has_ts and meta.obs_dim in (151, 131) and "joint_pos" in output_names:
        # X2 灵创动作资源：obs + time_step -> 参考动作输出
        meta.robot = "x2"
        meta.kind = "motion"
        meta.playable = True
        meta.num_actions = meta.num_actions or len(meta.joint_names)
        if not meta.joint_names:
            meta.warnings.append("ONNX metadata 缺少 joint_names，无法映射关节")
            meta.playable = False
        if not meta.time_step_total and "motion_phase" not in output_names:
            meta.warnings.append("缺少 time_step_total，无法确定动作长度")
            meta.playable = False
    elif (not has_ts) and n_a3 in (29, 25) and "actions" in output_names:
        # A3 动作资源：obs[1,5N+12] -> actions[N]，参考动作在配套 CSV
        meta.robot = "a3"
        meta.kind = "motion"
        meta.num_actions = n_a3
        meta.companion["csv"] = _find_csv(path)
        if meta.companion["csv"]:
            meta.playable = True
            if not meta.time_step_total:
                meta.time_step_total = csv_rows(meta.companion["csv"])
        else:
            meta.playable = False
            meta.note = "缺少配套参考动作 CSV（data.csv / 同名 csv），只能查看元数据"
        if not meta.joint_names:
            meta.note = "ONNX 缺少 joint_names metadata，无法映射关节"
            meta.playable = False
    elif (not has_ts) and meta.obs_dim >= 60:
        meta.kind = "locomotion"
        meta.robot = "x2" if (meta.obs_dim in (500, 480, 235, 208, 104, 100) or
                              meta.metadata.get("metadata_contract", "").startswith("agibot_x2")) else "unknown"
        meta.note = "速度/步态类策略，需要物理仿真闭环，播放器暂不支持直接回放"
    else:
        meta.kind = "unknown"
        if meta.obs_dim and not meta.joint_names:
            meta.note = "无法识别的 ONNX 输入/输出契约"

    meta.warnings = list(dict.fromkeys(meta.warnings))
    with _meta_lock:
        if len(_meta_cache) > 512:
            _meta_cache.clear()
        _meta_cache[key] = meta
    # 写磁盘缓存
    try:
        disk = _load_disk_cache()
        disk[key] = asdict(meta)
        _mark_dirty()
    except Exception:  # noqa: BLE001
        pass
    return meta


def get_session(path: str) -> ort.InferenceSession:
    path = os.path.abspath(path)
    with _session_lock:
        hit = _session_cache.get(path)
        if hit is not None:
            return hit
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    with _session_lock:
        if len(_session_cache) >= _SESSION_LIMIT:
            _session_cache.pop(next(iter(_session_cache)))
        _session_cache[path] = sess
    return sess


def inspect(path: str) -> Dict[str, Any]:
    meta = read_meta(path)
    return {
        "ok": True,
        "path": meta.path,
        "name": guess_display_name(meta.path),
        "size": meta.size,
        "mtime": meta.mtime,
        "robot": meta.robot,
        "kind": meta.kind,
        "playable": meta.playable,
        "note": meta.note,
        "inputs": meta.inputs,
        "outputs": meta.outputs,
        "metadata": {
            k: (v if len(v) <= 4000 else v[:4000] + "…(截断)")
            for k, v in meta.metadata.items()
        },
        "companion": meta.companion,
        "resource_config": meta.resource_config,
        "summary": meta.summary(),
        "warnings": meta.warnings,
    }


def library_entry(path: str, root: str) -> Dict[str, Any]:
    meta = read_meta(path)
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        rel = path
    return {
        "path": meta.path,
        "root": root,
        "relPath": rel,
        "name": guess_display_name(meta.path),
        "dir": os.path.dirname(meta.path),
        "size": meta.size,
        "mtime": meta.mtime,
        "robot": meta.robot,
        "kind": meta.kind,
        "playable": meta.playable,
        "note": meta.note,
        "companion": meta.companion,
        "meta": meta.summary(),
    }
