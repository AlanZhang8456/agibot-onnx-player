# -*- coding: utf-8 -*-
"""按需准备单个 ONNX 资源的 mjswan 运行时资产（不再重新构建整站）。

设计：
- 站点的前端/插件/机器人模型（scene.mjz）是固定的模板；
- 打开一个 ONNX 时，只准备这个资源自己的 policy.onnx / motion.npz / 场景目录，
  并把它追加到动态 manifest.json；
- 同一个文件第二次打开直接复用缓存（秒开）。

依赖：onnx、onnxruntime、numpy、mujoco（A3 的 FK 用；都在 onnx-player/.pylibs 里）。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent         # 代码/模板所在目录（PyInstaller 包里时可为只读）
MJS = HERE.parent                              # onnx-player/mjswan
# 可写的数据目录：打包成 exe 时由 launcher 指向 %LOCALAPPDATA%\AgiBotONNXPlayer
HOME = Path(os.environ.get("ONNX_PLAYER_HOME") or str(HERE))
ROOT = Path(os.environ.get("ONNX_PLAYER_ROOT") or str(MJS.parent))
APP = HOME / "app"
AGIBOT = APP / "agibot"
TEMPLATES = HOME / "templates"
RUNTIME = HOME / "runtime"
REGISTRY = RUNTIME / "registry.json"
MANIFEST = APP / "manifest.json"
LIB_INDEX = RUNTIME / "library_index.json"
A3_XML = ROOT / "models/a3/a3_t2d5/mjcf/a3_t2d5.xml"
ROBOT_NAMES = {"x2": "灵犀 X2", "a3": "远征 A3 T2.5"}
SHARED = AGIBOT / "_shared"


def ensure_shared_scenes() -> None:
    """机器人模型 scene.mjz 只存一份（X2 约 36MB），各资源通过相对路径共享。"""
    SHARED.mkdir(parents=True, exist_ok=True)
    for kind in ("a3", "x2"):
        dst = SHARED / f"{kind}_scene.mjz"
        src = TEMPLATES / f"{kind}_scene.mjz"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

sys.path.insert(0, str(MJS))
import prepare_a3  # noqa: E402
import prepare_x2  # noqa: E402


def floats(text: str) -> list[float]:
    return [float(x) for x in str(text).replace("\n", ",").split(",") if x.strip()]


def display_name(path: Path) -> str:
    skip = {"unzip", "dances", "resources", "resource", "onnx", "policy"}
    for part in reversed(path.parent.parts):
        p = part.strip()
        if not p or p in skip or p.isdigit() or p.startswith("未命名作品"):
            continue
        if p.startswith("0.0.") or p.startswith("$"):
            continue
        for prefix in ("linkcraft_resource_onnx_", "linkcraft_resource_", "X2-", "A3-"):
            if p.startswith(prefix):
                p = p[len(prefix):]
        if re.fullmatch(r"[0-9A-Z]{12,}", p):
            continue
        m = re.match(r"^[0-9A-Z]{12,}-(.+)$", p)
        if m:
            p = m.group(1)
        p = p.strip(" -_")
        if p:
            return p
    return path.stem


def ascii_slug(text: str, fallback: str = "resource") -> str:
    out = [c for c in text if c.isascii() and (c.isalnum() or c in "-_")]
    s = "".join(out).strip("-_")
    return s[:40] or fallback


def detect(onnx_path: Path) -> dict:
    """只读 ONNX 头/元数据，识别资源类型与策略契约。"""
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    md = {kv.key: kv.value for kv in m.metadata_props}
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    obs_dim = ins[0][1][1] if len(ins[0][1]) > 1 else 0
    has_ts = any(name == "time_step" for name, _ in ins)
    joint_names = [x.strip() for x in md.get("joint_names", "").split(",") if x.strip()]
    if not joint_names:
        raise ValueError("ONNX 缺少 joint_names metadata，无法识别为动作资源")
    n = len(joint_names)
    info = {
        "joint_names": joint_names,
        "body_names": [x.strip() for x in md.get("body_names", "").split(",") if x.strip()],
        "anchor": md.get("anchor_body_name") or ("torso_link" if has_ts else "pelvis_link"),
        "default": floats(md.get("default_joint_pos", "")),
        "scale": floats(md.get("action_scale", "")),
        "kp": floats(md.get("joint_stiffness", "")),
        "kd": floats(md.get("joint_damping", "")),
        "obs_dim": obs_dim,
        "time_step_total": int(float(md.get("time_step_total", "0") or 0)),
        "source": str(onnx_path),
    }
    if has_ts and obs_dim == 5 * n + 6:
        info["kind"] = "x2"
    elif not has_ts and len(ins) == 1 and obs_dim == 5 * n + 12:
        info["kind"] = "a3"
        csv = onnx_path.with_suffix(".csv")
        if not csv.exists():
            siblings = sorted(onnx_path.parent.glob("*.csv"))
            if len(siblings) != 1:
                raise ValueError("A3 资源需要同名配套 CSV（参考动作），当前目录找不到唯一 CSV")
            csv = siblings[0]
        info["csv"] = str(csv)
    else:
        raise ValueError(f"不是可播放的 X2/A3 动作资源（obs_dim={obs_dim}, joint_names={n}, time_step={has_ts}）")
    if len(info["default"]) != n or len(info["scale"]) != n:
        raise ValueError("ONNX metadata 的 default_joint_pos/action_scale 长度与 joint_names 不一致")
    return info


def _load_registry() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"entries": {}}


def _save_registry(reg: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REGISTRY)


def _scene_entry(kind: str, meta: dict, scene_id: str, label: str,
                 name: str | None = None) -> dict:
    template = json.loads((TEMPLATES / f"{kind}_scene.json").read_text(encoding="utf-8"))
    scene = copy.deepcopy(template)
    robot = ROBOT_NAMES[kind]
    scene["id"] = scene_id
    # mjswan 的 Scene 下拉框用 name 做唯一选项，重复会直接崩溃（黑屏），这里必须唯一
    scene["name"] = name or f"{robot} · {label}"
    scene["scene"] = f"../../templates/{kind}_scene.mjz"
    mdp = scene["mdps"][0]
    mdp["id"] = scene_id
    act = mdp["actions"]["joint_pos"]
    act["actuator_names"] = list(meta["joint_names"])
    act["scale"] = list(meta["scale"])
    act["stiffness"] = list(meta["kp"])
    kd = list(meta["kd"])
    act["damping"] = [v * 0.5 for v in kd] if kind == "a3" else kd
    cmd = mdp["commands"]["motion"]
    cmd["anchor_body_name"] = meta["anchor"]
    cmd["body_names"] = list(meta["body_names"])
    cmd["debug_vis"] = False
    mdp["observations"]["actor"][0]["shape"] = int(meta["obs_dim"])
    pol = scene["policies"][0]
    pol["id"] = scene_id
    pol["name"] = f"{robot} {label}"
    pol["mdp"] = scene_id
    pol["onnx"] = "policy/policy.onnx"
    pol["policy_joint_names"] = list(meta["joint_names"])
    pol["default_joint_pos"] = list(meta["default"])
    if kind == "x2":
        pol["in_keys"] = ["actor", "time_step"]
    else:
        pol.pop("in_keys", None)
    pol["motions"] = [{
        "name": f"{label} reference",
        "path": "assets/motion.npz",
        "fps": 50.0,
        "anchor_body_name": meta["anchor"],
        "body_names": list(meta["body_names"]),
        "dataset_joint_names": list(meta["joint_names"]),
        "default": True,
    }]
    for key in ("credit", "credits"):
        if key in scene and kind == "x2":
            scene.pop(key, None)
    return scene


def write_manifest() -> Path:
    reg = _load_registry()
    entries = sorted(reg["entries"].items(), key=lambda kv: kv[1].get("created", 0))
    # 去重显示名：旧版本可能留下同名 scene（用户从两个路径打开同一资源），
    # mjswan 的 Scene 下拉框遇到重复 label 会抛异常导致整页黑屏。
    used: set[str] = set()
    changed = False
    for src, entry in entries:
        scene = entry["scene"]
        name = scene.get("name") or entry.get("label") or src
        if name in used:
            suffix = entry["scene"]["id"][-6:]
            name = f"{name} [{suffix}]"
            while name in used:
                name = f"{name}*"
            scene["name"] = name
            changed = True
        used.add(name)
    if changed:
        _save_registry(reg)
    scenes = [entry["scene"] for _, entry in entries]
    manifest = {
        "format": 10,
        "version": "0.13.1",
        "uses_custom_js": True,
        "plugins": "assets/plugins.js",
        "projects": [{
            "id": "agibot",
            "name": "AgiBot 动作播放器",
            "scenes": scenes,
        }],
    }
    APP.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST)
    return MANIFEST


def prepared_list() -> list[dict]:
    reg = _load_registry()
    out = []
    for src, entry in reg["entries"].items():
        if not Path(src).exists():
            continue
        out.append({"source": src, "kind": entry["kind"], "label": entry.get("label", ""),
                    "scene": entry["scene"]["id"], "created": entry.get("created", 0),
                    "mtime": entry.get("mtime", 0)})
    out.sort(key=lambda e: e["created"], reverse=True)
    return out


def prepare(onnx_path: Path) -> dict:
    """准备（或复用缓存）一个资源的运行时资产，返回 registry entry。"""
    onnx_path = Path(onnx_path).expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"文件不存在：{onnx_path}")
    st = onnx_path.stat()
    reg = _load_registry()
    cached = reg["entries"].get(str(onnx_path))
    if cached:
        scene_id = cached["scene"]["id"]
        scene_dir = AGIBOT / scene_id
        shared = TEMPLATES / f"{cached['kind']}_scene.mjz"
        if (cached.get("mtime") == st.st_mtime and shared.exists()
                and (scene_dir / "policy" / "policy.onnx").exists()
                and (scene_dir / "assets" / "motion.npz").exists()):
            write_manifest()
            return cached

    meta = detect(onnx_path)
    kind = meta["kind"]
    label = display_name(onnx_path)
    token = hashlib.md5(f"{onnx_path}|{st.st_mtime}".encode("utf-8")).hexdigest()[:8]
    scene_id = f"{kind}_{ascii_slug(label)}_{token}"
    base_name = f"{ROBOT_NAMES[kind]} · {label}"
    used_names = {e["scene"].get("name") for e in reg["entries"].values()}
    display = base_name
    if display in used_names:
        display = f"{base_name} [{scene_id[-6:]}]"
    scene_dir = AGIBOT / scene_id
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    (scene_dir / "policy").mkdir(parents=True, exist_ok=True)
    (scene_dir / "assets").mkdir(parents=True, exist_ok=True)

    if kind == "a3":
        prepared = prepare_a3.prepare(onnx_path, Path(meta["csv"]), scene_dir / "assets")
        meta.update({k: prepared[k] for k in ("frames",)})
        shutil.copy2(onnx_path, scene_dir / "policy" / "policy.onnx")
        meta["body_names"] = prepared["body_names"]
    else:
        ref = prepare_x2.extract_reference(onnx_path, meta)
        mapped_bodies = ["pelvis" if b == "base_link" else b for b in meta["body_names"]]
        prepare_x2.build_loop_onnx(onnx_path, scene_dir / "policy" / "policy.onnx",
                                   ref["start"], ref["total"], int(meta["obs_dim"]))
        import numpy as np
        np.savez(
            scene_dir / "assets" / "motion.npz",
            joint_pos=np.ascontiguousarray(ref["joint_pos"]),
            joint_vel=np.ascontiguousarray(ref["joint_vel"]),
            body_pos_w=np.ascontiguousarray(ref["body_pos_w"]),
            body_quat_w=np.ascontiguousarray(ref["body_quat_w"]),
            body_lin_vel_w=np.ascontiguousarray(ref["body_lin_vel_w"]),
            body_ang_vel_w=np.ascontiguousarray(ref["body_ang_vel_w"]),
            body_names=np.array(mapped_bodies),
            fps=np.array([50.0], dtype=np.float32),
        )
        meta["body_names"] = mapped_bodies
        meta["frames"] = ref["total"]

    scene = _scene_entry(kind, meta, scene_id, label, display)
    entry = {
        "kind": kind,
        "label": label,
        "source": str(onnx_path),
        "mtime": st.st_mtime,
        "created": time.time(),
        "scene": scene,
    }
    reg["entries"][str(onnx_path)] = entry
    _save_registry(reg)
    write_manifest()
    return entry


def load_library() -> list[dict]:
    """可播放资源清单：优先用 build_site 的 scan_index，其次 player 自己的 library_index。"""
    candidates = [MJS / "cache" / "scan_index.json", LIB_INDEX]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        items = []
        for src, v in raw.items():
            if not v or not v.get("kind"):
                continue
            if not Path(src).exists():
                continue
            items.append({"source": src, "kind": v["kind"], "name": display_name(Path(src)),
                          "obs_dim": v.get("obs_dim"), "size": v.get("size", 0)})
        if items:
            items.sort(key=lambda e: (e["kind"], e["name"]))
            return items
    return []


def _walk_onnx(root: Path, max_files: int = 3000, max_depth: int = 6):
    skip_dirs = {".git", "node_modules", ".cache", ".venv", "__pycache__"}
    base = len(root.parts)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in skip_dirs) if depth < max_depth else []
        for name in sorted(filenames):
            if name.lower().endswith(".onnx"):
                yield Path(dirpath) / name
                count += 1
                if count >= max_files:
                    return


def rescan() -> int:
    """重新扫描 config.json 资源根目录，写 player/library_index.json；返回可播放数量。"""
    cfg_path = ROOT / "config.json"
    roots = []
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        roots = [Path(r["path"]) for r in cfg.get("roots", []) if Path(r["path"]).is_dir()]
    index: dict = {}
    played = 0
    for root in roots:
        for path in _walk_onnx(root):
            try:
                st = path.stat()
                meta = detect(path)
                index[str(path)] = {"kind": meta["kind"], "name": display_name(path),
                                    "obs_dim": meta["obs_dim"], "size": st.st_size,
                                    "source_mtime": st.st_mtime}
                played += 1
            except Exception:  # noqa: BLE001
                index[str(path)] = {"kind": None, "source_mtime": path.stat().st_mtime}
    index_path = RUNTIME / "library_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return played


def choose_file_dialog() -> str:
    """调用 Windows 原生文件选择框（服务器必须运行在交互桌面）。"""
    import subprocess
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.OpenFileDialog; "
        "$d.Title = '选择 ONNX 动作资源'; "
        "$d.Filter = 'ONNX 动作 (*.onnx)|*.onnx' | Out-Null; "
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($d.FileName) }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps],
                             capture_output=True, text=True, timeout=300)
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""
