# -*- coding: utf-8 -*-
"""把 onnx-player 的动作资源构建成 mjswan 静态站（替代旧自研前端/仿真栈）。

资源清单见 `resources.json`（相对 onnx-player 的资源路径或绝对路径）：
- kind=x2: 官方灵犀 X2 动作 ONNX（obs 151/131 + time_step 双输入），自动提取参考动作、
  生成循环包装 ONNX 与带 keyframe 的 MJCF；
- kind=a3: 远征 A3 动作 ONNX（obs 157/137 单输入）+ 配套 CSV 参考动作。

构建产物：`mjswan/site/`（index.html + manifest + 各 scene 资源），可直接用
`python -m http.server` 托管；`start.bat` 会检查 BUILD_INFO.json 后启动服务。

用法：
  python build_site.py                 # 增量构建 resources.json 全部资源
  python build_site.py --rebuild       # 强制重建资源缓存与站点
  python build_site.py --only x2       # 只构建某类资源（x2/a3）
  python build_site.py --limit 2       # 每种机器人最多 N 个 scene（快速预览）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # onnx-player/
SITE = HERE / "site"
CACHE = HERE / "cache"
RESOURCES_FILE = HERE / "resources.json"
BUILD_INFO = SITE / "BUILD_INFO.json"

sys.path.insert(0, str(HERE))
import mjswan_shim  # noqa: E402

import mjswan  # noqa: E402
from mjswan import register_observation  # noqa: E402
from mjswan.envs.mdp.actions import JointPositionActionCfg  # noqa: E402
from mjswan.envs.mdp.observations import ObservationBinding  # noqa: E402
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # noqa: E402


@dataclass
class MotionCommandCfg:
    """duck-typed mjlab MotionCommandCfg（mjswan 按类名绑定到原生 TrackingCommand）。"""

    motion_file: str
    anchor_body_name: str
    body_names: tuple
    entity_name: str = "robot"
    pose_range: dict = field(default_factory=dict)
    velocity_range: dict = field(default_factory=dict)
    joint_position_range: tuple = (0.0, 0.0)
    sampling_mode: str = "start"
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    debug_vis: bool = False

    class VizCfg:
        def __init__(self):
            self.mode = "ghost"
            self.ghost_color = (0.5, 0.7, 0.5, 0.5)

    viz: object = field(default_factory=VizCfg)


VIEWERS = {
    "a3": dict(lookat=(0.0, 0.0, 0.9), distance=4.6, elevation=-18.0, azimuth=140.0, body="pelvis_link"),
    "x2": dict(lookat=(0.0, 0.0, 0.8), distance=4.2, elevation=-18.0, azimuth=140.0, body="torso_link"),
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def load_resources(only: set[str] | None, limit: int | None) -> list[dict]:
    if not RESOURCES_FILE.exists():
        raise SystemExit(f"缺少资源清单：{RESOURCES_FILE}")
    items = json.loads(RESOURCES_FILE.read_text(encoding="utf-8"))
    items = [it for it in items if not only or it["kind"] in only]
    if limit:
        counts: dict[str, int] = {}
        picked = []
        for it in items:
            k = it["kind"]
            counts[k] = counts.get(k, 0) + 1
            if counts[k] <= limit:
                picked.append(it)
        items = picked
    return items


def sanitize(name: str) -> str:
    """只保留 ASCII 安全字符：MuJoCo 在 Windows 上打不开含中文的模型路径。"""
    keep = []
    for ch in name:
        if ch.isascii() and (ch.isalnum() or ch in "-_. "):
            keep.append("_" if ch == " " else ch)
    return "".join(keep)[:60].strip("_") or "resource"


def prepare_resource(item: dict, rebuild: bool) -> dict:
    """调用 prepare_x2/prepare_a3 生成该资源资产；带 mtime 缓存。"""
    kind = item["kind"]
    onnx_path = resolve(item["onnx"])
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 不存在：{onnx_path}")
    tag = f"{kind}_{sanitize(item.get('name') or onnx_path.parent.name)}_{sanitize(onnx_path.stem)[:20]}_{hashlib.md5(str(onnx_path.resolve()).encode('utf-8')).hexdigest()[:8]}"
    out_dir = CACHE / kind / tag
    meta_path = out_dir / "meta.json"
    sources = [onnx_path]
    if kind == "a3":
        csv_path = resolve(item["csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"A3 资源缺少配套 CSV：{csv_path}")
        sources.append(csv_path)
    fresh = (
        not rebuild
        and meta_path.exists()
        and meta_path.stat().st_mtime >= max(p.stat().st_mtime for p in sources)
    )
    if not fresh:
        if kind == "x2":
            cmd = [sys.executable, str(HERE / "prepare_x2.py"), str(onnx_path), "--out", str(out_dir)]
        else:
            cmd = [sys.executable, str(HERE / "prepare_a3.py"), str(onnx_path),
                   "--csv", str(resolve(item["csv"])), "--out", str(out_dir)]
        print(f"[prepare] {' '.join(cmd[-4:])}", flush=True)
        subprocess.check_call(cmd, cwd=str(HERE))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["_label"] = item.get("name") or meta.get("name") or onnx_path.parent.name
    meta["_kind"] = kind
    meta["_source"] = str(onnx_path)
    return meta


def observation_binding(kind: str, obs_dim: int) -> ObservationBinding:
    if kind == "a3":
        return ObservationBinding(ts_name="AgibotA3Obs", ts_src=str(HERE / "obs/AgibotA3Obs.ts"),
                                  defaults={"shape": int(obs_dim)})
    return ObservationBinding(ts_name="LingxiX2Obs", ts_src=str(HERE / "obs/LingxiX2Obs.ts"),
                              defaults={"shape": int(obs_dim)})


def add_scene(project, meta: dict, binding: ObservationBinding):
    kind = meta["_kind"]
    robot = "远征 A3 T2.5" if kind == "a3" else "灵犀 X2"
    scene_name = f"{robot} · {meta['_label']}"
    view = VIEWERS[kind]
    scene = project.add_scene(name=scene_name, spec=mujoco.MjSpec.from_file(meta["mjcf"]), control_dt=0.02)
    scene.set_viewer(mjswan.ViewerConfig(
        lookat=view["lookat"], distance=view["distance"], elevation=view["elevation"],
        azimuth=view["azimuth"],
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY, body_name=view["body"],
    ))
    scene.set_trace_env(mjswan_shim.make_trace_env_shim())

    common_actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(meta["joint_names"]),
            scale=meta["scale"],
            use_default_offset=True,
            stiffness=meta["kp"],
            damping=[v * 0.5 for v in meta["kd"]] if kind == "a3" else meta["kd"],
        ),
    }
    commands = {
        "motion": MotionCommandCfg(
            motion_file=meta["motion_npz"],
            anchor_body_name=meta["anchor"],
            body_names=tuple(meta["body_names"]),
            entity_name="robot",
        )
    }
    kwargs = dict(
        name=f"{robot} {meta['_label']}",
        policy_joint_names=meta["joint_names"],
        default_joint_pos=meta["default"],
        observations=ObservationGroupCfg(terms={"actor": ObservationTermCfg(func=binding)}),
        actions=common_actions,
        commands=commands,
    )
    if kind == "x2":
        kwargs["in_keys"] = ["actor", "time_step"]
        policy_onnx = meta["policy_loop_onnx"]
    else:
        policy_onnx = meta["_source"]
    policy = scene.add_policy(policy=onnx.load(policy_onnx), **kwargs)
    policy.add_motion(
        name=f"{meta['_label']} reference",
        source=meta["motion_npz"],
        fps=float(meta["fps"]),
        anchor_body_name=meta["anchor"],
        body_names=tuple(meta["body_names"]),
        dataset_joint_names=meta["joint_names"],
    )
    return {"name": scene_name, "kind": kind, "source": meta["_source"],
            "frames": meta.get("frames") or meta.get("total")}


def _display_name(path: Path) -> str:
    """从资源路径里取一个可读名字（跳过 unzip/0.0.1/dances/日期目录等）。"""
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
        # 去掉开头的长资源码（如 01KMM5DDN0W0GSV7Z476A5JRAZ-）
        if re.fullmatch(r"[0-9A-Z]{12,}", p):
            continue
        m = re.match(r"^[0-9A-Z]{12,}-(.+)$", p)
        if m:
            p = m.group(1)
        p = p.strip(" -_")
        if p:
            return p
    return path.stem


def _quick_meta(path: Path) -> dict | None:
    """只读 ONNX 头/元数据，判断是否是可进入静态站的资源。"""
    try:
        m = onnx.load(str(path), load_external_data=False)
    except Exception:  # noqa: BLE001
        return None
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    if not ins:
        return None
    md = {kv.key: kv.value for kv in m.metadata_props}
    obs_dim = ins[0][1][1] if len(ins[0][1]) > 1 else 0
    has_ts = any(name == "time_step" for name, _ in ins)
    if has_ts and obs_dim in (151, 131) and "joint_names" in md:
        n = int(md.get("num_q", "0") or 0) or len(md["joint_names"].split(","))
        if obs_dim != 5 * n + 6:
            return None
        return {"kind": "x2", "obs_dim": obs_dim, "joints": n, "name": _display_name(path),
                "source_mtime": path.stat().st_mtime}
    if not has_ts and len(ins) == 1 and obs_dim in (157, 137) and "joint_names" in md:
        n = len([x for x in md["joint_names"].split(",") if x.strip()])
        if obs_dim != 5 * n + 12:
            return None
        csv = path.with_suffix(".csv")
        if not csv.exists():
            siblings = sorted(path.parent.glob("*.csv"))
            if len(siblings) != 1:
                return None
            csv = siblings[0]
        return {"kind": "a3", "obs_dim": obs_dim, "joints": n, "name": _display_name(path),
                "csv": str(csv), "source_mtime": path.stat().st_mtime}
    return None


def _walk_onnx(root: Path, max_files: int = 3000, max_depth: int = 6):
    skip_dirs = {".git", "node_modules", ".cache", ".venv", "__pycache__", ".pytest_cache"}
    base_depth = len(root.parts)
    found = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in skip_dirs) if depth < max_depth else []
        for name in sorted(filenames):
            if name.lower().endswith(".onnx"):
                yield Path(dirpath) / name
                found += 1
                if found >= max_files:
                    return


def scan_items(only: set[str] | None, limit: int | None, force: bool) -> list[dict]:
    """扫描 config.json 里的资源根目录，产出可播放资源清单（结果缓存在 cache/scan_index.json）。"""
    cfg_path = ROOT / "config.json"
    roots = []
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        roots = [Path(r["path"]) for r in cfg.get("roots", []) if Path(r["path"]).is_dir()]
    if not roots:
        raise SystemExit("config.json 里没有可扫描的资源根目录")
    index_path = CACHE / "scan_index.json"
    index: dict = {}
    if index_path.exists() and not force:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            index = {}
    found: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        for path in _walk_onnx(root):
            key = str(path)
            st = path.stat()
            cached = index.get(key)
            if cached and cached.get("source_mtime") == st.st_mtime and cached.get("size") == st.st_size:
                meta = cached
            else:
                meta = _quick_meta(path)
                if meta is None:
                    index[key] = {"kind": None, "source_mtime": st.st_mtime, "size": st.st_size}
                    continue
                meta["size"] = st.st_size
                index[key] = meta
            if not meta or not meta.get("kind"):
                continue
            if key in seen:
                continue
            seen.add(key)
            found.append({"kind": meta["kind"], "name": meta.get("name") or _display_name(path),
                          "onnx": key, **({"csv": meta["csv"]} if meta.get("csv") else {})})
    CACHE.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    if only:
        found = [it for it in found if it["kind"] in only]
    # 同名去重，优先 A3 有 csv / X2 体积较小的资源
    found.sort(key=lambda it: (it["kind"], it["name"], it["onnx"]))
    picked: list[dict] = []
    seen_names: set[tuple] = set()
    for it in found:
        key = (it["kind"], it["name"])
        if key in seen_names:
            continue
        seen_names.add(key)
        picked.append(it)
    if limit:
        counts: dict[str, int] = {}
        limited = []
        for it in picked:
            k = it["kind"]
            counts[k] = counts.get(k, 0) + 1
            if counts[k] <= limit:
                limited.append(it)
        picked = limited
    print(f"[scan] roots={len(roots)} playable={len(found)} picked={len(picked)}", flush=True)
    for it in picked:
        print(f"  - {it['kind']} {it['name']} <- {it['onnx']}", flush=True)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="强制重建资源缓存与站点")
    ap.add_argument("--scan", action="store_true",
                    help="扫描 config.json 的资源根目录（可播放资源），而不是用 resources.json")
    ap.add_argument("--refresh-index", action="store_true", help="扫描时忽略 scan_index.json 缓存")
    ap.add_argument("--only", choices=["x2", "a3"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="每种机器人最多 N 个 scene（--scan 时默认 8）")
    ap.add_argument("--out", default=str(SITE))
    args = ap.parse_args()

    t0 = time.time()
    only = {args.only} if args.only else None
    if args.scan:
        items = scan_items(only, args.limit or 8, args.refresh_index)
    else:
        items = load_resources(only, args.limit)
    if not items:
        raise SystemExit("没有可构建的资源")

    mjswan_shim.patch_windows_esbuild()
    mjswan_shim.null_mjlab_contexts()
    mjswan_shim.register_tracking_motion_command()

    builder = mjswan.Builder()
    project = builder.add_project(name="AgiBot 动作播放器")
    built = []
    for item in items:
        meta = prepare_resource(item, args.rebuild)
        binding = observation_binding(meta["_kind"], int(meta["obs_dim"]))
        register_observation(binding.ts_name, binding)
        print(f"[scene] {meta['_kind']} {meta['_label']} <- {meta['_source']}", flush=True)
        built.append(add_scene(project, meta, binding))

    site = Path(args.out)
    builder.build(output_dir=str(site))
    info = {
        "builtAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "buildSeconds": round(time.time() - t0, 1),
        "scenes": built,
        "mode": "scan" if args.scan else "resources.json",
        "resourcesFile": None if args.scan else str(RESOURCES_FILE),
    }
    (site / "BUILD_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BUILD OK -> {site} ({len(built)} scenes, {info['buildSeconds']}s)", flush=True)


if __name__ == "__main__":
    main()
