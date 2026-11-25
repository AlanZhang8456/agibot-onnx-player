#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 onnx-player/models/robots.json（后端 /api/robots 的数据源）。

从两份 URDF 实际解析 rootLink 与可动关节（revolute/continuous，URDF 文件顺序），
避免手写 JSON 与模型漂移。生成后请运行 verify_assets.py 复核。

用法：
    python build_robots_json.py            # 写 models/robots.json
    python build_robots_json.py --check    # 只打印不写
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # onnx-player/tools/asset_prep
PROJECT_ROOT = SCRIPT_DIR.parents[1]                  # onnx-player
MODELS_DIR = PROJECT_ROOT / "models"
MOVABLE_TYPES = ("revolute", "continuous")

ROBOT_SPECS = [
    {
        "id": "x2_ultra",
        "name": "灵犀 X2 Ultra",
        "urdf_rel": "x2/x2_ultra.urdf",              # 相对 models/
        "mjcf_rel": "x2/mjcf/x2_ultra.xml",          # 相对 models/
        "upAxis": "z",
    },
    {
        "id": "a3_t2d5",
        "name": "远征 A3 T2.5",
        "urdf_rel": "a3/a3_t2d5/urdf/model.urdf",
        "mjcf_rel": "a3/a3_t2d5/mjcf/a3_t2d5.xml",
        "upAxis": "z",
    },
]


def parse_urdf(path: Path):
    root = ET.parse(path).getroot()
    links = [l.get("name") for l in root.findall("link") if l.get("name")]
    children = set()
    movable = []
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            children.add(child.get("link"))
        jtype = (joint.get("type") or "").strip()
        if jtype in MOVABLE_TYPES and joint.get("name"):
            movable.append(joint.get("name"))
    root_links = [l for l in links if l not in children]
    return root_links, movable


def build() -> dict:
    robots = []
    for spec in ROBOT_SPECS:
        urdf_path = MODELS_DIR / spec["urdf_rel"]
        if not urdf_path.is_file():
            raise SystemExit(f"URDF 不存在: {urdf_path}")
        root_links, movable = parse_urdf(urdf_path)
        if len(root_links) != 1:
            raise SystemExit(f"{spec['id']}: URDF 根 link 不唯一: {root_links}")
        mjcf_path = MODELS_DIR / spec["mjcf_rel"]
        if not mjcf_path.is_file():
            raise SystemExit(f"MJCF 不存在: {mjcf_path}")
        robots.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "urdf": "/models/" + spec["urdf_rel"],
                "rootLink": root_links[0],
                "joints": movable,
                "upAxis": spec["upAxis"],
                "mjcf": "models/" + spec["mjcf_rel"],
            }
        )
    return {"robots": robots}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成 models/robots.json")
    parser.add_argument("--check", action="store_true", help="只打印，不写文件")
    args = parser.parse_args(argv)

    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    out = MODELS_DIR / "robots.json"
    if not args.check:
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"written: {out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
