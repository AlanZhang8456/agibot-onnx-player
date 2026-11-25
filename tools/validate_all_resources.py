# -*- coding: utf-8 -*-
"""批量校验资源库里所有可播放 ONNX 资源能否生成轨迹（开发自检用）。

用法（在 onnx-player 目录）：
    set PYTHONPATH=.pylibs
    python -X utf8 tools/validate_all_resources.py [--limit N] [--only x2|a3]
结果写到 tools/validate_report.txt。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 批量校验默认关闭 MuJoCo 仿真轨道（否则 134 个 A3 资源每个要跑 1~3s 仿真）
os.environ.setdefault("ONNX_PLAYER_NO_SIM", "1")

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from backend import library, motion_a3, motion_x2, onnx_meta as om  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    data = library.scan_library()
    entries = [e for e in data["entries"] if e["playable"]]
    if args.only:
        entries = [e for e in entries if e["robot"] == args.only]
    if args.limit:
        entries = entries[: args.limit]
    print(f"待校验可播放资源：{len(entries)}")

    lines = []
    ok = fail = 0
    t_all = time.time()
    counts = {}
    for i, e in enumerate(entries, 1):
        robot = e["robot"]
        counts[robot] = counts.get(robot, 0) + 1
        t0 = time.time()
        try:
            meta = om.read_meta(e["path"])
            if robot == "x2":
                res = motion_x2.build_x2_tracks(e["path"], meta)
            else:
                res = motion_a3.build_a3_tracks(e["path"], meta)
            dt = time.time() - t0
            msg = (f"[OK] {robot} {e['name']} | {res['frameCount']}帧 {res['duration']}s "
                   f"| {len(res['tracks'])}轨 | {dt:.2f}s | {e['relPath']}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            msg = f"[FAIL] {robot} {e['name']} | {exc} | {e['relPath']}"
            fail += 1
        print(msg, flush=True)
        lines.append(msg)

    lines.append("")
    lines.append(f"合计：{ok} 成功 / {fail} 失败 / {len(entries)} 个，耗时 {time.time()-t_all:.1f}s，"
                 f"分布 {counts}")
    print(lines[-1])
    with open(os.path.join(PROJECT, "tools", "validate_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
