# -*- coding: utf-8 -*-
"""后端离线自测：不依赖浏览器，直接调用扫描/检查/轨迹生成。

运行：
    cd onnx-player
    set PYTHONPATH=.pylibs
    python -X utf8 -m backend.tests.test_backend
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT)

from backend import library, motion_a3, motion_x2, onnx_meta as om  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + str(detail)) if detail else ''}")
    if not cond:
        FAILS.append(name)


def find_sample(robot, need_policy_csv=False):
    data = library.scan_library()
    for e in data["entries"]:
        if e["robot"] != robot or not e["playable"]:
            continue
        if need_policy_csv and not e["companion"].get("csv"):
            continue
        return e
    return None


def main():
    t0 = time.time()
    # 1. 扫描
    data = library.scan_library()
    check("library 扫描返回 ok", data.get("ok") is True)
    print(f"      roots={[r['path'] for r in data['roots']]}")
    print(f"      entries={len(data['entries'])}, errors={len(data['errors'])}")
    for r in data.get("rootScans", []):
        print(f"      - {r['path']}: {r['count']} 个，{r['scan_ms']}ms")

    x2 = find_sample("x2")
    a3 = find_sample("a3")
    check("找到可播放 X2 资源", x2 is not None, x2 and x2["path"])
    check("找到可播放 A3 资源", a3 is not None, a3 and a3["path"])

    if x2:
        info = om.inspect(x2["path"])
        check("X2 inspect 输出 inputs", len(info["inputs"]) >= 2)
        t = time.time()
        res = motion_x2.build_x2_tracks(x2["path"], om.read_meta(x2["path"]))
        check("X2 轨迹生成 ok", res.get("frameCount", 0) > 0,
              f"{res['frameCount']} 帧 / {res['duration']}s / {time.time()-t:.2f}s")
        track = res["tracks"][0]
        n = len(res["jointNames"])
        check("X2 帧长度一致",
              len(track["rootPos"]) == res["frameCount"] == len(track["joints"]) == len(track["rootQuat"]))
        check("X2 关节维度一致", all(len(j) == n for j in track["joints"]))
        import math
        finite = all(math.isfinite(v) for j in track["joints"] for v in j)
        check("X2 数值有限", finite)
        zs = [p[2] for p in track["rootPos"]]
        check("X2 根高度合理 (0~1.3m)", 0.0 <= min(zs) and max(zs) < 1.3, f"{min(zs):.2f}~{max(zs):.2f}")
        print(f"      X2 关节角范围: {min(min(j) for j in track['joints']):.2f} ~ "
              f"{max(max(j) for j in track['joints']):.2f} rad")

    if a3:
        res = motion_a3.build_a3_tracks(a3["path"], om.read_meta(a3["path"]))
        check("A3 轨迹生成 ok", res.get("frameCount", 0) > 0, f"{res['frameCount']} 帧")
        ids = [t["id"] for t in res["tracks"]]
        check("A3 含 ref + 第二条轨迹(sim/policy)", ids[:2] in (["ref", "sim"], ["ref", "policy"]), ids)
        n = len(res["jointNames"])
        check("A3 关节数 29/25", n in (29, 25), n)
        for tr in res["tracks"]:
            check(f"A3 track {tr['id']} 长度一致",
                  len(tr["joints"]) == res["frameCount"] and all(len(j) == n for j in tr["joints"]))

    # 2. 错误处理
    try:
        om.read_meta(os.path.join(PROJECT, "no_such_file.onnx"))
        check("不存在文件应报错", False)
    except om.MetaError:
        check("不存在文件应报错", True)

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    if FAILS:
        print("失败项：", FAILS)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
