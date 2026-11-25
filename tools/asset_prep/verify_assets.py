#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""onnx-player 3D 模型资产校验脚本。

检查内容
--------
1. 用 xml.etree 解析 X2 / A3 的 URDF：
   - 所有 <mesh filename="..."> 必须是相对 URDF 文件的路径（不能是 package:// 或绝对路径），
     且目标文件真实存在（含大小写一致性检查，兼容大小写敏感的静态服务器）。
   - 打印根 link 与全部可动关节（revolute / continuous）的名称、顺序、类型。
2. X2 URDF 可动关节集合 vs X2 ONNX joint_names（29 个，Isaac 交错序）：
   - 报告 ONNX 中未被 URDF 覆盖的关节；并列出现代 URDF 多出的关节（head 等）。
3. A3 URDF 可动关节集合是否覆盖 A3 ONNX joint_names（29 个）。
4. 用 mujoco 加载 models/a3/a3_t2d5/mjcf/a3_t2d5.xml 与 models/x2/mjcf/x2_ultra.xml
   （附带加载 X2 scene.xml），打印 nq / nv / nu 与关节名。
5. 校验 models/robots.json 与 URDF 实际内容一致（rootLink / joints 顺序 / urdf、mjcf 路径存在）。

用法
----
    set PYTHONPATH=<repo>/onnx-player/.pylibs
    python verify_assets.py                 # 控制台输出 + 写 verify_report.txt
    python verify_assets.py --no-report     # 只打印
    python verify_assets.py --report out.txt

退出码：0 = 全部检查通过；1 = 存在失败项。
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

# 允许直接 `python verify_assets.py` 运行：把随项目附带的 .pylibs 加进 sys.path。
_PYLIBS = PROJECT_ROOT / ".pylibs"
if _PYLIBS.is_dir() and str(_PYLIBS) not in sys.path:
    sys.path.insert(0, str(_PYLIBS))

MOVABLE_TYPES = ("revolute", "continuous")

# X2 策略 ONNX 输出关节名（29 个，Isaac 交错序）。
X2_ONNX_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]

# A3 策略 ONNX 输出关节名（29 个）。
A3_ONNX_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# robots.json 里应有的机器人条目（与任务规格一致）。
EXPECTED_ROBOTS = {
    "x2_ultra": {
        "name": "灵犀 X2 Ultra",
        "urdf": "/models/x2/x2_ultra.urdf",
        "mjcf": "models/x2/mjcf/x2_ultra.xml",
        "rootLink": "base_link",
        "onnx_joint_names": X2_ONNX_JOINT_NAMES,
    },
    "a3_t2d5": {
        "name": "远征 A3 T2.5",
        "urdf": "/models/a3/a3_t2d5/urdf/model.urdf",
        "mjcf": "models/a3/a3_t2d5/mjcf/a3_t2d5.xml",
        "rootLink": "pelvis_link",
        "onnx_joint_names": A3_ONNX_JOINT_NAMES,
    },
}

# 需要 mujoco 加载的 MJCF（相对 models/ 目录）。scene.xml 为附加检查。
MUJCF_TARGETS = [
    ("A3 a3_t2d5", "a3/a3_t2d5/mjcf/a3_t2d5.xml", True),
    ("X2 x2_ultra", "x2/mjcf/x2_ultra.xml", True),
    ("X2 scene", "x2/mjcf/scene.xml", False),
]


class Reporter:
    """同时写 stdout / 报告文件的极简 reporter。"""

    def __init__(self, streams=()):
        self.streams = list(streams)
        self.failures = 0
        self.warnings = 0

    def line(self, text: str = "") -> None:
        for st in self.streams:
            try:
                st.write(text + "\n")
                st.flush()
            except Exception:  # noqa: BLE001 - 报告写失败不应掩盖校验结果
                pass

    def ok(self, text: str) -> None:
        self.line("  [PASS] " + text)

    def fail(self, text: str) -> None:
        self.failures += 1
        self.line("  [FAIL] " + text)

    def warn(self, text: str) -> None:
        self.warnings += 1
        self.line("  [WARN] " + text)

    def info(self, text: str) -> None:
        self.line("  [info] " + text)


# --------------------------------------------------------------------------- #
# URDF 解析
# --------------------------------------------------------------------------- #
class UrdfInfo:
    def __init__(self, path: Path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.robot_name = self.root.get("name") or ""
        self.links = [l.get("name") for l in self.root.findall("link") if l.get("name")]

        joints = []
        self.children = set()
        for j in self.root.findall("joint"):
            name = j.get("name")
            jtype = (j.get("type") or "").strip()
            joints.append((name, jtype))
            child = j.find("child")
            if child is not None and child.get("link"):
                self.children.add(child.get("link"))
        self.joints = joints
        self.root_links = [l for l in self.links if l not in self.children]

        # 所有 <mesh filename="..."> 引用（visual + collision），保留出现顺序。
        self.mesh_refs = []
        for mesh in self.root.iter("mesh"):
            fn = mesh.get("filename")
            if fn:
                self.mesh_refs.append(fn)

    @property
    def movable_joints(self):
        return [(n, t) for (n, t) in self.joints if t in MOVABLE_TYPES]


def check_urdf(rep: Reporter, label: str, urdf_path: Path) -> UrdfInfo | None:
    rep.line("")
    rep.line("=" * 78)
    rep.line(f"[URDF] {label}: {urdf_path}")
    rep.line("=" * 78)
    if not urdf_path.is_file():
        rep.fail(f"URDF 文件不存在: {urdf_path}")
        return None
    try:
        info = UrdfInfo(urdf_path)
    except ET.ParseError as exc:
        rep.fail(f"URDF XML 解析失败: {exc}")
        return None

    rep.ok(f"XML 解析成功（robot name = {info.robot_name!r}）")
    rep.info(f"links = {len(info.links)}，joints = {len(info.joints)}，"
             f"root link(s) = {info.root_links}")

    # 根 link 唯一性（urdf-loader 需要单根）。
    if len(info.root_links) == 1:
        rep.ok(f"URDF 根 link: {info.root_links[0]}")
    else:
        rep.fail(f"URDF 根 link 不唯一: {info.root_links}")
    if not info.movable_joints:
        rep.fail("URDF 中没有任何 revolute/continuous 关节")

    # mesh 引用路径检查（相对化 + 存在性 + 大小写一致性）。
    urdf_dir = urdf_path.parent
    uniq = sorted(set(info.mesh_refs))
    rep.info(f"<mesh> 引用共 {len(info.mesh_refs)} 处，去重后 {len(uniq)} 个文件")
    bad_style = 0
    missing = 0
    case_bad = 0
    for fn in uniq:
        norm = fn.replace("\\", "/")
        if norm.startswith("package://"):
            rep.fail(f"mesh 仍为 package:// 路径（浏览器不可解析）: {fn}")
            bad_style += 1
            continue
        if norm.startswith("/") or norm.startswith("file:") or \
                (len(norm) > 2 and norm[1] == ":"):
            rep.fail(f"mesh 为绝对路径/URL（浏览器不可解析）: {fn}")
            bad_style += 1
            continue
        target = (urdf_dir / norm).resolve()
        if not target.is_file():
            rep.fail(f"mesh 文件缺失: {fn} -> {target}")
            missing += 1
            continue
        # 大小写一致性（Windows 上 exists() 不区分大小写，浏览器静态服务器可能区分）。
        try:
            if target.name not in {p.name for p in target.parent.iterdir()}:
                rep.warn(f"文件名大小写与磁盘不一致: {fn}（磁盘名 {target.name}）")
                case_bad += 1
        except OSError as exc:
            rep.warn(f"无法列目录检查大小写: {target.parent} ({exc})")
    if not bad_style:
        rep.ok(f"所有 {len(uniq)} 个 mesh 均为相对路径（无 package:// / 绝对路径）")
    if not missing and uniq:
        rep.ok(f"所有 {len(uniq)} 个 mesh 文件均真实存在")
    if bad_style or missing:
        rep.fail(f"mesh 路径检查未通过（路径风格问题 {bad_style}，缺失 {missing}）")

    # 可动关节清单（URDF 顺序）。
    rep.line("")
    rep.line(f"  可动关节（{len(info.movable_joints)} 个，URDF 文件顺序）:")
    rep.line("   idx | type       | joint name")
    rep.line("   ----+------------+-----------------------------------")
    for idx, (name, jtype) in enumerate(info.movable_joints):
        rep.line(f"   {idx:>3} | {jtype:<10} | {name}")
    return info


# --------------------------------------------------------------------------- #
# ONNX 覆盖检查
# --------------------------------------------------------------------------- #
def check_onnx_coverage(rep: Reporter, label: str, info: UrdfInfo,
                        onnx_names) -> None:
    rep.line("")
    rep.line("-" * 78)
    rep.line(f"[ONNX] {label} URDF 可动关节 vs ONNX joint_names")
    rep.line("-" * 78)
    rep.info(f"ONNX joint 数 = {len(onnx_names)}，URDF 可动关节数 = {len(info.movable_joints)}")
    if len(set(onnx_names)) != len(onnx_names):
        rep.fail("ONNX joint_names 自身存在重复项")
    if len(onnx_names) != 29:
        rep.fail(f"ONNX joint 数应为 29，实际 {len(onnx_names)}")

    urdf_movable = [n for (n, _) in info.movable_joints]
    urdf_set = set(urdf_movable)
    onnx_set = set(onnx_names)

    missing = [n for n in onnx_names if n not in urdf_set]
    extra = [n for n in urdf_movable if n not in onnx_set]
    covered = [n for n in onnx_names if n in urdf_set]

    rep.info(f"交集合计: {len(covered)}/{len(onnx_names)}")
    rep.info(f"ONNX 顺序映射到 URDF（用于后端索引 -> 关节名）:")
    for idx, n in enumerate(onnx_names):
        rep.line(f"   onnx[{idx:>2}] {n:<32} -> URDF order index "
                 f"{urdf_movable.index(n) if n in urdf_set else '-'}")
    if missing:
        rep.fail(f"URDF 缺少 ONNX 关节 {len(missing)} 个: {missing}")
    else:
        rep.ok(f"URDF 覆盖全部 {len(onnx_names)} 个 ONNX 关节")
    if extra:
        rep.info(f"URDF 多出的可动关节（ONNX 不驱动，播放器可用默认/0 位姿）: {extra}")
    else:
        rep.info("URDF 没有多余的可动关节")


# --------------------------------------------------------------------------- #
# mujoco 加载检查
# --------------------------------------------------------------------------- #
def check_mujoco(rep: Reporter, label: str, xml_path: Path, required: bool) -> None:
    if not xml_path.is_file():
        msg = f"MJCF 文件不存在: {xml_path}"
        (rep.fail if required else rep.warn)(msg)
        return
    try:
        import mujoco  # noqa: PLC0415 - 延迟导入，便于没有 mujoco 时仍能跑 URDF 检查
    except ImportError as exc:
        (rep.fail if required else rep.warn)(
            f"无法 import mujoco（{exc}）；请设置 PYTHONPATH=<onnx-player>/.pylibs")
        return

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # noqa: BLE001 - mujoco 抛的异常类型不固定
        (rep.fail if required else rep.warn)(f"mujoco 加载失败: {xml_path} -> {exc}")
        return

    rep.ok(f"mujoco 加载成功: {xml_path}")
    rep.info(f"nq = {model.nq}  nv = {model.nv}  nu = {model.nu}  "
             f"nbody = {model.nbody}  njnt = {model.njnt}  nmesh = {model.nmesh}")
    joint_types = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    joint_names = []
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"<joint{j}>"
        joint_names.append(jname)
    rep.info(f"关节（{len(joint_names)} 个，MJCF 顺序）:")
    for j in range(model.njnt):
        jname = joint_names[j]
        jtype = joint_types.get(int(model.jnt_type[j]), str(int(model.jnt_type[j])))
        rep.line(f"   [{j:>2}] {jtype:<6} {jname}")
    act_names = []
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"<act{a}>"
        act_names.append(aname)
    if act_names:
        rep.info(f"执行器（{len(act_names)} 个）: {act_names}")
    else:
        rep.info("执行器: 无（nu = 0）")


# --------------------------------------------------------------------------- #
# robots.json 一致性检查
# --------------------------------------------------------------------------- #
def resolve_models_url(url_path: str) -> Path:
    """把 /models/xxx 形式的 URL 映射到 models/ 下的本地文件。"""
    prefix = "/models/"
    rel = url_path[len(prefix):] if url_path.startswith(prefix) else url_path
    return MODELS_DIR / rel


def check_robots_json(rep: Reporter, infos: dict) -> None:
    rep.line("")
    rep.line("=" * 78)
    rep.line(f"[robots.json] {MODELS_DIR / 'robots.json'}")
    rep.line("=" * 78)
    robots_json = MODELS_DIR / "robots.json"
    if not robots_json.is_file():
        rep.fail("robots.json 不存在")
        return
    try:
        data = json.loads(robots_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        rep.fail(f"robots.json 读取/解析失败: {exc}")
        return

    robots = data.get("robots")
    if not isinstance(robots, list):
        rep.fail("robots.json 顶层结构应为 {\"robots\": [...]}")
        return
    rep.ok(f"JSON 解析成功，robots 条目数 = {len(robots)}")

    seen = set()
    for robot in robots:
        rid = robot.get("id")
        if not rid:
            rep.fail(f"robots.json 条目缺少 id: {robot}")
            continue
        seen.add(rid)
        rep.line("")
        rep.info(f"robot id = {rid!r}")
        expected = EXPECTED_ROBOTS.get(rid)
        if expected is None:
            rep.fail(f"未知 robot id（期望 {sorted(EXPECTED_ROBOTS)}）: {rid}")
            continue
        for key in ("id", "name", "urdf", "rootLink", "joints", "upAxis", "mjcf"):
            if key not in robot:
                rep.fail(f"{rid}: 缺少字段 {key!r}")
        for key, want in (("name", expected["name"]), ("urdf", expected["urdf"]),
                          ("mjcf", expected["mjcf"]), ("upAxis", "z")):
            if robot.get(key) != want:
                rep.fail(f"{rid}.{key} = {robot.get(key)!r}，期望 {want!r}")

        # URDF 路径存在 + 内容一致。
        urdf_rel = robot.get("urdf", "")
        urdf_file = resolve_models_url(urdf_rel)
        if not urdf_file.is_file():
            rep.fail(f"{rid}: urdf 路径不存在 -> {urdf_file}")
        else:
            rep.ok(f"{rid}: urdf 存在 -> {urdf_file.relative_to(MODELS_DIR).as_posix()}")

        # mjcf 路径存在（robots.json 约定为相对 onnx-player 项目根的 "models/..." 路径）。
        mjcf_rel = robot.get("mjcf", "")
        mjcf_file = (PROJECT_ROOT / mjcf_rel).resolve()
        if not mjcf_file.is_file():
            rep.fail(f"{rid}: mjcf 路径不存在 -> {mjcf_file}")
        elif not mjcf_rel.startswith("models/"):
            rep.fail(f"{rid}.mjcf 应以 'models/' 开头（项目根相对路径），实际 {mjcf_rel!r}")
        else:
            rep.ok(f"{rid}: mjcf 存在 -> {mjcf_rel}")

        info = infos.get(rid)
        if info is None:
            rep.warn(f"{rid}: 没有对应 URDF 解析结果，跳过 rootLink/joints 内容比对")
            continue
        root_want = robot.get("rootLink")
        if info.root_links and root_want != info.root_links[0]:
            rep.fail(f"{rid}.rootLink = {root_want!r}，URDF 实际根 link = {info.root_links[0]!r}")
        else:
            rep.ok(f"{rid}.rootLink = {root_want!r}（与 URDF 一致）")

        joints = robot.get("joints") or []
        movable = [n for (n, _) in info.movable_joints]
        if joints == movable:
            rep.ok(f"{rid}.joints 与 URDF 可动关节顺序完全一致（{len(joints)} 个）")
        else:
            missing = [n for n in movable if n not in set(joints)]
            extra = [n for n in joints if n not in set(movable)]
            if missing or extra:
                rep.fail(f"{rid}.joints 与 URDF 集合不一致：缺少 {missing}，多出 {extra}")
            else:
                rep.fail(f"{rid}.joints 与 URDF 集合一致但顺序不同")
            rep.info(f"   robots.json joints = {joints}")
            rep.info(f"   URDF movable     = {movable}")

        onnx_names = expected["onnx_joint_names"]
        not_driven = [n for n in movable if n not in set(onnx_names)]
        if not_driven:
            rep.info(f"{rid}: 不被 ONNX 驱动的可动关节 = {not_driven}")

    missing_robots = sorted(set(EXPECTED_ROBOTS) - seen)
    if missing_robots:
        rep.fail(f"robots.json 缺少机器人条目: {missing_robots}")
    else:
        rep.ok("robots.json 覆盖全部期望机器人条目")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="onnx-player 模型资产校验")
    parser.add_argument("--report", default=str(SCRIPT_DIR / "verify_report.txt"),
                        help="报告输出文件（默认 tools/asset_prep/verify_report.txt）")
    parser.add_argument("--no-report", action="store_true", help="不写报告文件")
    args = parser.parse_args(argv)

    report_fp = None
    if not args.no_report:
        report_fp = open(args.report, "w", encoding="utf-8", newline="\n")
    rep = Reporter([sys.stdout] + ([report_fp] if report_fp else []))

    try:
        rep.line("=" * 78)
        rep.line("onnx-player 模型资产校验")
        rep.line("=" * 78)
        rep.line(f"project root : {PROJECT_ROOT}")
        rep.line(f"models dir   : {MODELS_DIR}")
        rep.line(f"python       : {sys.version.split()[0]}")

        infos = {}
        for rid, expected in EXPECTED_ROBOTS.items():
            urdf_file = resolve_models_url(expected["urdf"])
            info = check_urdf(rep, rid, urdf_file)
            if info is not None:
                infos[rid] = info
            check_onnx_coverage(rep, rid, info, expected["onnx_joint_names"]) \
                if info is not None else None

        rep.line("")
        rep.line("=" * 78)
        rep.line("[MuJoCo] MJCF 加载检查")
        rep.line("=" * 78)
        for label, rel, required in MUJCF_TARGETS:
            check_mujoco(rep, label, MODELS_DIR / rel, required)

        check_robots_json(rep, infos)
    finally:
        rep.line("")
        rep.line("=" * 78)
        rep.line("结果汇总")
        rep.line("=" * 78)
        rep.line(f"FAIL = {rep.failures}  WARN = {rep.warnings}  "
                 f"=> {'PASS' if rep.failures == 0 else 'FAILED'}")
        if report_fp:
            rep.line(f"报告已写入: {args.report}")
            report_fp.close()

    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
