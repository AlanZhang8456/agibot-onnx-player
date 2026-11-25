# -*- coding: utf-8 -*-
"""A3（远征 A3 T2.5）ONNX 动作跟踪策略的 MuJoCo 闭环回放探针。

目标
----
把 A3 动作资源（`*.onnx` 策略 + 同名参考动作 `*.csv`）放进 MuJoCo，用
obs[157] = [cmd_pos(29), cmd_vel(29), projected_gravity(3), base_ang_vel(3),
            joint_pos(29), joint_vel(29), last_action(29), motion_anchor_ori_b(6)]
每 0.02 s 推理一次，PD 力矩闭环驱动 MJCF 中的 29 个驱动关节，记录仿真轨迹，
并与 CSV 参考动作比较，输出消融指标。

对外接口（可直接被 onnx-player/backend 照搬）
--------------------------------------------
    from probe import simulate, DEFAULT_VARIANT
    result = simulate(onnx_path, csv_path, variant={...}, max_frames=1500)
    result["frames"]  -> {"rootPos": (N,3), "rootQuat": (N,4) wxyz, "joints": (N,29) onnx 顺序}
    result["metrics"] -> dict(...)

命令行
------
    python probe.py --onnx X.onnx --csv X.csv                     # 基线
    python probe.py --onnx X.onnx --csv X.csv --set imu_frame=torso,dof_vel_scale=0.05
    python probe.py --onnx X.onnx --csv X.csv --ablation --frames 800 --out abl.json
    python probe.py --help

依赖：numpy / onnx / onnxruntime / mujoco（见 scratch/sim_probe/REPORT.md）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
FPS = 50.0
POLICY_DT = 0.02
ONNX_OBS_DIM = 157

A3_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "samples", "a3")
DEFAULT_ONNX = os.path.join(A3_DIR_DEFAULT, "太极4-v03", "data.onnx")
DEFAULT_CSV = os.path.join(A3_DIR_DEFAULT, "太极4-v03", "data.csv")

# 播放器内置 A3 MJCF（来自 models/a3/a3_t2d5/mjcf，由官方 A3-A3U-robot-model 复制）
DEFAULT_XML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "a3", "a3_t2d5", "mjcf", "a3_t2d5.xml"))

# CSV 参考动作的关节列分组顺序（A3 default.yaml / 数据集约定）：
#   左腿6、右腿6、腰3(waist_yaw, waist_roll, waist_pitch)、左臂7、右臂7
_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw"]
CSV_JOINT_ORDER = (
    [f"left_{j}_joint" for j in _LEG]
    + [f"right_{j}_joint" for j in _LEG]
    + _WAIST
    + [f"left_{j}_joint" for j in _ARM]
    + [f"right_{j}_joint" for j in _ARM]
)
CSV_JOINT_ORDER_GROUPS = (
    [f"left_{j}_joint" for j in _LEG],
    [f"right_{j}_joint" for j in _LEG],
    list(_WAIST),
    [f"left_{j}_joint" for j in _ARM],
    [f"right_{j}_joint" for j in _ARM],
)

# 变体默认值 = 消融后的最佳配置（见 REPORT.md）：
#   anchor_ori 用 State_Mimic.cpp 取法 + anchor=pelvis；pelvis IMU；ang_vel*0.25；joint_vel*0.05；
#   joint_pos 相对；dt=0.001×20 子步；kd=metadata*0.5（MJCF 关节自带阻尼，需扣减）。
DEFAULT_VARIANT = dict(
    imu_frame="pelvis",        # pelvis | torso : projected_gravity / base_ang_vel 的 IMU 体轴系
    anchor_frame="pelvis",     # pelvis | torso : motion_anchor_ori_b 的 anchor 体
    anchor_mode="cpp",         # identity | rows | cpp | cpp_T : anchor 6D 取法（cpp=State_Mimic.cpp）
    ang_vel_scale=0.25,        # base_ang_vel 观测缩放（510c/default.yaml ang_vel_scale）
    dof_vel_scale=0.05,        # joint_vel 观测缩放（510c/default.yaml dof_vel_scale）
    joint_pos_mode="rel",      # rel | abs : joint_pos 观测是 q-default 还是 q
    cmd_lookahead=0,           # 0 | 1 : 参考动作是否前瞻一帧
    cmd_vel_mode="gradient",   # gradient | backward : cmd_vel 差分方式
    sim_dt=0.001,              # 物理子步长
    substeps=20,               # 每个策略步的物理子步数（sim_dt*substeps 应 = 0.02）
    init_mode="settle",        # exact | settle | fade : 初始对齐/过渡方式
    settle_s=0.2,              # settle 模式下先 PD 保持首帧的时长
    fade_hold_s=0.5,           # fade 模式下从默认站姿保持的时长
    fade_s=1.5,                # fade 模式余弦插值到首帧的时长
    action_clip=0.0,           # 0=不截断；否则 raw action 截断到 [-x, x]
    obs_clip=100.0,            # 0=不截断；否则 obs 截断到 [-x, x]（default.yaml obs_limit=100）
    action_source="onnx",      # onnx=策略闭环 | ref=直接用参考关节角做 PD（物理重定向回放基线）
    obs_source="sim",          # sim=仿真状态观测 | oracle=用参考状态造观测（诊断开关）
    debug_record=False,        # True 时在结果里附带每帧 raw action / PD target
    kp_scale=1.0,              # PD 刚度缩放（metadata joint_stiffness）
    kd_scale=0.5,              # PD 阻尼缩放（metadata joint_damping）；MJCF 已有关节阻尼，实测 0.5 最佳
    gains_source="metadata",   # metadata | deploy : PD 增益来源（deploy = 510c/default.yaml kp_limb/kd_limb）
    kd_mode="scale",           # scale | minus_joint_damping : kd=metadata*kd_scale 或减去 MJCF 关节阻尼
    ang_vel_sign=1,            # +1/-1 : base_ang_vel 符号（诊断用）
    cmd_order="policy",        # policy | csv : command 关节块顺序（诊断用）
    stop_on_fall=True,         # 摔倒后提前停止（指标按已跑帧计算）
    torque_clamp=True,         # 依赖 MuJoCo jnt_actfrcrange 自动限幅（仅记录，不额外处理）
)

# A3 官方部署配置 510c/default.yaml 中的 PD 增益（组序：左腿/右腿/腰/左臂/右臂）
_DEPLOY_KP_LIMB = (
    [80, 120, 80, 250, 80, 80], [80, 120, 80, 250, 80, 80], [85, 50, 50],
    [40, 40, 30, 30, 30, 20, 20], [40, 40, 30, 30, 30, 20, 20],
)
_DEPLOY_KD_LIMB = (
    [3, 4, 3, 8, 2, 2], [3, 4, 3, 8, 2, 2], [3, 2, 2],
    [5, 5, 4, 4, 4, 2, 2], [5, 5, 4, 4, 4, 2, 2],
)


def deploy_gains(joint_names):
    """按关节名返回 510c/default.yaml 的 kp/kd（组序=CSV 关节列顺序）。"""
    kp_map, kd_map = {}, {}
    for gi, group in enumerate(CSV_JOINT_ORDER_GROUPS):
        for j, name in enumerate(group):
            kp_map[name] = float(_DEPLOY_KP_LIMB[gi][j])
            kd_map[name] = float(_DEPLOY_KD_LIMB[gi][j])
    kp = np.array([kp_map[n] for n in joint_names])
    kd = np.array([kd_map[n] for n in joint_names])
    return kp, kd

# ---------------------------------------------------------------------------
# ONNX 元数据 / 会话
# ---------------------------------------------------------------------------
_SESSION_CACHE = {}


def _split_floats(text):
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


def _split_names(text):
    return [p.strip() for p in str(text).replace("\n", ",").split(",") if p.strip()]


def load_onnx_meta(onnx_path):
    """读取 ONNX metadata_props，返回 obs/action 契约所需字段。"""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    md = {kv.key: kv.value for kv in model.metadata_props}
    names = _split_names(md.get("joint_names", ""))
    if not names:
        raise ValueError(f"ONNX 缺少 joint_names metadata：{onnx_path}")
    default = np.asarray(_split_floats(md["default_joint_pos"]), dtype=np.float64)
    scale = np.asarray(_split_floats(md["action_scale"]), dtype=np.float64)
    stiffness = np.asarray(_split_floats(md.get("joint_stiffness", "")), dtype=np.float64)
    damping = np.asarray(_split_floats(md.get("joint_damping", "")), dtype=np.float64)
    for label, arr in (("default_joint_pos", default), ("action_scale", scale)):
        if arr.size != len(names):
            raise ValueError(f"ONNX metadata {label} 长度 {arr.size} != joint_names {len(names)}")
    if stiffness.size != len(names) or damping.size != len(names):
        raise ValueError("ONNX metadata 缺少 joint_stiffness / joint_damping，无法做 PD")
    obs_dim = 0
    for vi in model.graph.input:
        if vi.name == "obs":
            dims = [d.dim_value if d.HasField("dim_value") else -1
                    for d in vi.type.tensor_type.shape.dim]
            obs_dim = int(dims[-1]) if dims else 0
    return dict(
        joint_names=names,
        default_joint_pos=default,
        action_scale=scale,
        joint_stiffness=stiffness,
        joint_damping=damping,
        obs_dim=obs_dim,
        time_step_total=int(round(float(md.get("time_step_total", 0) or 0))),
        anchor_body_name=md.get("anchor_body_name", ""),
        task=md.get("task", ""),
        metadata=md,
    )


def get_session(onnx_path):
    import onnxruntime as ort

    key = os.path.abspath(onnx_path)
    sess = _SESSION_CACHE.get(key)
    if sess is not None:
        return sess
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(key, so, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[key] = sess
    return sess


# ---------------------------------------------------------------------------
# 参考动作 CSV
# ---------------------------------------------------------------------------
def load_reference(csv_path, joint_names):
    """读取参考动作 CSV，并把关节重排到 ONNX joint_names 顺序。

    CSV：root_xyz(3) + root_quat_xyzw(4) + len(CSV_JOINT_ORDER) 个关节角，50 Hz。
    """
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
    n = len(joint_names)
    if raw.shape[1] != 7 + len(CSV_JOINT_ORDER):
        # 兼容 25DOF 资源：CSV 为 7 + N，N 是 ONNX 关节数
        if raw.shape[1] == 7 + n:
            order = [x for x in CSV_JOINT_ORDER if x in set(joint_names)]
            if len(order) != n:
                raise ValueError(f"CSV 列数 {raw.shape[1]} 无法映射到 {n} 个 ONNX 关节")
        else:
            raise ValueError(
                f"CSV 列数 {raw.shape[1]} != 7 + 29（A3 参考动作格式）；"
                f"ONNX 关节数 {n}")
    else:
        order = list(CSV_JOINT_ORDER)
    idx = [order.index(name) for name in joint_names]
    pos = raw[:, 7:7 + len(order)][:, idx]
    quat_xyzw = raw[:, 3:7].copy()
    quat_xyzw /= np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    root_pos = raw[:, 0:3].copy()
    vel = np.gradient(pos, POLICY_DT, axis=0)
    return dict(root_pos=root_pos, root_quat_xyzw=quat_xyzw, joints=pos,
                vel=vel, frames=raw.shape[0], frames_raw=raw)


# ---------------------------------------------------------------------------
# 四元数 / 旋转工具（numpy float64）
# ---------------------------------------------------------------------------
def quat_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    out[..., 0] = q[..., 3]
    out[..., 1] = q[..., 0]
    out[..., 2] = q[..., 1]
    out[..., 3] = q[..., 2]
    return out


def quat_wxyz_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def mat_to_quat_wxyz(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return np.array([w / n, x / n, y / n, z / n])


def axis_rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def yaw_of_rotmat(R):
    """返回仅保留世界 Z 偏航的旋转矩阵。"""
    yaw = math.atan2(R[1, 0], R[0, 0])
    return axis_rot("z", yaw)


def projected_gravity(R):
    """R 为 body->world 旋转矩阵，返回机体系重力方向 R^T @ [0,0,-1]。"""
    return R.T @ np.array([0.0, 0.0, -1.0])


def rot_tilt(R):
    """机体 z 轴与世界 z 轴夹角（rad）。"""
    z_axis = R[:, 2]
    c = max(-1.0, min(1.0, float(z_axis[2])))
    return math.acos(c)


def _ang_vel_world(R0, R1, dt=POLICY_DT):
    """由两帧旋转矩阵差计算世界系角速度。"""
    dR = R1 @ R0.T
    theta = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0)))
    if theta > 1e-9:
        s = 2.0 * math.sin(theta)
        w = (theta / s) * np.array([dR[2, 1] - dR[1, 2],
                                    dR[0, 2] - dR[2, 0],
                                    dR[1, 0] - dR[0, 1]])
    else:
        w = np.zeros(3)
    return w / dt


# ---------------------------------------------------------------------------
# MuJoCo 模型缓存 + 关节/传感器映射
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


def get_model(xml_path):
    """加载（或复用）MJCF，并建立 ONNX 关节名 -> MuJoCo 索引映射。"""
    key = os.path.abspath(xml_path)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    import mujoco

    t0 = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(key)
    load_s = time.perf_counter() - t0

    def jid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise ValueError(f"MJCF 中找不到关节 {name}")
        return i

    def aid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"MJCF 中找不到执行器 {name}")
        return i

    def bid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"MJCF 中找不到 body {name}")
        return i

    def sid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)

    info = dict(
        model=m,
        xml_path=key,
        load_s=load_s,
        nq=m.nq, nv=m.nv, nu=m.nu,
        pelvis_bid=bid("pelvis_link"),
        torso_bid=bid("torso_Link"),
        pelvis_gyro_sid=sid("pelvis-angular-velocity"),
        torso_gyro_sid=sid("torso-angular-velocity"),
        pelvis_quat_sid=sid("pelvis-orientation"),
        torso_quat_sid=sid("torso-orientation"),
        free_jid=jid("free_joint"),
        head_jids=[jid("head_yaw_joint"), jid("head_pitch_joint")],
        jid=jid, aid=aid,
    )
    _MODEL_CACHE[key] = info
    return info


def build_joint_map(info, joint_names):
    """按 ONNX joint_names 顺序建立 MuJoCo qpos/dof/actuator 索引数组。"""
    import mujoco

    m = info["model"]
    n = len(joint_names)
    qpos_adr = np.zeros(n, dtype=np.int64)
    dof_adr = np.zeros(n, dtype=np.int64)
    act_ids = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(joint_names):
        j = info["jid"](name)
        qpos_adr[i] = m.jnt_qposadr[j]
        dof_adr[i] = m.jnt_dofadr[j]
        act_ids[i] = info["aid"](name)
    return dict(qpos_adr=qpos_adr, dof_adr=dof_adr, act_ids=act_ids)


def torso_mat_from_ref(ref, joint_names, k):
    """参考动作第 k 帧的 torso_Link 体轴系（pelvis 根四元数 + waist 关节级联）。"""
    R = quat_wxyz_to_mat(ref["root_quat_wxyz"][k])
    jp = ref["joints"][k]
    wi = {name: i for i, name in enumerate(joint_names)}
    if "waist_yaw_joint" in wi:
        R = R @ axis_rot("z", float(jp[wi["waist_yaw_joint"]]))
    if "waist_roll_joint" in wi:
        R = R @ axis_rot("x", float(jp[wi["waist_roll_joint"]]))
    if "waist_pitch_joint" in wi:
        R = R @ axis_rot("y", float(jp[wi["waist_pitch_joint"]]))
    return R


# ---------------------------------------------------------------------------
# 核心：simulate
# ---------------------------------------------------------------------------
def _merge_variant(variant):
    v = dict(DEFAULT_VARIANT)
    if variant:
        bad = [k for k in variant if k not in DEFAULT_VARIANT]
        if bad:
            raise KeyError(f"未知 variant 键：{bad}；可用：{list(DEFAULT_VARIANT)}")
        v.update(variant)
    if abs(v["sim_dt"] * v["substeps"] - POLICY_DT) > 1e-9:
        raise ValueError(
            f"sim_dt*substeps = {v['sim_dt'] * v['substeps']} != {POLICY_DT}（策略 50Hz）")
    return v


def simulate(onnx_path, csv_path, variant=None, max_frames=None,
             xml_path=DEFAULT_XML, verbose=False):
    """在 MuJoCo 中闭环回放 A3 ONNX 动作策略。

    Parameters
    ----------
    onnx_path : str  ONNX 策略（obs[1,157] -> actions[1,29]）
    csv_path  : str  同名参考动作 CSV（50Hz，root+29 关节角）
    variant   : dict 见 DEFAULT_VARIANT；None 表示用默认值
    max_frames: int|None 最多跑多少个策略步（默认按 CSV 全长 - lookahead）
    xml_path  : str  A3 MJCF 路径
    verbose   : bool 打印进度

    Returns
    -------
    dict:
        frames:   {"rootPos": (N,3) list, "rootQuat": (N,4) wxyz list, "joints": (N,29) list}
        metrics:  指标 dict（关节误差 / 根漂移 / 摔倒 / 耗时）
        variant / meta / ref_frames / n_frames
    """
    import mujoco

    t_start = time.perf_counter()
    v = _merge_variant(variant)
    meta = load_onnx_meta(onnx_path)
    joint_names = meta["joint_names"]
    n = len(joint_names)

    info = get_model(xml_path)
    m = info["model"]
    m.opt.timestep = float(v["sim_dt"])
    jmap = build_joint_map(info, joint_names)
    qpos_adr = jmap["qpos_adr"]
    dof_adr = jmap["dof_adr"]
    act_ids = jmap["act_ids"]

    # PD 增益 / action scale / default（全部在 ONNX 关节顺序）
    if v["gains_source"] == "deploy":
        kp, kd = deploy_gains(joint_names)
        kp = kp * float(v["kp_scale"])
        kd = kd * float(v["kd_scale"])
    else:
        kp = meta["joint_stiffness"].copy() * float(v["kp_scale"])
        kd = meta["joint_damping"].copy() * float(v["kd_scale"])
    if v["kd_mode"] == "minus_joint_damping" and v["gains_source"] == "metadata":
        # MJCF 关节自带 viscous damping，仿真里已生效；从 metadata kd 中扣除，避免重复阻尼
        joint_damp = m.dof_damping[jmap["dof_adr"]].copy()
        kd = np.maximum(kd - joint_damp, 0.1)
    act_scale = meta["action_scale"].copy()
    default_pos = meta["default_joint_pos"].copy()

    ref = load_reference(csv_path, joint_names)
    ref["root_quat_wxyz"] = quat_xyzw_to_wxyz(ref["root_quat_xyzw"])
    if v["cmd_vel_mode"] == "backward":
        vel = np.zeros_like(ref["joints"])
        vel[1:] = (ref["joints"][1:] - ref["joints"][:-1]) / POLICY_DT
        ref["vel"] = vel
    # command 关节块顺序：policy=ONNX joint_names 序；csv=CSV 组序（诊断用）
    csv_perm = [joint_names.index(nm) for nm in CSV_JOINT_ORDER
                if nm in joint_names]
    if v["cmd_order"] == "csv" and len(csv_perm) == n:
        ref["cmd_joints"] = ref["joints"][:, csv_perm]
        ref["cmd_vel"] = ref["vel"][:, csv_perm]
    else:
        ref["cmd_joints"] = ref["joints"]
        ref["cmd_vel"] = ref["vel"]
    total_frames = ref["frames"]
    lookahead = int(v["cmd_lookahead"])
    n_steps = total_frames - lookahead
    if max_frames is not None:
        n_steps = min(n_steps, int(max_frames))
    if n_steps < 1:
        raise ValueError("参考动作帧数不足")

    # ---------------- 初始化状态 ----------------
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    root0_pos = ref["root_pos"][0].copy()
    root0_quat = ref["root_quat_wxyz"][0].copy()
    j0 = ref["joints"][0].copy()

    def set_qpos(root_pos, root_quat_wxyz, joints):
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat_wxyz / np.linalg.norm(root_quat_wxyz)
        d.qpos[qpos_adr] = joints
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def apply_pd(target):
        """按每物理子步状态计算 PD 力矩，写入 ctrl。"""
        for _ in range(int(v["substeps"])):
            q = d.qpos[qpos_adr]
            qd = d.qvel[dof_adr]
            d.ctrl[act_ids] = kp * (target - q) - kd * qd
            mujoco.mj_step(m, d)

    def body_R(bid):
        return quat_wxyz_to_mat(d.xquat[bid])

    def gyro(sid):
        if sid < 0:
            return np.zeros(3)
        adr = m.sensor_adr[sid]
        return np.array(d.sensordata[adr:adr + 3], dtype=np.float64)

    def imu_R():
        return body_R(info["pelvis_bid"] if v["imu_frame"] == "pelvis"
                      else info["torso_bid"])

    # 初始姿态：默认 exact/settle 直接放到 CSV 首帧；fade 从默认站姿过渡
    if v["init_mode"] in ("exact", "settle"):
        set_qpos(root0_pos, root0_quat, j0)
    elif v["init_mode"] == "fade":
        # 默认站姿（ONNX default_joint_pos）+ 首帧根高度，先保持再余弦插值
        stand_root_pos = root0_pos.copy()
        set_qpos(stand_root_pos, root0_quat, default_pos)
        hold_steps = int(round(float(v["fade_hold_s"]) / (v["sim_dt"] * v["substeps"])))
        for _ in range(hold_steps):
            apply_pd(default_pos)
        prev_target = default_pos.copy()
        fade_steps = int(round(float(v["fade_s"]) / (v["sim_dt"] * v["substeps"])))
        for s in range(1, fade_steps + 1):
            p = s / fade_steps
            p = 0.5 * (1.0 - math.cos(math.pi * p))
            target = prev_target + (j0 - prev_target) * p
            apply_pd(target)
            prev_target = target
    else:
        raise ValueError(f"未知 init_mode: {v['init_mode']}")

    if v["init_mode"] == "settle" and v["settle_s"] > 0:
        settle_steps = int(round(float(v["settle_s"]) / (v["sim_dt"] * v["substeps"])))
        for _ in range(settle_steps):
            apply_pd(j0)

    # ---------------- init_quat（deploy 的初始 yaw 对齐） ----------------
    R_robot_anchor0 = body_R(info["pelvis_bid"] if v["anchor_frame"] == "pelvis"
                             else info["torso_bid"])
    if v["anchor_frame"] == "torso":
        R_ref_anchor0 = torso_mat_from_ref(ref, joint_names, 0)
    else:
        R_ref_anchor0 = quat_wxyz_to_mat(ref["root_quat_wxyz"][0])
    R_init = yaw_of_rotmat(R_robot_anchor0) @ yaw_of_rotmat(R_ref_anchor0).T

    # ---------------- 闭环主循环 ----------------
    sess = get_session(onnx_path)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    obs_dim = sess.get_inputs()[0].shape[-1]
    if obs_dim != ONNX_OBS_DIM:
        raise ValueError(f"期望 obs 维度 {ONNX_OBS_DIM}，实际 {obs_dim}")

    prev_action = np.zeros(n, dtype=np.float64)
    sim_joints = np.zeros((n_steps, n), dtype=np.float64)
    sim_root_pos = np.zeros((n_steps, 3), dtype=np.float64)
    sim_root_quat = np.zeros((n_steps, 4), dtype=np.float64)

    onnx_time = 0.0
    fall = False
    fall_frame = -1
    fallen_offset = 0
    n_used = n_steps
    dbg_actions = np.zeros((n_steps, n), dtype=np.float64) if v["debug_record"] else None
    dbg_targets = np.zeros((n_steps, n), dtype=np.float64) if v["debug_record"] else None
    loop_start = time.perf_counter()

    for k in range(n_steps):
        # --- obs 时刻状态（时间 k*0.02）---
        sim_joints[k] = d.qpos[qpos_adr]
        sim_root_pos[k] = d.xpos[info["pelvis_bid"]]
        sim_root_quat[k] = d.xquat[info["pelvis_bid"]]
        if v["action_source"] == "ref":
            # 物理重定向回放：直接 PD 跟踪参考关节角（不跑策略，用于评估 plant 保真度）
            target = ref["joints"][k + lookahead].copy()
            if fall:
                fallen_offset += 1
                if fallen_offset > 25:
                    n_used = k
                    break
            if dbg_targets is not None:
                dbg_targets[k] = target
            apply_pd(target)
            R_pelvis = quat_wxyz_to_mat(d.xquat[info["pelvis_bid"]])
            if (float(d.xpos[info["pelvis_bid"]][2]) < 0.4
                    or rot_tilt(R_pelvis) > 1.0) and not fall:
                fall = True
                fall_frame = k
            continue

        if fall:
            fallen_offset += 1
            if fallen_offset > 25:  # 摔倒后再记录 0.5s
                n_used = k
                break

        kk = k + lookahead
        cmd_pos = ref["cmd_joints"][kk]
        cmd_vel = ref["cmd_vel"][kk]

        if v["obs_source"] == "oracle":
            # 诊断：用“参考状态”造观测（等价于理想状态估计），隔离 obs 组装误差
            if v["imu_frame"] == "pelvis":
                R_imu = quat_wxyz_to_mat(ref["root_quat_wxyz"][kk])
                if kk + 1 < total_frames:
                    R_nxt = quat_wxyz_to_mat(ref["root_quat_wxyz"][kk + 1])
                    w_w = _ang_vel_world(R_imu, R_nxt)
                    w_b = R_imu.T @ w_w
                else:
                    w_b = np.zeros(3)
            else:
                R_imu = torso_mat_from_ref(ref, joint_names, kk)
                if kk + 1 < total_frames:
                    R_nxt = torso_mat_from_ref(ref, joint_names, kk + 1)
                    w_b = R_imu.T @ _ang_vel_world(R_imu, R_nxt)
                else:
                    w_b = np.zeros(3)
            jp_obs = ref["joints"][kk] - default_pos if v["joint_pos_mode"] == "rel" \
                else ref["joints"][kk].copy()
            jv_obs = ref["vel"][kk] * float(v["dof_vel_scale"])
        else:
            R_imu = imu_R()
            if v["imu_frame"] == "pelvis":
                w_b = gyro(info["pelvis_gyro_sid"])
            else:
                w_b = gyro(info["torso_gyro_sid"])
            q_now = d.qpos[qpos_adr]
            qd_now = d.qvel[dof_adr]
            if v["joint_pos_mode"] == "rel":
                jp_obs = q_now - default_pos
            else:
                jp_obs = q_now.copy()
            jv_obs = qd_now * float(v["dof_vel_scale"])
        g_b = projected_gravity(R_imu)
        base_ang_vel = w_b * (float(v["ang_vel_scale"]) * float(v["ang_vel_sign"]))

        # motion_anchor_ori_b
        amp = v["anchor_mode"]
        if amp == "identity":
            anchor = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        else:
            if v["obs_source"] == "oracle":
                R_real = (quat_wxyz_to_mat(ref["root_quat_wxyz"][kk])
                          if v["anchor_frame"] == "pelvis"
                          else torso_mat_from_ref(ref, joint_names, kk))
            else:
                R_real = body_R(info["pelvis_bid"] if v["anchor_frame"] == "pelvis"
                                else info["torso_bid"])
            if v["anchor_frame"] == "torso":
                R_ref = torso_mat_from_ref(ref, joint_names, kk)
            else:
                R_ref = quat_wxyz_to_mat(ref["root_quat_wxyz"][kk])
            R_rel = R_real.T @ (R_init @ R_ref)
            if amp == "rows":       # unitree_rl_lab observations.py: matrix_from_quat(...)[:, :2]
                anchor = np.array([R_rel[0, 0], R_rel[0, 1], R_rel[0, 2],
                                   R_rel[1, 0], R_rel[1, 1], R_rel[1, 2]])
            elif amp == "cpp":      # State_Mimic.cpp：转置后取 (0,0)(0,1)(1,0)(1,1)(2,0)(2,1)
                anchor = np.array([R_rel[0, 0], R_rel[0, 1], R_rel[1, 0],
                                   R_rel[1, 1], R_rel[2, 0], R_rel[2, 1]])
            elif amp == "cpp_T":    # 同上但按 “两个列向量” 顺序展平（诊断用）
                anchor = np.array([R_rel[0, 0], R_rel[1, 0], R_rel[0, 1],
                                   R_rel[1, 1], R_rel[0, 2], R_rel[1, 2]])
            else:
                raise ValueError(f"未知 anchor_mode: {amp}")

        obs = np.concatenate([cmd_pos, cmd_vel, g_b, base_ang_vel,
                              jp_obs, jv_obs, prev_action, anchor]).astype(np.float32)
        if v["obs_clip"] and float(v["obs_clip"]) > 0:
            obs = np.clip(obs, -float(v["obs_clip"]), float(v["obs_clip"]))
        if obs.shape[0] != ONNX_OBS_DIM:
            raise AssertionError(f"obs 组装维度 {obs.shape[0]} != {ONNX_OBS_DIM}")

        t1 = time.perf_counter()
        action = sess.run([out_name], {in_name: obs.reshape(1, -1)})[0][0].astype(np.float64)
        onnx_time += time.perf_counter() - t1

        if not np.all(np.isfinite(action)):
            action = np.zeros(n, dtype=np.float64)
        if v["action_clip"] and float(v["action_clip"]) > 0:
            action = np.clip(action, -float(v["action_clip"]), float(v["action_clip"]))
        prev_action = action.copy()

        target = action * act_scale + default_pos
        if dbg_actions is not None:
            dbg_actions[k] = action
            dbg_targets[k] = target
        apply_pd(target)

        # --- 摔倒检测（用步进后的状态）---
        R_pelvis = quat_wxyz_to_mat(d.xquat[info["pelvis_bid"]])
        z = float(d.xpos[info["pelvis_bid"]][2])
        tilt = rot_tilt(R_pelvis)
        if (z < 0.4 or tilt > 1.0) and not fall:
            fall = True
            fall_frame = k

        if verbose and (k + 1) % 250 == 0:
            err = np.abs(sim_joints[k] - ref["joints"][k]).mean()
            print(f"  [sim] frame {k + 1}/{n_steps} joint_err={err:.4f} "
                  f"z={z:.3f} tilt={tilt:.2f}")

    wall_s = time.perf_counter() - loop_start
    n_eval = n_used

    # ---------------- 指标 ----------------
    sim_joints = sim_joints[:n_eval]
    sim_root_pos = sim_root_pos[:n_eval]
    sim_root_quat = sim_root_quat[:n_eval]
    ref_joints = ref["joints"][:n_eval]
    ref_root_pos = ref["root_pos"][:n_eval]
    ref_root_quat = ref["root_quat_wxyz"][:n_eval]

    # 关节跟踪：k>=1（k=0 由初始化精确对齐，不计入）
    sl = slice(1, n_eval)
    abs_err = np.abs(sim_joints[sl] - ref_joints[sl])
    if abs_err.size == 0:
        abs_err = np.abs(sim_joints - ref_joints)
        sl = slice(0, n_eval)
    per_joint = abs_err.mean(axis=0)
    joint_err_view = np.abs(sim_joints - ref_joints)

    root_err_xyz = sim_root_pos - ref_root_pos
    root_xy_dist = np.linalg.norm(root_err_xyz[:, :2], axis=1)
    final_root_xy = float(root_xy_dist[-1]) if n_eval else float("nan")

    # 根姿态误差（旋转角）
    def quat_angle(q1, q2):
        dot = abs(float(np.dot(q1, q2)))
        dot = max(-1.0, min(1.0, dot))
        return 2.0 * math.acos(dot)

    root_ori_err = np.array([quat_angle(sim_root_quat[i], ref_root_quat[i])
                             for i in range(n_eval)])

    # 摔倒指标
    z_min = float(sim_root_pos[:, 2].min()) if n_eval else float("nan")
    R_last = quat_wxyz_to_mat(sim_root_quat[-1]) if n_eval else np.eye(3)
    tilt_max = float(max(rot_tilt(quat_wxyz_to_mat(sim_root_quat[i]))
                         for i in range(n_eval))) if n_eval else 0.0

    mean_err = float(abs_err.mean())
    p90 = float(np.percentile(abs_err, 90))
    p95 = float(np.percentile(abs_err, 95))
    max_err = float(abs_err.max())
    # 综合评分（越低越好）：关节误差 + 根漂移惩罚 + 摔倒惩罚
    score = mean_err + 0.5 * final_root_xy + (2.0 if fall else 0.0)

    metrics = dict(
        mean_abs_rad=mean_err,
        p90_abs_rad=p90,
        p95_abs_rad=p95,
        max_abs_rad=max_err,
        per_joint_mean_rad={name: float(per_joint[i]) for i, name in enumerate(joint_names)},
        joint_err_view_rad=float(joint_err_view.mean()),
        root_xy_drift_final_m=final_root_xy,
        root_xy_err_mean_m=float(root_xy_dist.mean()),
        root_pos_err_mean_m=float(np.linalg.norm(root_err_xyz, axis=1).mean()),
        root_z_min_m=z_min,
        root_z_mean_m=float(sim_root_pos[:, 2].mean()),
        root_z_final_m=float(sim_root_pos[-1, 2]) if n_eval else float("nan"),
        root_ori_err_mean_rad=float(root_ori_err.mean()),
        root_ori_err_final_rad=float(root_ori_err[-1]) if n_eval else float("nan"),
        tilt_max_rad=tilt_max,
        tilt_final_rad=float(rot_tilt(R_last)),
        fall=bool(fall),
        fall_frame=(int(fall_frame) if fall else -1),
        frames_run=int(n_eval),
        frames_total=int(n_steps),
        truncated=bool(n_eval < n_steps),
        total_s=float(time.perf_counter() - t_start),
        sim_s=float(wall_s),
        ms_per_frame=float(1000.0 * wall_s / max(n_eval, 1)),
        onnx_ms_per_frame=float(1000.0 * onnx_time / max(n_eval, 1)),
        model_load_s=float(info["load_s"]),
        score=float(score),
        action_mean_abs=float(np.mean(np.abs(prev_action))),
    )

    # ---------------- 输出帧（供播放器直接使用） ----------------
    frames = dict(
        rootPos=[[round(float(x), 6) for x in p] for p in sim_root_pos],
        rootQuat=[[round(float(x), 6) for x in q] for q in sim_root_quat],
        joints=[[round(float(x), 6) for x in j] for j in sim_joints],
    )

    result = dict(
        frames=frames,
        metrics=metrics,
        variant=v,
        meta=dict(joint_names=joint_names,
                  obs_dim=ONNX_OBS_DIM,
                  num_actions=n,
                  task=meta["task"],
                  anchor_body_name=meta["anchor_body_name"],
                  time_step_total=meta["time_step_total"],
                  csv_frames=int(ref["frames"])),
        n_frames=int(n_eval),
        ref_frames=int(ref["frames"]),
    )
    if dbg_actions is not None:
        result["debug"] = dict(
            actions=[[round(float(x), 5) for x in a] for a in dbg_actions[:n_eval]],
            targets=[[round(float(x), 5) for x in t] for t in dbg_targets[:n_eval]],
        )
    return result


# ---------------------------------------------------------------------------
# 内置消融实验
# ---------------------------------------------------------------------------
def _variant_tag(variant, base):
    diffs = [f"{k}={variant[k]}" for k in sorted(variant) if variant[k] != base[k]]
    return ",".join(diffs) if diffs else "baseline"


def ablation_specs():
    """返回 [(name, variant dict)]：覆盖任务要求的消融项。

    基线 = DEFAULT_VARIANT（= 消融后最佳配置）。每项只改一个维度。
    """
    base = dict(DEFAULT_VARIANT)
    specs = [("best", {})]
    # 1) IMU 体轴系 / anchor 体 / anchor 6D 公式
    specs += [
        ("imu_torso", dict(imu_frame="torso")),
        ("anchor_identity", dict(anchor_mode="identity")),
        ("anchor_rows", dict(anchor_mode="rows")),
        ("anchor_cpp_T", dict(anchor_mode="cpp_T")),
        ("anchor_frame_torso", dict(anchor_frame="torso")),
    ]
    # 2) obs 缩放 / joint_pos
    specs += [
        ("ang_vel_scale_1.0", dict(ang_vel_scale=1.0)),
        ("dof_vel_scale_1.0", dict(dof_vel_scale=1.0)),
        ("joint_pos_abs", dict(joint_pos_mode="abs")),
    ]
    # 3) lookahead / 差分 / 子步
    specs += [
        ("lookahead_1", dict(cmd_lookahead=1)),
        ("cmd_vel_backward", dict(cmd_vel_mode="backward")),
        ("substep_10_dt0.002", dict(sim_dt=0.002, substeps=10)),
    ]
    # 4) 初始对齐 / 过渡
    specs += [
        ("init_exact", dict(init_mode="exact")),
        ("init_settle_0.5", dict(settle_s=0.5)),
        ("init_fade", dict(init_mode="fade")),
    ]
    # 5) PD 增益来源 / kd 处理
    specs += [
        ("kd_scale_1.0", dict(kd_scale=1.0)),
        ("kd_minus_joint_damping", dict(kd_mode="minus_joint_damping")),
        ("gains_deploy", dict(gains_source="deploy")),
    ]
    # 6) 诊断开关：plant 保真度 / 理想状态观测（开环）
    specs += [
        ("diag_ref_pd", dict(action_source="ref")),
        ("diag_oracle_obs", dict(obs_source="oracle")),
    ]
    out = []
    for name, ov in specs:
        v = dict(base)
        v.update(ov)
        out.append((name, v))
    return out


def _print_metrics_row(name, r):
    mt = r["metrics"]
    print(f"{name:32s} err={mt['mean_abs_rad']:7.4f} p90={mt['p90_abs_rad']:7.4f} "
          f"drift={mt['root_xy_drift_final_m']:6.3f} zmin={mt['root_z_min_m']:6.3f} "
          f"fall={str(mt['fall']):5s} frames={mt['frames_run']:5d} "
          f"ms/f={mt['ms_per_frame']:6.1f} score={mt['score']:7.3f}")


def run_ablation(onnx_path, csv_path, max_frames, out_path=None, xml_path=DEFAULT_XML):
    rows = []
    print(f"\n=== 消融实验: {os.path.basename(onnx_path)}  frames<={max_frames} ===")
    print(f"{'variant':32s} {'joint err':>7s} {'p90':>7s} {'drift':>6s} "
          f"{'zmin':>6s} {'fall':>5s} {'frames':>6s} {'ms/f':>6s} {'score':>7s}")
    for name, variant in ablation_specs():
        r = simulate(onnx_path, csv_path, variant=variant, max_frames=max_frames,
                     xml_path=xml_path)
        _print_metrics_row(name, r)
        row = dict(name=name, variant=variant, metrics=r["metrics"],
                   n_frames=r["n_frames"])
        rows.append(row)
    rows.sort(key=lambda x: x["metrics"]["score"])
    print("\n-- 按 score 排序 --")
    for i, row in enumerate(rows):
        print(f"{i + 1:2d}. {row['name']:32s} score={row['metrics']['score']:.4f} "
              f"err={row['metrics']['mean_abs_rad']:.4f} fall={row['metrics']['fall']}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dict(onnx=onnx_path, csv=csv_path, max_frames=max_frames,
                           rows=rows), f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {out_path}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="A3 ONNX 策略 MuJoCo 闭环回放探针")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--csv", default=None, help="默认取同名 .csv")
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--set", default="", help="变体覆盖，如 imu_frame=torso,dof_vel_scale=0.05")
    ap.add_argument("--frames", type=int, default=None, help="最多策略步数")
    ap.add_argument("--out", default=None, help="单次结果 JSON 输出路径")
    ap.add_argument("--frames-out", default=None, help="轨迹帧 JSON 输出路径")
    ap.add_argument("--ablation", action="store_true", help="跑内置消融实验")
    ap.add_argument("--ablation-out", default=None, help="消融结果 JSON 输出路径")
    args = ap.parse_args(argv)

    onnx_path = os.path.abspath(args.onnx)
    csv_path = args.csv
    if not csv_path:
        csv_path = os.path.splitext(onnx_path)[0] + ".csv"
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(onnx_path):
        raise SystemExit(f"ONNX 不存在：{onnx_path}")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV 不存在：{csv_path}")

    if args.ablation:
        run_ablation(onnx_path, csv_path, args.frames, args.ablation_out, args.xml)
        return

    variant = {}
    for part in args.set.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, val = part.partition("=")
        k = k.strip()
        if k not in DEFAULT_VARIANT:
            raise SystemExit(f"未知 variant 键 {k}; 可用 {list(DEFAULT_VARIANT)}")
        cur = DEFAULT_VARIANT[k]
        if isinstance(cur, bool):
            variant[k] = val.strip().lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            variant[k] = int(val)
        elif isinstance(cur, float):
            variant[k] = float(val)
        else:
            variant[k] = val.strip()

    r = simulate(onnx_path, csv_path, variant=variant, max_frames=args.frames,
                 xml_path=args.xml, verbose=True)
    mt = r["metrics"]
    print(f"\n=== {os.path.basename(onnx_path)} | frames={r['n_frames']} ===")
    print(f"  mean|q_sim-q_ref| = {mt['mean_abs_rad']:.4f} rad   p90 = {mt['p90_abs_rad']:.4f}  "
          f"p95 = {mt['p95_abs_rad']:.4f}  max = {mt['max_abs_rad']:.4f}")
    print(f"  root: drift_final={mt['root_xy_drift_final_m']:.4f} m  "
          f"pos_err_mean={mt['root_pos_err_mean_m']:.4f} m  z_min={mt['root_z_min_m']:.4f} m")
    print(f"  fall={mt['fall']} fall_frame={mt['fall_frame']} tilt_max={mt['tilt_max_rad']:.3f} rad")
    print(f"  time: {mt['total_s']:.2f}s  ({mt['ms_per_frame']:.1f} ms/frame, "
          f"onnx {mt['onnx_ms_per_frame']:.1f} ms/frame)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False)
        print(f"[saved] {args.out}")
    if args.frames_out:
        with open(args.frames_out, "w", encoding="utf-8") as f:
            json.dump(dict(frames=r["frames"], jointNames=r["meta"]["joint_names"],
                           fps=FPS), f, ensure_ascii=False)
        print(f"[saved frames] {args.frames_out}")


if __name__ == "__main__":
    main()
