/**
 * 灵犀 X2 动作跟踪策略的观测（151 维 29-DOF / 131 维 25-DOF），
 * 对齐官方 SDK extra/x2_rl_deploy 的 RlDefault 布局：
 *   command_dof_pos(n) + command_dof_vel(n)   ← 参考动作当前帧（TrackingCommand.command）
 *   projected_gravity(3)                      ← 根（pelvis）姿态
 *   base_ang_vel(3) * 0.25                    ← 根角速度（MuJoCo free joint qvel[3:6] 已是 body 系）
 *   dof_pos(n) = q - default_joint_pos（obs_scales.dof_pos = 1.0）
 *   dof_vel(n) * 0.05
 *   actions(n)
 * 观测整体 clip 到 ±100（部署端 obs_clip）。
 *
 * 说明：X2 实机部署用 torso IMU；MuJoCo 里 torso_link 与 IMU 安装系有偏差，用根（pelvis）
 * 姿态做重力/角速度时闭环明显更稳（poc_x2/sim_variants.py：root 不摔、torso 摔），
 * 这是与官方部署的已知近似。n 由 obs 维度反推：obs = 5n + 6，绑定默认 shape 由 Python 传入。
 */
import { ObservationBase } from 'mjswan/observation';

type Quat = [number, number, number, number]; // wxyz

function matFromQuat(q: Quat): number[] {
  const [w, x, y, z] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
  ];
}

function worldToBody(q: Quat, v: [number, number, number]): [number, number, number] {
  const m = matFromQuat(q);
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

export class LingxiX2Obs extends ObservationBase {
  private defaultPos: Float32Array | null = null;
  private obsSize = 151;
  private nj = 29;

  constructor(runner: any, config: any) {
    super(runner, config);
    const raw = Number(config?.shape ?? config?.obs_size ?? 151);
    const size = Number.isFinite(raw) && raw >= 11 && (raw - 6) % 5 === 0 ? raw : 151;
    this.obsSize = size;
    this.nj = (size - 6) / 5;
  }

  get size(): number {
    return this.obsSize;
  }

  private normQuat(raw: ArrayLike<number> | undefined | null): Quat {
    if (!raw || raw.length < 4) return [1, 0, 0, 0];
    const w = raw[0], x = raw[1], y = raw[2], z = raw[3];
    const n = Math.hypot(w, x, y, z) || 1;
    return [w / n, x / n, y / n, z / n];
  }

  compute(state: {
    jointPos: Float32Array;
    jointVel?: Float32Array;
    rootQuat?: Float32Array;
    rootAngVel?: Float32Array;
  }): Float32Array {
    const runner = this.runner as unknown as {
      getContext?: () => any;
      getLastActions?: () => Float32Array;
      getDefaultJointPos?: () => Float32Array;
    };
    const ctx = runner.getContext?.() ?? null;
    const commandManager = ctx?.commandManager ?? null;

    const nj = this.nj;
    let command = commandManager?.getCommand?.('motion') as Float32Array | undefined;
    if (!command || command.length !== 2 * nj) command = new Float32Array(2 * nj);

    const rootQuat = this.normQuat(state.rootQuat as ArrayLike<number> | undefined);
    const gravity = worldToBody(rootQuat, [0, 0, -1]);
    const baseAng = state.rootAngVel ?? new Float32Array(3);

    const jointPos = state.jointPos;
    const jointVel = state.jointVel ?? new Float32Array(jointPos.length);
    const defaultPos = this.defaultPos ?? (this.defaultPos = runner.getDefaultJointPos?.() ?? new Float32Array(jointPos.length));
    const lastAction = runner.getLastActions?.() ?? new Float32Array(jointPos.length);

    const out = new Float32Array(this.obsSize);
    let o = 0;
    out.set(command, o); o += 2 * nj;
    out[o++] = gravity[0]; out[o++] = gravity[1]; out[o++] = gravity[2];
    out[o++] = (baseAng[0] ?? 0) * 0.25;
    out[o++] = (baseAng[1] ?? 0) * 0.25;
    out[o++] = (baseAng[2] ?? 0) * 0.25;
    for (let i = 0; i < Math.min(nj, jointPos.length); i++) out[o + i] = jointPos[i] - (defaultPos[i] ?? 0);
    o += nj;
    for (let i = 0; i < Math.min(nj, jointVel.length); i++) out[o + i] = jointVel[i] * 0.05;
    o += nj;
    for (let i = 0; i < Math.min(nj, lastAction.length); i++) out[o + i] = lastAction[i] ?? 0;

    for (let i = 0; i < out.length; i++) out[i] = Math.max(-100, Math.min(100, out[i]));
    return out;
  }
}
