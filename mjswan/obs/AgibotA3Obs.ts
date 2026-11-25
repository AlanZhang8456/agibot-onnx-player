/**
 * AgiBot A3 动作跟踪策略的观测（157 维 29-DOF / 137 维 25-DOF），
 * 严格对齐 A3 ONNX metadata 的 observation_names：
 *   command(2n) = ref joint_pos(n) + ref joint_vel(n)      ← mjswan TrackingCommand.command
 *   projected_gravity(3)                                   ← pelvis root quat
 *   base_ang_vel(3) * 0.25                                 ← pelvis root ang vel
 *                                                             (MuJoCo free joint qvel[3:6] 已是 body 系)
 *   joint_pos_rel(n) = q - default_joint_pos
 *   joint_vel(n) * 0.05
 *   last_action(n)
 *   motion_anchor_ori_b(6) = R(robot_anchor^-1 * ref_anchor) 的前两列展平
 *     [R00,R01,R10,R11,R20,R21]（与 mjlab matrix_from_quat(ori)[..., :2] 一致）
 *
 * obs = 5n + 12，n 由绑定默认 shape 反推（29 关节 → 157，25 关节 → 137）。
 */
import { ObservationBase } from 'mjswan/observation';

type Quat = [number, number, number, number]; // wxyz

function quatInv(q: Quat): Quat {
  return [q[0], -q[1], -q[2], -q[3]];
}

function quatMul(a: Quat, b: Quat): Quat {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

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

export class AgibotA3Obs extends ObservationBase {
  private cacheDefault: Float32Array | null = null;
  private obsSize = 157;
  private nj = 29;
  private commandBuf: Float32Array;

  constructor(runner: any, config: any) {
    super(runner, config);
    const raw = Number(config?.shape ?? config?.obs_size ?? 157);
    const size = Number.isFinite(raw) && raw >= 17 && (raw - 12) % 5 === 0 ? raw : 157;
    this.obsSize = size;
    this.nj = (size - 12) / 5;
    this.commandBuf = new Float32Array(2 * this.nj);
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
      getContext?: () => { commandManager?: any } | null;
      getLastActions?: () => Float32Array;
      getDefaultJointPos?: () => Float32Array;
    };
    const ctx = runner.getContext?.() ?? null;
    const commandManager = ctx?.commandManager ?? null;
    const motionTerm = commandManager?.getTerm?.('motion') ?? null;

    const nj = this.nj;
    let command = commandManager?.getCommand?.('motion') as Float32Array | undefined;
    if (!command || command.length !== 2 * nj) {
      this.commandBuf.fill(0);
      command = this.commandBuf;
    }

    const rootQuat = this.normQuat(state.rootQuat as ArrayLike<number> | undefined);
    const gravity = worldToBody(rootQuat, [0, 0, -1]);
    const bodyAng = state.rootAngVel ?? new Float32Array(3);

    const jointPos = state.jointPos;
    const jointVel = state.jointVel ?? new Float32Array(jointPos.length);
    const defaultPos = this.cacheDefault ?? (this.cacheDefault = runner.getDefaultJointPos?.() ?? new Float32Array(jointPos.length));
    const lastAction = runner.getLastActions?.() ?? new Float32Array(jointPos.length);

    const refQuatRaw = motionTerm?.getStateField?.('anchor_quat_w') ?? null;
    const refQuat = this.normQuat(refQuatRaw as ArrayLike<number> | undefined);
    const rel = quatMul(quatInv(rootQuat), refQuat);
    const relMat = matFromQuat(rel);
    const ori6 = [relMat[0], relMat[1], relMat[3], relMat[4], relMat[6], relMat[7]];

    const out = new Float32Array(this.obsSize);
    let o = 0;
    out.set(command, o); o += 2 * nj;
    out[o++] = gravity[0]; out[o++] = gravity[1]; out[o++] = gravity[2];
    out[o++] = (bodyAng[0] ?? 0) * 0.25;
    out[o++] = (bodyAng[1] ?? 0) * 0.25;
    out[o++] = (bodyAng[2] ?? 0) * 0.25;
    for (let i = 0; i < Math.min(nj, jointPos.length); i++) out[o + i] = jointPos[i] - (defaultPos[i] ?? 0);
    o += nj;
    for (let i = 0; i < Math.min(nj, jointVel.length); i++) out[o + i] = jointVel[i] * 0.05;
    o += nj;
    for (let i = 0; i < Math.min(nj, lastAction.length); i++) out[o + i] = lastAction[i] ?? 0;
    o += nj;
    for (let i = 0; i < 6; i++) out[o + i] = ori6[i] ?? 0;
    return out;
  }
}
