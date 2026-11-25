/**
 * ONNX 播放器前端入口
 * ------------------------------------------------------------------
 * 职责：
 *  1. 左侧资源库：GET /api/library（X2/A3 分组、搜索、点击播放）
 *  2. 中间 3D：urdf-loader 加载 /api/robots 的 URDF，按 /api/tracks 的逐帧
 *     rootPos(z-up)+rootQuat([w,x,y,z]) 与关节角回放；A3 双轨迹用 ghost 叠加
 *  3. 底部播放条：播放/暂停、时间轴 scrub、倍速、循环、重播、逐帧快捷键
 *  4. 右侧面板：元数据（/api/inspect 的 IO/metadata/warnings）+ 关节实时角度与曲线
 *  5. 健壮性：后端离线提示+重试、?selftest=1 内置 mock（不请求后端）、
 *     全局错误 toast、window.__APP_READY__
 *
 * 坐标约定：three.js 是 y-up，URDF/tracks 是 z-up。
 *   scene -> zup(rotation.x = -90°) -> mount(rootPos/rootQuat) -> URDF robot
 * 这样 rootPos/rootQuat 可直接按 z-up 世界系赋值，由 zup 统一转成 y-up。
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

// ===========================================================================
// 常量 / 工具
// ===========================================================================

const DEG = 180 / Math.PI;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ESC_MAP[ch]);

function fmtTime(sec) {
  return Number.isFinite(sec) && sec > 0 ? sec.toFixed(2) : '0.00';
}

function fmtSize(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(i === 0 || value >= 100 ? 0 : 1)} ${units[i]}`;
}

function basename(path) {
  const text = String(path || '');
  const idx = Math.max(text.lastIndexOf('\\'), text.lastIndexOf('/'));
  return idx >= 0 ? text.slice(idx + 1) : text;
}

/** 清理用户粘贴的路径：去首尾空白、去 Explorer「复制为路径」带的外层引号 */
function normalizeManualPath(raw) {
  let text = String(raw ?? '').trim();
  if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
    text = text.slice(1, -1).trim();
  }
  return text;
}

function robotKey(robot) {
  const value = String(robot || '').toLowerCase();
  if (value.includes('x2')) return 'x2';
  if (value.includes('a3')) return 'a3';
  return 'unknown';
}

function robotLabel(robot) {
  const key = robotKey(robot);
  return key === 'x2' ? '灵犀 X2' : key === 'a3' ? '远征 A3' : '未知机型';
}

function robotBadgeHtml(robot) {
  const key = robotKey(robot);
  const cls = key === 'x2' ? 'b-x2' : key === 'a3' ? 'b-a3' : 'b-unknown';
  const label = key === 'x2' ? 'X2' : key === 'a3' ? 'A3' : '?';
  return `<span class="badge ${cls}">${label}</span>`;
}

function kindLabel(kind) {
  if (kind === 'motion') return '动作';
  if (kind === 'locomotion') return '行走策略';
  if (kind === 'unknown') return '未知';
  return kind || '未知';
}

function kindBadgeHtml(kind) {
  const cls = kind === 'motion' ? 'b-motion' : kind === 'locomotion' ? 'b-locomotion' : 'b-unknown-kind';
  return `<span class="badge ${cls}">${escapeHtml(kindLabel(kind))}</span>`;
}

function playableHtml(playable, extra = '') {
  return playable
    ? `<span class="playable-tag">● 可播放</span>${extra}`
    : `<span class="playable-tag no">● 不可播放</span>${extra}`;
}

// ===========================================================================
// 运行模式：?selftest=1 时所有 /api 请求都走内置 mock，不碰后端
// ===========================================================================

const PARAMS = new URLSearchParams(window.location.search);
const SELFTEST = PARAMS.get('selftest') === '1' || PARAMS.get('selftest') === 'true';
// 联调入口：非 selftest 模式下 ?path=<URL 编码的绝对 ONNX 路径> 会在初始化后自动播放该资源
const AUTO_PATH = SELFTEST ? '' : (PARAMS.get('path') || '').trim();
const MOCK_URDF_URL = '/web/dev/fixtures/mock_robot.urdf';
const MOCK_JOINTS = [
  'torso_yaw_joint',
  'left_shoulder_pitch_joint',
  'left_elbow_joint',
  'right_shoulder_pitch_joint',
  'right_elbow_joint',
];

const SELFTEST_ROBOTS = [
  {
    id: 'x2_mock',
    name: '灵犀 X2（自检占位模型）',
    urdf: MOCK_URDF_URL,
    rootLink: 'base_link',
    joints: MOCK_JOINTS,
    upAxis: 'z',
  },
  {
    id: 'a3_mock',
    name: '远征 A3（自检占位模型）',
    urdf: MOCK_URDF_URL,
    rootLink: 'base_link',
    joints: MOCK_JOINTS,
    upAxis: 'z',
  },
];

const MOCK_ROOT = 'C:\\AgiBotResources';
const SELFTEST_ENTRIES = [
  {
    path: `${MOCK_ROOT}\\特技连踢__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip\\policy.onnx`,
    root: MOCK_ROOT,
    name: '特技连踢',
    dir: `${MOCK_ROOT}\\特技连踢__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip`,
    size: 1473145,
    mtime: 1769900000,
    robot: 'x2',
    kind: 'motion',
    playable: true,
    companion: { yaml: 'x2a.yaml', csv: null, media: [] },
    meta: { obs_dim: 151, num_actions: 29, time_step_total: 337, duration: 6.74 },
    note: '',
  },
  {
    path: `${MOCK_ROOT}\\A3_行走展示__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip\\policy.onnx`,
    root: MOCK_ROOT,
    name: 'A3 行走展示',
    dir: `${MOCK_ROOT}\\A3_行走展示__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip`,
    size: 2214500,
    mtime: 1769900000,
    robot: 'a3',
    kind: 'motion',
    playable: true,
    companion: { yaml: 'a3_motion.yaml', csv: 'reference.csv', media: ['expression.mp4'] },
    meta: { obs_dim: 176, num_actions: 31, duration: 6.74 },
    note: '含 ref + policy 两条轨迹',
  },
  {
    path: `${MOCK_ROOT}\\A3_速度策略__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip\\policy.onnx`,
    root: MOCK_ROOT,
    name: 'A3 速度策略（行走）',
    dir: `${MOCK_ROOT}\\A3_速度策略__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip`,
    size: 980000,
    mtime: 1769900000,
    robot: 'a3',
    kind: 'locomotion',
    playable: false,
    companion: { yaml: null, csv: null, media: [] },
    meta: { obs_dim: 176, num_actions: 31 },
    note: '行走/速度策略，播放器暂不支持',
  },
  {
    path: `${MOCK_ROOT}\\未知资源__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip\\policy.onnx`,
    root: MOCK_ROOT,
    name: '未知资源',
    dir: `${MOCK_ROOT}\\未知资源__linkcraft_resource_onnx_01KMMOCK\\0.0.1\\unzip`,
    size: 512000,
    mtime: 1769900000,
    robot: 'unknown',
    kind: 'unknown',
    playable: false,
    companion: { yaml: null, csv: null, media: [] },
    meta: {},
    note: '无法识别的 ONNX',
  },
];

const MOCK_FPS = 50;
const MOCK_DURATION = 6.74;
const MOCK_FRAMES = Math.round(MOCK_DURATION * MOCK_FPS); // 337

/** 自测用程序化轨迹（z-up：rootPos=[x,y,z]，rootQuat=[w,x,y,z]，关节顺序=MOCK_JOINTS） */
function buildMockTrack(def) {
  const rootPos = [];
  const rootQuat = [];
  const joints = [];
  const euler = new THREE.Euler();
  const quat = new THREE.Quaternion();
  const wArm = Math.PI * 2 * 0.5; // 2s 摆臂周期
  const wSlow = Math.PI * 2 * 0.25; // 4s 慢摆周期
  for (let i = 0; i < MOCK_FRAMES; i += 1) {
    const t = i / MOCK_FPS;
    const swing = Math.sin(wArm * t + def.phase);
    const slow = Math.sin(wSlow * t + def.phase * 0.5);
    const cosSwing = Math.cos(wArm * t + def.phase);
    joints.push([
      0.34 * def.amp * slow,
      0.62 * def.amp * swing,
      0.30 + 0.24 * def.amp * cosSwing,
      -0.62 * def.amp * swing,
      0.30 + 0.24 * def.amp * Math.cos(wArm * t + def.phase + Math.PI * 0.55),
    ]);
    rootPos.push([
      0.12 * Math.sin(wSlow * t + def.rootBias),
      0.03 * Math.cos(wSlow * t + def.phase),
      0.80 + 0.010 * Math.sin(wArm * t + def.phase),
    ]);
    const yaw = def.rootYawAmp * Math.sin(wSlow * t + def.phase) + def.rootBias;
    const roll = 0.025 * Math.sin(wArm * t + def.phase);
    const pitch = 0.018 * Math.cos(wSlow * t);
    euler.set(roll, pitch, yaw, 'XYZ');
    quat.setFromEuler(euler);
    rootQuat.push([quat.w, quat.x, quat.y, quat.z]);
  }
  return { id: def.id, label: def.label, defaultVisible: def.defaultVisible, rootPos, rootQuat, joints };
}

function buildMockTracks(kind) {
  const defs =
    kind === 'a3'
      ? [
          { id: 'ref', label: 'ONNX 参考动作（CSV ref）', defaultVisible: true, phase: 0, amp: 1, rootBias: 0, rootYawAmp: 0.32 },
          { id: 'policy', label: 'ONNX 策略输出（policy）', defaultVisible: true, phase: 1.3, amp: 0.75, rootBias: 0.22, rootYawAmp: 0.22 },
        ]
      : [{ id: 'ref', label: 'ONNX 参考动作', defaultVisible: true, phase: 0, amp: 1, rootBias: 0, rootYawAmp: 0.2 }];
  return {
    ok: true,
    path: '',
    robot: kind,
    modelHint: kind === 'a3' ? 'a3_mock' : 'x2_mock',
    fps: MOCK_FPS,
    duration: MOCK_DURATION,
    frameCount: MOCK_FRAMES,
    jointNames: MOCK_JOINTS.slice(),
    tracks: defs.map(buildMockTrack),
    meta: { obs_dim: kind === 'a3' ? 176 : 151, num_actions: kind === 'a3' ? 31 : 29, time_step_total: MOCK_FRAMES },
    warnings: kind === 'a3' ? ['自检数据：policy 轨迹为 mock 正弦叠加，仅用于 UI 验证。'] : [],
  };
}

function buildMockInspect(entry) {
  const isA3 = robotKey(entry.robot) === 'a3';
  return {
    ok: true,
    path: entry.path,
    size: entry.size,
    robot: entry.robot,
    kind: entry.kind,
    playable: entry.playable,
    inputs: [
      { name: 'obs', shape: [1, isA3 ? 176 : 151], dtype: 'float32' },
      { name: 'time_step', shape: [1, 1], dtype: 'float32' },
    ],
    outputs: [{ name: 'actions', shape: [1, isA3 ? 31 : 29], dtype: 'float32' }],
    metadata: {
      joint_names: MOCK_JOINTS.join(','),
      obs_dim: String(isA3 ? 176 : 151),
      num_actions: String(isA3 ? 31 : 29),
      time_step_total: String(MOCK_FRAMES),
      mock: 'selftest=1',
      note: '此资源数据由前端内置 mock 生成，未请求后端。',
    },
    companion: entry.companion,
    warnings: ['自检模式：使用内置 mock 资源，未请求本地后端。'],
  };
}

/** 统一的数据入口：selftest 分支完全不出网，其余走 /api */
const selftestApi = {
  async health() {
    await sleep(40);
    return { ok: true, version: '0.1.0-selftest', roots: [{ path: MOCK_ROOT, label: `${MOCK_ROOT}（自检）` }] };
  },
  async robots() {
    await sleep(40);
    return { ok: true, robots: SELFTEST_ROBOTS };
  },
  async library() {
    await sleep(120);
    return { ok: true, entries: SELFTEST_ENTRIES };
  },
  async inspect(path) {
    await sleep(80);
    const entry = SELFTEST_ENTRIES.find((item) => item.path === path) || SELFTEST_ENTRIES[0];
    return buildMockInspect(entry);
  },
  async tracks(path) {
    await sleep(240);
    const entry = SELFTEST_ENTRIES.find((item) => item.path === path) || SELFTEST_ENTRIES[0];
    const data = buildMockTracks(robotKey(entry.robot) === 'a3' ? 'a3' : 'x2');
    data.path = entry.path;
    return data;
  },
};

// ===========================================================================
// DOM 引用
// ===========================================================================

const dom = {
  serverPill: $('server-pill'),
  serverDot: $('server-dot'),
  serverText: $('server-text'),
  modelSelect: $('model-select'),
  btnRetry: $('btn-retry'),

  rootSelect: $('root-select'),
  btnRefresh: $('btn-refresh'),
  searchInput: $('search-input'),
  manualPathInput: $('manual-path-input'),
  btnLoadPath: $('btn-load-path'),
  libHint: $('lib-hint'),
  entryList: $('entry-list'),
  libCount: $('lib-count'),

  viewport: $('viewport'),
  canvas: $('viewer-canvas'),
  nowPlaying: $('now-playing'),
  ghostToggleWrap: $('ghost-toggle-wrap'),
  ghostToggle: $('ghost-toggle'),
  btnResetView: $('btn-reset-view'),
  viewOverlay: $('view-overlay'),
  viewOverlayTitle: $('view-overlay-title'),
  viewOverlayText: $('view-overlay-text'),
  loadProgress: $('load-progress'),
  loadProgressFill: $('load-progress-fill'),
  loadElapsed: $('load-elapsed'),
  offlineOverlay: $('offline-overlay'),
  offlineDetail: $('offline-detail'),
  btnOfflineRetry: $('btn-offline-retry'),
  viewError: $('view-error'),
  viewErrorText: $('view-error-text'),
  btnDismissError: $('btn-dismiss-error'),

  btnPlay: $('btn-play'),
  btnReplay: $('btn-replay'),
  timeCurrent: $('time-current'),
  timeTotal: $('time-total'),
  timeline: $('timeline'),
  speedSelect: $('speed-select'),
  loopToggle: $('loop-toggle'),
  fpsInfo: $('fps-info'),

  tabs: document.querySelector('.tabs'),
  metaBody: $('meta-body'),
  jointCurve: $('joint-curve'),
  curveCaption: $('curve-caption'),
  jointList: $('joint-list'),
  jointsCount: $('joints-count'),
  jointsFrame: $('joints-frame'),
  toastStack: $('toast-stack'),
};

// ===========================================================================
// 全局状态
// ===========================================================================

const state = {
  selftest: SELFTEST,
  mode: SELFTEST ? 'selftest' : 'backend',
  online: false,
  health: null,
  roots: [],
  root: '',

  robots: [],
  modelId: null,

  entries: [],
  search: '',
  libraryLoading: false,
  libraryLoaded: false,
  libraryError: '',
  loadingEntryPath: '',
  resourceToken: 0,
  // 手动路径加载失败原因（显示在右侧 warnings，方便不在根目录的资源排查）
  manualError: '',
  lastResourceError: '',

  entry: null,
  inspect: null,
  trackData: null,
  primaryIndex: 0,
  ghostIndex: -1,
  ghostVisible: false,

  activeTab: 'meta',
  selectedJoint: null,
  jointRows: new Map(),
  // ?path= 自动加载完成前不设置 __APP_READY__，方便 headless 集成截图等“已开始播放”再截
  deferReady: Boolean(AUTO_PATH),
};

const play = {
  playing: false,
  time: 0,
  speed: 1,
  loop: true,
  frameCount: 0,
  fps: 50,
  duration: 0,
  scrubbing: false,
  lastFrame: -1,
};

/** three.js 场景与模型相关状态 */
const view = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  zup: null,
  mount: null,
  ghostMount: null,
  robot: null,
  ghost: null,
  ghostMaterial: null,
  robotMeta: null,
  placeholder: null,
  modelToken: 0,
  modelLoading: false,
  modelReady: false,
  modelName: '',
  jointMap: [],
  jointMissing: [],
  ghostJointMap: [],
  meshErrors: [],
  meshPending: 0,
  lastWidth: 0,
  lastHeight: 0,
  wheelInited: false,
};

// ===========================================================================
// Toast / 覆盖层 / 服务状态
// ===========================================================================

function toast(message, kind = 'info', ms = 2800) {
  if (!message) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = String(message);
  dom.toastStack.append(el);
  while (dom.toastStack.childElementCount > 4) dom.toastStack.firstElementChild.remove();
  setTimeout(() => {
    el.classList.add('leaving');
    setTimeout(() => el.remove(), 220);
  }, ms);
}

const reportedErrors = new Map();
function reportGlobalError(message) {
  const now = Date.now();
  const last = reportedErrors.get(message) || 0;
  if (now - last < 4000) return;
  reportedErrors.set(message, now);
  toast(message, 'error', 4200);
}

function setServerStatus(kind, text) {
  const cls = kind === 'online' ? 'online' : kind === 'connecting' ? 'connecting' : kind === 'offline' ? 'offline' : '';
  dom.serverPill.className = `server-pill ${cls}`.trim();
  dom.serverText.textContent = text;
}

function showViewOverlay(title, detail = '') {
  dom.viewOverlayTitle.textContent = title;
  dom.viewOverlayText.textContent = detail;
  dom.loadProgress.hidden = true;
  dom.loadProgressFill.style.width = '0%';
  dom.loadElapsed.hidden = true;
  dom.loadElapsed.textContent = '';
  dom.viewOverlay.hidden = false;
}

function setViewOverlayDetail(detail) {
  dom.viewOverlayText.textContent = detail;
}

function hideViewOverlay() {
  dom.viewOverlay.hidden = true;
  dom.loadProgress.hidden = true;
  dom.loadElapsed.hidden = true;
}

function showViewError(message) {
  dom.viewErrorText.textContent = message;
  dom.viewError.hidden = false;
}

function hideViewError() {
  dom.viewError.hidden = true;
}

function showOffline(detail) {
  if (detail) dom.offlineDetail.textContent = detail;
  dom.offlineOverlay.hidden = false;
  hideViewOverlay();
  dom.viewport.classList.add('is-offline');
}

function hideOffline() {
  dom.offlineOverlay.hidden = true;
  dom.viewport.classList.remove('is-offline');
}

// ===========================================================================
// 后端 API（自动区分 selftest）
// ===========================================================================

function isNetworkError(err) {
  if (!err) return false;
  if (err.name === 'AbortError') return true;
  return /failed to fetch|networkerror|load failed|connection refused|无法连接/i.test(String(err.message || err));
}

function errorMessage(err) {
  if (!err) return '未知错误';
  if (err.name === 'AbortError') return '请求超时（后端可能仍在推理）';
  const msg = String(err.message || err);
  if (/failed to fetch|networkerror|load failed/i.test(msg)) return '无法连接后端服务';
  return msg;
}

async function requestJson(url, options = {}, timeout = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal, headers: { Accept: 'application/json', ...(options.headers || {}) } });
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(`后端返回的不是 JSON（HTTP ${response.status}）`);
      }
    }
    if (!response.ok) throw new Error(data?.error || `HTTP ${response.status}`);
    if (!data || data.ok === false) throw new Error(data?.error || '接口返回 ok=false');
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function apiGet(path, timeout = 30000) {
  return requestJson(path, { method: 'GET' }, timeout);
}

async function apiPost(path, body, timeout = 180000) {
  return requestJson(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    },
    timeout,
  );
}

const api = {
  health: () => (SELFTEST ? selftestApi.health() : apiGet('/api/health', 5000)),
  robots: () => (SELFTEST ? selftestApi.robots() : apiGet('/api/robots', 20000)),
  library: (root) => {
    if (SELFTEST) return selftestApi.library();
    const query = root ? `?root=${encodeURIComponent(root)}` : '';
    return apiGet(`/api/library${query}`, 90000);
  },
  inspect: (path) => (SELFTEST ? selftestApi.inspect(path) : apiGet(`/api/inspect?path=${encodeURIComponent(path)}`, 45000)),
  tracks: (path) =>
    SELFTEST ? selftestApi.tracks(path) : apiPost('/api/tracks', { path, robot: 'auto' }, 180000),
};

// ===========================================================================
// 3D 场景初始化
// ===========================================================================

function initViewer() {
  const renderer = new THREE.WebGLRenderer({
    canvas: dom.canvas,
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070c14);
  scene.fog = new THREE.Fog(0x070c14, 16, 46);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 200);
  camera.position.set(2.6, 2.1, 3.4);
  camera.lookAt(0, 0.9, 0);

  const controls = new OrbitControls(camera, dom.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.minDistance = 0.8;
  controls.maxDistance = 26;
  controls.minPolarAngle = 0.12;
  controls.maxPolarAngle = Math.PI * 0.492; // 限制俯仰：相机不钻到地面以下
  controls.target.set(0, 0.9, 0);
  controls.update();

  // z-up 世界 -> three y-up：机器人/根位姿都挂在 zup 下
  const zup = new THREE.Group();
  zup.rotation.x = -Math.PI / 2;
  scene.add(zup);

  const mount = new THREE.Group();
  mount.name = 'root-mount';
  zup.add(mount);

  const ghostMount = new THREE.Group();
  ghostMount.name = 'ghost-root-mount';
  ghostMount.visible = false;
  zup.add(ghostMount);

  // 环境光 / 方向光 / 补光
  scene.add(new THREE.HemisphereLight(0x8ec5ff, 0x0a0f18, 1.05));
  scene.add(new THREE.AmbientLight(0x5b7a99, 0.55));

  const keyLight = new THREE.DirectionalLight(0xffffff, 2.3);
  keyLight.position.set(3.4, 6.2, 3.0);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 24;
  keyLight.shadow.camera.left = -5;
  keyLight.shadow.camera.right = 5;
  keyLight.shadow.camera.top = 5;
  keyLight.shadow.camera.bottom = -5;
  keyLight.shadow.bias = -0.0008;
  scene.add(keyLight);
  scene.add(keyLight.target);

  const fillLight = new THREE.DirectionalLight(0x7dd3fc, 0.7);
  fillLight.position.set(-4.2, 3.2, -2.6);
  scene.add(fillLight);

  const rimLight = new THREE.PointLight(0x22d3ee, 9, 9, 2);
  rimLight.position.set(0, 2.6, -2.6);
  scene.add(rimLight);

  // 阴影接收平面 + 地面网格 + 科技感光晕环
  const shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.4 }));
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = 0.001;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  const grid = new THREE.GridHelper(16, 32, 0x1e3b55, 0x12202f);
  grid.material.transparent = true;
  grid.material.opacity = 0.92;
  grid.position.y = 0.002;
  scene.add(grid);

  const glow = new THREE.Mesh(
    new THREE.RingGeometry(0.34, 0.42, 64),
    new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false }),
  );
  glow.rotation.x = -Math.PI / 2;
  glow.position.y = 0.004;
  scene.add(glow);
  const glowInner = new THREE.Mesh(
    new THREE.CircleGeometry(0.3, 48),
    new THREE.MeshBasicMaterial({ color: 0x0e7490, transparent: true, opacity: 0.12, depthWrite: false }),
  );
  glowInner.rotation.x = -Math.PI / 2;
  glowInner.position.y = 0.003;
  scene.add(glowInner);

  Object.assign(view, { renderer, scene, camera, controls, zup, mount, ghostMount });

  resizeViewer();
  if ('ResizeObserver' in window) new ResizeObserver(resizeViewer).observe(dom.viewport);
  window.addEventListener('resize', () => {
    resizeViewer();
    if (state.activeTab === 'joints') drawJointCurve(currentFrameIndex());
  });

  requestAnimationFrame(animationLoop);
}

function resizeViewer() {
  if (!view.renderer) return;
  const width = Math.max(1, Math.floor(dom.viewport.clientWidth));
  const height = Math.max(1, Math.floor(dom.viewport.clientHeight));
  view.lastWidth = width;
  view.lastHeight = height;
  view.renderer.setSize(width, height, false);
  view.camera.aspect = width / height;
  view.camera.updateProjectionMatrix();
}

let lastTimestamp = 0;
function animationLoop(timestamp) {
  requestAnimationFrame(animationLoop);
  const dt = lastTimestamp ? Math.min((timestamp - lastTimestamp) / 1000, 0.1) : 0;
  lastTimestamp = timestamp;
  // 布局晚于初始化完成时（CDP/无头窗口尺寸变化）自动补齐画布尺寸
  if (view.renderer) {
    const w = dom.viewport.clientWidth;
    const h = dom.viewport.clientHeight;
    if (w > 0 && h > 0 && (w !== view.lastWidth || h !== view.lastHeight)) resizeViewer();
  }
  tickPlayback(dt);
  if (view.controls) view.controls.update();
  if (view.renderer) view.renderer.render(view.scene, view.camera);
}

/** 相机取景：按当前机器人包围盒自动对焦 */
function resetView() {
  if (!view.zup || !view.camera || !view.controls) return;
  // 必须先把父级（zup 的 -90°X）矩阵刷新到 world，否则 bbox 会算在 z-up 局部坐标里
  if (view.scene) view.scene.updateMatrixWorld(true);
  const box = new THREE.Box3();
  let hasBox = false;
  for (const obj of [view.mount, view.ghostMount]) {
    if (!obj) continue;
    if (obj === view.ghostMount && !view.ghostMount.visible) continue;
    const child = obj.children.find((item) => item.visible !== false);
    if (!child) continue;
    const childBox = new THREE.Box3().setFromObject(child);
    if (childBox.isEmpty()) continue;
    box.union(childBox);
    hasBox = true;
  }
  if (!hasBox) box.set(new THREE.Vector3(-0.6, 0, -0.6), new THREE.Vector3(0.6, 1.8, 0.6));

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 1.0);
  const distance = span * 1.65 + 0.8;
  const dir = new THREE.Vector3(0.82, 0.46, 1.0).normalize();
  view.controls.target.copy(center);
  view.camera.position.copy(center).addScaledVector(dir, distance);
  view.camera.near = Math.max(0.02, span * 0.01);
  view.camera.far = Math.max(80, distance * 12);
  view.camera.updateProjectionMatrix();
  view.controls.update();
}

// ===========================================================================
// 模型加载 / ghost / 占位
// ===========================================================================

function disposeObject3D(root, { disposeGeometry = true } = {}) {
  if (!root) return;
  const materials = new Set();
  root.traverse((obj) => {
    if (obj.isMesh || obj.isLine || obj.isPoints) {
      if (disposeGeometry && obj.geometry) obj.geometry.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of mats) if (mat) materials.add(mat);
    }
  });
  for (const mat of materials) {
    mat.dispose();
    if (mat.map) mat.map.dispose();
  }
}

function clearGhost() {
  if (!view.ghost) return;
  view.ghostMount.remove(view.ghost);
  // ghost 与主模型共享 geometry，不能 dispose，否则主模型也跟着坏掉
  disposeObject3D(view.ghost, { disposeGeometry: false });
  if (view.ghostMaterial) {
    view.ghostMaterial.dispose();
    view.ghostMaterial = null;
  }
  view.ghost = null;
  view.ghostJointMap = [];
  view.ghostMount.visible = false;
}

function clearModelScene() {
  clearGhost();
  if (view.robot) {
    view.mount.remove(view.robot);
    disposeObject3D(view.robot);
    view.robot = null;
  }
  if (view.placeholder) {
    view.mount.remove(view.placeholder);
    disposeObject3D(view.placeholder);
    view.placeholder = null;
  }
  view.robotMeta = null;
  view.jointMap = [];
  view.jointMissing = [];
  view.meshErrors = [];
  view.meshPending = 0;
  view.modelReady = false;
}

function makePlaceholder() {
  // 注意：占位模型挂在 mount 下，而 mount 位于 z-up 的 zup 组内，
  // 所以这里按 URDF 的 z-up 坐标摆放（z 为高度），由 zup 统一转成 y-up。
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.62, metalness: 0.22 });
  const accentMat = new THREE.MeshStandardMaterial({ color: 0x0e7490, emissive: 0x0891b2, emissiveIntensity: 0.5, roughness: 0.4 });
  const add = (geo, mat, x, y, z) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    group.add(mesh);
  };
  add(new THREE.BoxGeometry(0.34, 0.26, 0.2), bodyMat, 0, 0, 0.82); // 骨盆
  add(new THREE.BoxGeometry(0.3, 0.38, 0.54), bodyMat, 0, 0, 1.13); // 躯干
  add(new THREE.BoxGeometry(0.2, 0.22, 0.2), bodyMat, 0, 0, 1.52); // 头
  add(new THREE.BoxGeometry(0.01, 0.14, 0.05), accentMat, 0.102, 0, 1.52); // 面罩
  add(new THREE.BoxGeometry(0.09, 0.09, 0.34), bodyMat, 0, 0.245, 1.12); // 左臂
  add(new THREE.BoxGeometry(0.09, 0.09, 0.34), bodyMat, 0, -0.245, 1.12); // 右臂
  add(new THREE.BoxGeometry(0.11, 0.11, 0.8), bodyMat, 0, 0.085, 0.4); // 左腿
  add(new THREE.BoxGeometry(0.11, 0.11, 0.8), bodyMat, 0, -0.085, 0.4); // 右腿
  return group;
}

/**
 * 加载机器人 URDF，返回 Promise<boolean>。
 * 网格全部就绪（或失败）后才算 ready，保证相机取景/截图稳定。
 */
function loadModel(robotMeta) {
  if (!robotMeta || !robotMeta.urdf) {
    toast('模型信息缺少 urdf 路径', 'warn');
    return Promise.resolve(false);
  }
  const token = ++view.modelToken;
  clearModelScene();
  hideViewError();
  state.modelId = robotMeta.id || null;
  view.robotMeta = robotMeta;
  view.modelName = robotMeta.name || robotMeta.id || basename(robotMeta.urdf);
  view.modelLoading = true;
  if (dom.modelSelect.value !== state.modelId) dom.modelSelect.value = state.modelId || '';
  showViewOverlay(`正在加载模型：${view.modelName}`, robotMeta.urdf);

  let pendingMeshes = 0;
  let meshTotal = 0;
  let meshDone = 0;
  let urdfLoaded = false;
  let settled = false;
  let resolveFn = null;
  const loadStartedAt = performance.now();
  const promise = new Promise((resolve) => {
    resolveFn = resolve;
  });

  // 大模型（X2 URDF+STL 约 89MB）加载进度：网格计数 + 进度条 + 已等待秒数
  const updateLoadProgress = (note) => {
    if (settled) return;
    const elapsed = (performance.now() - loadStartedAt) / 1000;
    if (meshTotal > 0) {
      dom.loadProgress.hidden = false;
      const percent = Math.min(100, (meshDone / meshTotal) * 100);
      dom.loadProgressFill.style.width = `${percent.toFixed(1)}%`;
      dom.viewOverlayText.textContent = `已加载网格 ${meshDone}/${meshTotal}（${percent.toFixed(0)}%）${note ? ` · ${note}` : ''}（URDF+STL 大模型首次加载较慢，请稍候）`;
    } else if (note) {
      dom.viewOverlayText.textContent = note;
    }
    dom.loadElapsed.hidden = false;
    dom.loadElapsed.textContent = `已等待 ${elapsed.toFixed(0)} 秒 · 模型加载完成后自动开始播放`;
  };
  const progressTimer = setInterval(() => updateLoadProgress(''), 500);

  const finish = (ok, err) => {
    if (settled) return;
    settled = true;
    clearTimeout(watchdog);
    clearInterval(progressTimer);
    view.modelLoading = false;
    resolveFn(ok);
    if (!ok) {
      hideViewOverlay();
      view.modelReady = false;
      view.robot = null;
      view.placeholder = makePlaceholder();
      view.mount.add(view.placeholder);
      view.jointMap = [];
      view.jointMissing = state.trackData?.jointNames?.slice() || [];
      const msg = errorMessage(err);
      showViewError(`模型加载失败：${msg}\n已降级为占位模型，播放条与信息面板仍可用。`);
      toast(`模型加载失败：${msg}`, 'error', 4800);
      renderJointPanel();
      resetView();
      markAppReady();
    }
  };

  // 被更新的模型加载请求取代：静默取消，不弹错误、不降级占位
  const cancelQuietly = () => {
    if (settled) return;
    settled = true;
    clearTimeout(watchdog);
    clearInterval(progressTimer);
    view.modelLoading = false;
    resolveFn(false);
    markAppReady();
  };

  const markReady = () => {
    if (settled || !urdfLoaded || pendingMeshes > 0) return;
    clearInterval(progressTimer);
    view.modelReady = true;
    view.modelLoading = false;
    if (view.meshErrors.length) {
      toast(`模型已加载，但有 ${view.meshErrors.length} 个网格失败（模型可能不完整），详见控制台`, 'warn', 4200);
      for (const meshPath of view.meshErrors.slice(0, 12)) console.warn('[ONNX 播放器] 网格加载失败：', meshPath);
    }
    hideViewOverlay();
    rebuildJointMap();
    rebuildGhost();
    if (state.trackData) applyFrame(currentFrameIndex());
    resetView();
    renderJointPanel();
    renderMetadata();
    finish(true);
    markAppReady();
  };

  const manager = new THREE.LoadingManager();
  const loader = new URDFLoader(manager);
  // package:// 兜底：把包名映射到 /models/<pkg>（若后端 URDF 用 package:// 且目录结构不同，需要后端给绝对路径）
  loader.packages = (pkg) => `/models/${pkg}`;
  loader.loadMeshCb = (path, mgr, material, meshCallback) => {
    pendingMeshes += 1;
    meshTotal += 1;
    updateLoadProgress('解析网格路径…');
    let doneFlag = false;
    const settleMesh = (obj, err) => {
      if (doneFlag) return;
      doneFlag = true;
      clearTimeout(meshGuard);
      pendingMeshes = Math.max(0, pendingMeshes - 1);
      meshDone += 1;
      if (err) view.meshErrors.push(String(path));
      meshCallback(obj, err);
      updateLoadProgress('');
      markReady();
    };
    // 默认 loader 只认 STL/DAE，其它格式不会回调 -> 20s 兜底
    const meshGuard = setTimeout(() => settleMesh(null, new Error(`网格加载超时：${path}`)), 20000);
    loader.defaultMeshLoader(path, mgr, material, (obj, err) => settleMesh(obj, err));
  };

  const watchdog = setTimeout(() => {
    if (settled) return;
    if (urdfLoaded && view.robot) {
      toast('模型网格仍在加载（超过 25 秒），先展示已加载部分', 'warn', 4200);
      pendingMeshes = 0;
      markReady();
    } else {
      finish(false, new Error('模型加载超时（25 秒）'));
    }
  }, 25000);

  loader.load(
    robotMeta.urdf,
    (model) => {
      if (token !== view.modelToken) {
        disposeObject3D(model);
        cancelQuietly();
        return;
      }
      urdfLoaded = true;
      view.robot = model;
      model.traverse((obj) => {
        if (obj.isMesh) {
          obj.castShadow = true;
          obj.receiveShadow = false;
          obj.frustumCulled = true;
        }
      });
      view.mount.add(model);
      updateLoadProgress(meshTotal > 0 ? 'URDF 解析完成，正在加载网格…' : 'URDF 解析完成');
      markReady();
    },
    undefined,
    (err) => {
      if (token !== view.modelToken) {
        cancelQuietly();
        return;
      }
      finish(false, err);
    },
  );

  return promise;
}

function rebuildJointMap() {
  view.jointMap = [];
  view.jointMissing = [];
  const fd = state.trackData;
  if (!fd) return;
  const names = fd.jointNames || [];
  if (!view.robot) {
    view.jointMissing = names.slice();
    return;
  }
  for (let i = 0; i < names.length; i += 1) {
    const name = names[i];
    const joint = view.robot.joints ? view.robot.joints[name] : null;
    if (joint && joint.jointType !== 'fixed') view.jointMap.push({ index: i, name, joint });
    else view.jointMissing.push(name);
  }
  if (view.jointMissing.length) {
    toast(`当前模型缺少 ${view.jointMissing.length} 个轨迹关节，对应关节将忽略`, 'warn', 4200);
  }
}

function rebuildGhost() {
  clearGhost();
  const fd = state.trackData;
  const hasGhost = Boolean(fd && fd.tracks && fd.tracks.length > 1 && view.robot);
  if (!hasGhost) {
    dom.ghostToggleWrap.hidden = true;
    return;
  }
  dom.ghostToggleWrap.hidden = false;
  dom.ghostToggle.checked = state.ghostVisible;
  if (!state.ghostVisible) return;

  const ghost = view.robot.clone(true);
  const material = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    emissive: 0x0ea5e9,
    emissiveIntensity: 0.7,
    transparent: true,
    opacity: 0.34,
    roughness: 0.45,
    metalness: 0.1,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  ghost.traverse((obj) => {
    if (obj.isMesh) {
      obj.material = material;
      obj.castShadow = false;
      obj.receiveShadow = false;
      obj.renderOrder = 4;
    }
  });
  view.ghost = ghost;
  view.ghostMaterial = material;
  view.ghostMount.add(ghost);
  view.ghostMount.visible = true;

  view.ghostJointMap = [];
  const fdNames = fd.jointNames || [];
  for (let i = 0; i < fdNames.length; i += 1) {
    const name = fdNames[i];
    const joint = ghost.joints ? ghost.joints[name] : null;
    if (joint && joint.jointType !== 'fixed') view.ghostJointMap.push({ index: i, name, joint });
  }
}

// ===========================================================================
// 轨迹数据 / 逐帧应用
// ===========================================================================

function currentFrameIndex() {
  if (!state.trackData) return 0;
  return clamp(Math.round(play.time * play.fps), 0, Math.max(0, play.frameCount - 1));
}

function applyTrackPose(track, frame, mount, jointMap) {
  if (!track) return;
  const pos = track.rootPos ? track.rootPos[frame] : null;
  const quat = track.rootQuat ? track.rootQuat[frame] : null;
  if (mount) {
    if (pos && pos.length >= 3) mount.position.set(pos[0], pos[1], pos[2]);
    if (quat && quat.length >= 4) {
      // 契约 [w, x, y, z] -> three 的 set(x, y, z, w)
      mount.quaternion.set(quat[1], quat[2], quat[3], quat[0]).normalize();
    }
  }
  const row = track.joints ? track.joints[frame] : null;
  if (row && jointMap && jointMap.length) {
    for (const item of jointMap) {
      const value = row[item.index];
      if (Number.isFinite(value)) item.joint.setJointValue(value);
    }
  }
}

function applyFrame(frameInput) {
  const fd = state.trackData;
  if (!fd) return;
  const frame = clamp(Math.round(frameInput), 0, Math.max(0, play.frameCount - 1));
  const primary = fd.tracks[state.primaryIndex];
  applyTrackPose(primary, frame, view.mount, view.jointMap);
  if (view.ghost && state.ghostIndex >= 0) {
    applyTrackPose(fd.tracks[state.ghostIndex], frame, view.ghostMount, view.ghostJointMap);
  }
  play.lastFrame = frame;
  updateTransportUi(frame);
  if (state.activeTab === 'joints') updateJointReadout(frame);
}

function setTrackData(data) {
  state.trackData = data;
  play.fps = Number(data.fps) > 0 ? Number(data.fps) : 50;
  const firstRoot = data.tracks?.[0]?.rootPos;
  play.frameCount = Math.max(1, Math.round(Number(data.frameCount) || firstRoot?.length || 1));
  play.duration = Number(data.duration) > 0 ? Number(data.duration) : play.frameCount / play.fps;
  play.time = 0;
  play.playing = false;
  play.lastFrame = -1;
  play.scrubbing = false;

  const tracks = data.tracks || [];
  const visibleIndex = tracks.findIndex((track) => track.defaultVisible !== false);
  state.primaryIndex = visibleIndex >= 0 ? visibleIndex : 0;
  state.ghostIndex = tracks.length > 1 ? (state.primaryIndex === 0 ? 1 : 0) : -1;
  state.ghostVisible = state.ghostIndex >= 0 ? tracks[state.ghostIndex].defaultVisible === true : false;

  dom.timeline.min = '0';
  dom.timeline.max = String(play.frameCount - 1);
  dom.timeline.value = '0';
  dom.timeline.disabled = false;
  setPlaying(false);
  updateTransportUi(0);
  updateNowPlaying();
}

function setPlaying(next) {
  const canPlay = Boolean(state.trackData);
  play.playing = Boolean(next) && canPlay;
  dom.btnPlay.textContent = play.playing ? '⏸' : '▶';
  dom.btnPlay.title = play.playing ? '暂停（空格）' : '播放（空格）';
  dom.btnPlay.classList.toggle('playing', play.playing);
  if (play.playing) {
    if (play.time >= play.duration) play.time = 0;
    applyFrame(currentFrameIndex());
  }
}

function seekFrame(frame) {
  if (!state.trackData) return;
  const target = clamp(Math.round(frame), 0, play.frameCount - 1);
  play.time = target / play.fps;
  applyFrame(target);
}

function seekTime(time) {
  if (!state.trackData) return;
  play.time = clamp(Number(time) || 0, 0, play.duration);
  applyFrame(currentFrameIndex());
}

function tickPlayback(dt) {
  if (!state.trackData || !play.playing || play.scrubbing) return;
  play.time += dt * play.speed;
  if (play.time >= play.duration) {
    if (play.loop) {
      play.time = play.duration > 0 ? play.time % play.duration : 0;
    } else {
      play.time = play.duration;
      setPlaying(false);
    }
  }
  const frame = currentFrameIndex();
  if (frame !== play.lastFrame) applyFrame(frame);
}

function updateTransportUi(frame) {
  const safeFrame = clamp(frame, 0, Math.max(0, play.frameCount - 1));
  dom.timeCurrent.textContent = fmtTime(play.frameCount > 0 ? safeFrame / play.fps : 0);
  dom.timeTotal.textContent = fmtTime(play.duration);
  if (document.activeElement !== dom.timeline || !play.scrubbing) {
    dom.timeline.value = String(safeFrame);
  }
  const percent = play.frameCount > 1 ? (safeFrame / (play.frameCount - 1)) * 100 : 0;
  dom.timeline.style.setProperty('--fill', `${percent.toFixed(2)}%`);
  dom.fpsInfo.innerHTML = `${play.fps} fps · ${play.frameCount} 帧<span class="dim"> · 空格播放 · ←→逐帧</span>`;
}

function updateNowPlaying() {
  const entry = state.entry;
  if (!entry) {
    dom.nowPlaying.textContent = view.modelReady ? `模型：${view.modelName}` : '未选择资源';
    return;
  }
  const fd = state.trackData;
  const trackText = fd ? `${fd.tracks.length} 条轨迹 · ${play.fps} fps` : '';
  dom.nowPlaying.innerHTML =
    `<span class="np-robot">${escapeHtml(robotLabel(entry.robot))}</span>${escapeHtml(entry.name || basename(entry.path))}` +
    (trackText ? `<span class="np-track">${escapeHtml(trackText)}</span>` : '');
}

// ===========================================================================
// 资源库
// ===========================================================================

function normalizeEntry(raw) {
  const entry = raw && typeof raw === 'object' ? raw : {};
  return {
    ...entry,
    path: String(entry.path || ''),
    name: entry.name || basename(entry.path) || '未命名资源',
    robot: robotKey(entry.robot),
    kind: kindLabel(entry.kind) === '动作' ? 'motion' : kindLabel(entry.kind) === '行走策略' ? 'locomotion' : 'unknown',
    playable: Boolean(entry.playable),
    duration: Number(entry.meta?.duration) || null,
  };
}

function updateLibHint() {
  const hint = dom.libHint;
  hint.className = 'lib-hint';
  hint.textContent = '';
  if (state.selftest) {
    hint.textContent = '自检模式 · 内置 mock 数据（不请求后端）';
    return;
  }
  if (state.libraryLoading) {
    hint.classList.add('loading');
    hint.textContent = '正在扫描资源目录…';
    return;
  }
  if (state.libraryError) {
    hint.classList.add('error');
    hint.append(document.createTextNode(`资源库加载失败：${state.libraryError}`));
    const retry = document.createElement('span');
    retry.className = 'retry-link';
    retry.textContent = '重试';
    retry.addEventListener('click', () => loadLibrary(state.root));
    hint.append(retry);
    return;
  }
  if (!state.online && !state.selftest) {
    hint.classList.add('error');
    hint.textContent = '后端未连接，无法读取资源库。';
    return;
  }
  const playable = state.entries.filter((entry) => entry.playable).length;
  hint.textContent = `扫描到 ${state.entries.length} 个资源 · 可播放 ${playable} 个 · 点击条目开始播放`;
}

function makeEntryElement(entry) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'entry';
  btn.dataset.path = entry.path;
  btn.title = entry.path;
  if (state.entry && state.entry.path === entry.path) btn.classList.add('active');
  if (!entry.playable) btn.classList.add('disabled');
  if (state.loadingEntryPath === entry.path) btn.classList.add('loading');

  const line1 = document.createElement('div');
  line1.className = 'entry-line1';
  const name = document.createElement('span');
  name.className = 'entry-name';
  name.textContent = entry.name;
  line1.append(name);

  const robotBadge = document.createElement('span');
  const rk = robotKey(entry.robot);
  robotBadge.className = `badge ${rk === 'x2' ? 'b-x2' : rk === 'a3' ? 'b-a3' : 'b-unknown'}`;
  robotBadge.textContent = rk === 'x2' ? 'X2' : rk === 'a3' ? 'A3' : '?';
  line1.append(robotBadge);

  const kindBadge = document.createElement('span');
  kindBadge.className = `badge ${entry.kind === 'motion' ? 'b-motion' : entry.kind === 'locomotion' ? 'b-locomotion' : 'b-unknown-kind'}`;
  kindBadge.textContent = kindLabel(entry.kind);
  line1.append(kindBadge);

  const line2 = document.createElement('div');
  line2.className = 'entry-line2';
  const metaParts = [];
  if (entry.duration) metaParts.push(`${entry.duration.toFixed(2)}s`);
  metaParts.push(fmtSize(entry.size));
  const metaSpan = document.createElement('span');
  metaSpan.textContent = metaParts.join(' · ');
  line2.append(metaSpan);

  const sep = document.createElement('span');
  sep.className = 'dot-sep';
  sep.textContent = '·';
  line2.append(sep);

  const playTag = document.createElement('span');
  playTag.className = entry.playable ? 'playable-tag' : 'playable-tag no';
  playTag.textContent = entry.playable ? '可播放' : entry.kind === 'locomotion' ? '暂不支持' : '不可播放';
  line2.append(playTag);

  btn.append(line1, line2);
  return btn;
}

function renderLibrary() {
  const list = dom.entryList;
  const scrollTop = list.scrollTop;
  list.textContent = '';

  const query = state.search.trim().toLowerCase();
  const entries = state.entries.filter((entry) => {
    if (!query) return true;
    return String(entry.name || '').toLowerCase().includes(query) || String(entry.path || '').toLowerCase().includes(query);
  });

  const groups = [
    { key: 'x2', title: '灵犀 X2', items: [] },
    { key: 'a3', title: '远征 A3', items: [] },
    { key: 'unknown', title: '其它 / 未知', items: [] },
  ];
  for (const entry of entries) {
    const group = groups.find((item) => item.key === robotKey(entry.robot)) || groups[2];
    group.items.push(entry);
  }

  let rendered = 0;
  for (const group of groups) {
    if (!group.items.length) continue;
    const title = document.createElement('div');
    title.className = 'entry-group-title';
    const label = document.createElement('span');
    label.textContent = group.title;
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = String(group.items.length);
    title.append(label, count);
    list.append(title);
    for (const entry of group.items) list.append(makeEntryElement(entry));
    rendered += group.items.length;
  }

  if (!rendered && !state.libraryLoading) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = query
      ? `没有匹配「${state.search.trim()}」的资源。`
      : state.libraryError
        ? '资源库为空或扫描失败，点上方「刷新」重试。'
        : '暂无资源。把 ONNX 资源放到默认根目录后点「刷新」。';
    list.append(empty);
  }

  dom.libCount.textContent = `共 ${state.entries.length} 个资源${query ? ` · 匹配 ${rendered} 个` : ''}`;
  updateLibHint();
  list.scrollTop = scrollTop;
}

function populateRoots() {
  dom.rootSelect.textContent = '';
  const all = document.createElement('option');
  all.value = '';
  all.textContent = '全部根目录';
  dom.rootSelect.append(all);
  for (const root of state.roots) {
    const option = document.createElement('option');
    option.value = root.path || '';
    option.textContent = root.label || root.path || '';
    dom.rootSelect.append(option);
  }
  dom.rootSelect.value = state.root || '';
}

function populateModelSelect() {
  dom.modelSelect.textContent = '';
  if (!state.robots.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = state.selftest ? '（自检无模型）' : '（后端未提供模型）';
    dom.modelSelect.append(option);
    dom.modelSelect.disabled = true;
    return;
  }
  dom.modelSelect.disabled = false;
  for (const robot of state.robots) {
    const option = document.createElement('option');
    option.value = robot.id;
    option.textContent = robot.name || robot.id;
    dom.modelSelect.append(option);
  }
  dom.modelSelect.value = state.modelId || state.robots[0].id;
}

async function loadLibrary(root = state.root) {
  if (!SELFTEST && !state.online) {
    renderLibrary();
    return;
  }
  state.root = root || '';
  state.libraryLoading = true;
  state.libraryError = '';
  renderLibrary();
  try {
    const data = await api.library(state.root);
    state.entries = (data.entries || []).map(normalizeEntry);
    state.libraryLoaded = true;
    state.libraryError = '';
  } catch (err) {
    state.libraryError = errorMessage(err);
    state.entries = [];
    state.libraryLoaded = false;
    if (isNetworkError(err) && !SELFTEST) {
      state.online = false;
      setServerStatus('offline', '后端未连接');
      showOffline(`读取资源库失败：${state.libraryError}\n请确认后端正在运行（默认 http://127.0.0.1:8766），然后重试。`);
    }
  } finally {
    state.libraryLoading = false;
    renderLibrary();
    markAppReady();
  }
}

function pickRobotForTracks(trackData) {
  const robots = state.robots || [];
  if (!robots.length) return null;
  const hint = String(trackData?.modelHint || '');
  let match = robots.find((robot) => robot.id === hint);
  if (!match && hint) match = robots.find((robot) => hint.includes(robot.id) || robot.id.includes(hint));
  if (!match && trackData?.robot) {
    const robotParam = String(trackData.robot);
    match = robots.find((robot) => robot.id === robotParam) || robots.find((robot) => robot.id.includes(robotParam) || robotParam.includes(robot.id));
  }
  if (!match) {
    const robotKeyWanted = robotKey(trackData?.robot || hint);
    match = robots.find((robot) => robotKey(robot.id).includes(robotKeyWanted) || robotKey(robot.name).includes(robotKeyWanted));
  }
  return match || robots[0];
}

async function selectEntry(entry) {
  if (!entry) return;
  if (!entry.playable) {
    toast(
      entry.kind === 'locomotion'
        ? '该资源是行走/速度策略，播放器暂不支持轨迹回放'
        : '该资源不可播放（缺参考动作或无法识别 ONNX 输出）',
      'warn',
      4200,
    );
    return;
  }
  const token = ++state.resourceToken;
  state.loadingEntryPath = entry.path;
  state.lastResourceError = '';
  renderLibrary();
  showViewOverlay(`正在检查资源：${entry.name}`, entry.path);

  try {
    const inspect = await api.inspect(entry.path);
    if (token !== state.resourceToken) return;
    if (inspect && inspect.playable === false) {
      throw new Error(`该资源不可播放（inspect.playable=false，kind=${inspect.kind || entry.kind || 'unknown'}）`);
    }
    state.entry = normalizeEntry({ ...entry, ...pickInspectFields(inspect) });
    state.inspect = inspect;
    renderMetadata();

    setViewOverlayDetail('正在生成播放轨迹：ONNX 逐帧推理可能需要几秒…');
    const trackData = await api.tracks(entry.path);
    if (token !== state.resourceToken) return;
    await activateTracks(entry, inspect, trackData);
  } catch (err) {
    if (token !== state.resourceToken) return;
    hideViewOverlay();
    const msg = errorMessage(err);
    state.lastResourceError = msg;
    showViewError(`加载「${entry.name}」失败：${msg}`);
    toast(`加载失败：${msg}`, 'error', 4600);
    if (isNetworkError(err) && !SELFTEST) {
      state.online = false;
      setServerStatus('offline', '后端未连接');
      showOffline(`后端连接中断：${msg}\n请确认服务仍在运行，然后重试。`);
    }
  } finally {
    if (token === state.resourceToken) {
      state.loadingEntryPath = '';
      hideViewOverlay();
      renderLibrary();
    }
  }
}

function pickInspectFields(inspect) {
  if (!inspect) return {};
  return {
    size: inspect.size,
    robot: inspect.robot,
    kind: inspect.kind,
    playable: inspect.playable,
    path: inspect.path,
  };
}

async function activateTracks(entry, inspect, rawData) {
  const data = rawData || {};
  const tracks = Array.isArray(data.tracks) ? data.tracks.filter((track) => track && Array.isArray(track.rootPos)) : [];
  if (!tracks.length) throw new Error('后端返回的轨迹为空（tracks 无 rootPos 逐帧数据）');
  data.tracks = tracks;
  data.frameCount = Math.max(1, Math.round(Number(data.frameCount) || tracks[0].rootPos.length || 1));
  data.fps = Number(data.fps) > 0 ? Number(data.fps) : 50;
  if (!Array.isArray(data.jointNames)) data.jointNames = [];

  // 同一资源切换：先把旧轨迹停掉，避免旧帧继续驱动
  state.trackData = null;
  state.primaryIndex = 0;
  state.ghostIndex = -1;

  const robot = pickRobotForTracks(data);
  if (robot && robot.id !== state.modelId) {
    showViewOverlay(`正在切换模型：${robot.name || robot.id}`, `modelHint: ${data.modelHint || data.robot || 'auto'}`);
    await loadModel(robot);
    const stillCurrent = state.entry && state.entry.path === entry.path;
    if (!stillCurrent && state.entry) return;
  } else if (!view.robot && !view.placeholder && state.robots.length) {
    await loadModel(state.robots[0]);
  }

  setTrackData(data);
  rebuildJointMap();
  rebuildGhost();
  applyFrame(0);
  // 模型是在没有轨迹数据时加载的，markReady 里的取景基于默认姿态；这里按首帧位姿重新取景
  resetView();
  setPlaying(true);
  renderMetadata();
  renderJointPanel();
  updateNowPlaying();
  hideViewOverlay();
  hideViewError();
  state.manualError = '';

  const missingNote = view.jointMissing.length ? `，${view.jointMissing.length} 个关节模型缺失` : '';
  toast(`开始播放：${entry.name} · ${data.tracks.length} 条轨迹${missingNote}`, 'info', 2400);
  markAppReady();
}

function toggleGhost(visible) {
  state.ghostVisible = Boolean(visible);
  if (!state.ghostVisible) {
    if (view.ghostMount) view.ghostMount.visible = false;
  } else {
    if (!view.ghost && view.robot && state.trackData) rebuildGhost();
    if (view.ghostMount) view.ghostMount.visible = Boolean(view.ghost);
  }
  applyFrame(currentFrameIndex());
}

/**
 * 按绝对路径加载资源（供 ?path= 与 window.__loadResource 使用）。
 * 资源库里有条目就复用（保留元数据），否则构造临时条目再走 inspect+tracks。
 * @returns {Promise<boolean>} 是否成功生成并开始播放轨迹
 */
async function loadResourceByPath(path) {
  const target = String(path || '').trim();
  if (!target) throw new Error('path 参数为空');
  let entry = state.entries.find((item) => item.path === target) || null;

  // 当前资源库（某个 root）里没有该 path 时，退一步用默认根目录全量扫一次
  if (!entry && state.online) {
    try {
      const data = await api.library('');
      const list = (data.entries || []).map(normalizeEntry);
      const merged = new Map(state.entries.map((item) => [item.path, item]));
      for (const item of list) if (!merged.has(item.path)) merged.set(item.path, item);
      state.entries = [...merged.values()];
      renderLibrary();
      entry = state.entries.find((item) => item.path === target) || null;
    } catch (err) {
      console.warn('[ONNX 播放器] 按 path 加载前扫描资源库失败：', err);
    }
  }

  if (!entry) {
    entry = normalizeEntry({ path: target, name: basename(target), robot: 'unknown', kind: 'motion', playable: true });
  }
  await selectEntry(entry);
  return Boolean(state.trackData && state.entry && state.entry.path === target);
}

// 联调辅助：window.__loadResource('<绝对路径>') 等价于在资源库点击该资源
// （模块加载后即可用，不依赖初始化完成；selectEntry 内部会等模型/接口就绪）
window.__loadResource = (path) =>
  loadResourceByPath(path)
    .then((ok) => {
      state.manualError = ok ? '' : state.lastResourceError || '未能生成播放轨迹（后端 /api/tracks 校验失败，详见提示）';
      renderMetadata();
      return ok;
    })
    .catch((err) => {
      const msg = errorMessage(err);
      state.manualError = msg;
      renderMetadata();
      toast(`加载资源失败：${msg}`, 'error', 5200);
      showViewError(`加载资源失败：${msg}`);
      return false;
    });

/** 资源库「手动路径」：加载不在 config.json 根目录里的 ONNX 文件 */
async function loadManualPath() {
  const raw = normalizeManualPath(dom.manualPathInput.value);
  if (!raw) {
    toast('先在「手动路径」输入框里粘贴 ONNX 绝对路径', 'warn');
    dom.manualPathInput.focus();
    return;
  }
  if (!SELFTEST && !state.online) {
    toast('后端未连接，先点顶栏「重试」连接后再加载', 'warn', 3400);
    return;
  }
  dom.manualPathInput.value = raw;
  const button = dom.btnLoadPath;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '加载中…';
  try {
    const ok = await window.__loadResource(raw);
    if (ok) {
      toast(`已按手动路径加载：${basename(raw)}`, 'info', 2200);
    } else if (!state.manualError) {
      state.manualError = state.lastResourceError || '手动路径加载失败（后端未生成播放轨迹）';
    }
  } catch (err) {
    state.manualError = errorMessage(err);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
    renderMetadata();
  }
}

// ===========================================================================
// 右侧：元数据面板
// ===========================================================================

function renderMetadata() {
  const body = dom.metaBody;
  const entry = state.entry;
  const inspect = state.inspect;
  const fd = state.trackData;
  if (!entry) {
    const manualWarning = state.manualError
      ? `<section class="card"><h3>warnings <span class="tag">1</span></h3><ul class="warn-list"><li>手动路径加载失败：${escapeHtml(state.manualError)}</li></ul></section>`
      : '';
    body.innerHTML = `<p class="empty">从左侧资源库选择一个可播放的 ONNX 资源，或在「手动路径」里粘贴 ONNX 绝对路径加载（不在根目录里的文件也能播）。</p>${manualWarning}`;
    return;
  }

  const robot = inspect?.robot || entry.robot || 'unknown';
  const kind = inspect?.kind || entry.kind || 'unknown';
  const playable = inspect?.playable ?? entry.playable;
  const size = inspect?.size ?? entry.size;
  const path = inspect?.path || entry.path || '';
  const duration = fd?.duration ?? entry.duration;
  const parts = [];

  // 资源概览
  parts.push(`
    <section class="card">
      <h3>资源 <span class="tag">${escapeHtml(robotLabel(robot))}</span></h3>
      <p class="entry-title">${escapeHtml(entry.name || basename(path))}</p>
      <div class="entry-sub">
        ${robotBadgeHtml(robot)}
        ${kindBadgeHtml(kind)}
        ${playableHtml(playable)}
      </div>
      <div class="kv" style="margin-top:10px">
        <div class="kv-row"><span class="k">路径</span><span class="v">${escapeHtml(path)}</span></div>
        <div class="kv-row"><span class="k">大小</span><span class="v">${fmtSize(size)}</span></div>
        <div class="kv-row"><span class="k">robot</span><span class="v">${escapeHtml(robot)}</span></div>
        <div class="kv-row"><span class="k">kind</span><span class="v">${escapeHtml(kind)} · ${escapeHtml(kindLabel(kind))}</span></div>
        ${duration ? `<div class="kv-row"><span class="k">时长</span><span class="v">${fmtTime(Number(duration))} s</span></div>` : ''}
      </div>
    </section>
  `);

  // 输入 / 输出
  const inputs = Array.isArray(inspect?.inputs) ? inspect.inputs : [];
  const outputs = Array.isArray(inspect?.outputs) ? inspect.outputs : [];
  if (inputs.length || outputs.length) {
    const rows = [];
    for (const item of inputs) {
      rows.push(`<tr><td class="io-name">${escapeHtml(item.name || '?')}</td><td>${escapeHtml(formatShape(item.shape))}</td><td>${escapeHtml(item.dtype || '?')}</td><td>输入</td></tr>`);
    }
    for (const item of outputs) {
      rows.push(`<tr><td class="io-name">${escapeHtml(item.name || '?')}</td><td>${escapeHtml(formatShape(item.shape))}</td><td>${escapeHtml(item.dtype || '?')}</td><td>输出</td></tr>`);
    }
    parts.push(`
      <section class="card">
        <h3>输入 / 输出 <span class="tag">${inputs.length} in · ${outputs.length} out</span></h3>
        <table class="io-table">
          <thead><tr><th>name</th><th>shape</th><th>dtype</th><th>方向</th></tr></thead>
          <tbody>${rows.join('')}</tbody>
        </table>
      </section>
    `);
  }

  // ONNX metadata_props 键值表
  const metadata = inspect?.metadata && typeof inspect.metadata === 'object' ? inspect.metadata : {};
  const metaKeys = Object.keys(metadata);
  parts.push(`
    <section class="card">
      <h3>metadata <span class="tag">${metaKeys.length} 项</span></h3>
      ${metaKeys.length
        ? `<div class="scroll-box"><div class="kv">${metaKeys
            .map((key) => {
              const value = String(metadata[key] ?? '');
              return `<div class="kv-row"><span class="k">${escapeHtml(key)}</span><span class="v" title="${escapeHtml(value)}">${escapeHtml(value.length > 500 ? `${value.slice(0, 500)}…` : value)}</span></div>`;
            })
            .join('')}</div></div>`
        : '<p class="empty">该 ONNX 没有 metadata_props。</p>'}
    </section>
  `);

  // 轨迹信息
  if (fd) {
    const trackItems = fd.tracks
      .map((track, index) => {
        const role = index === state.primaryIndex ? '主轨（驱动模型）' : index === state.ghostIndex ? 'ghost（叠加）' : '';
        const swatch = index === state.primaryIndex ? 'primary' : 'ghost';
        return `<div class="track-item"><span class="swatch ${swatch}"></span><span class="t-id">${escapeHtml(track.id || `#${index}`)}</span><span class="t-label">${escapeHtml(track.label || '')}</span><span class="t-role">${escapeHtml(role)}</span></div>`;
      })
      .join('');
    const trackMeta = fd.meta && typeof fd.meta === 'object' ? fd.meta : {};
    const trackMetaText = Object.keys(trackMeta).length
      ? Object.entries(trackMeta)
          .map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : String(value)}`)
          .join(' · ')
      : '—';
    parts.push(`
      <section class="card">
        <h3>播放轨迹 <span class="tag">${fd.tracks.length} 条</span></h3>
        <div class="track-legend">${trackItems}</div>
        <div class="kv" style="margin-top:9px">
          <div class="kv-row"><span class="k">fps</span><span class="v">${play.fps}</span></div>
          <div class="kv-row"><span class="k">帧数</span><span class="v">${fd.frameCount}（当前帧 ${currentFrameIndex() + 1}）</span></div>
          <div class="kv-row"><span class="k">时长</span><span class="v">${fmtTime(fd.duration)} s</span></div>
          <div class="kv-row"><span class="k">modelHint</span><span class="v">${escapeHtml(fd.modelHint || '—')}</span></div>
          <div class="kv-row"><span class="k">meta</span><span class="v">${escapeHtml(trackMetaText)}</span></div>
        </div>
      </section>
    `);
  } else if (entry.playable) {
    parts.push('<section class="card"><p class="empty">点击左侧资源加载播放轨迹。</p></section>');
  }

  // warnings（手动路径失败 + inspect + tracks + 模型网格）
  const warnings = [];
  if (state.manualError) warnings.push(`手动路径加载失败：${state.manualError}`);
  for (const item of inspect?.warnings || []) warnings.push(String(item));
  for (const item of fd?.warnings || []) warnings.push(String(item));
  for (const key of Object.keys(metadata)) {
    if (/warn|error|fail/i.test(key) && String(metadata[key]).trim()) warnings.push(`${key}: ${metadata[key]}`);
  }
  if (view.meshErrors.length) warnings.push(`模型网格加载失败 ${view.meshErrors.length} 个，3D 显示可能不完整。`);
  if (view.jointMissing.length) warnings.push(`当前模型缺少 ${view.jointMissing.length} 个轨迹关节：${view.jointMissing.slice(0, 6).join(', ')}${view.jointMissing.length > 6 ? '…' : ''}`);
  if (warnings.length) {
    parts.push(`
      <section class="card">
        <h3>warnings <span class="tag">${warnings.length}</span></h3>
        <ul class="warn-list">${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </section>
    `);
  }

  body.innerHTML = parts.join('');
}

function formatShape(shape) {
  if (Array.isArray(shape)) return `[${shape.join(', ')}]`;
  if (shape == null) return '?';
  return String(shape);
}

// ===========================================================================
// 右侧：关节面板 + 曲线
// ===========================================================================

function jointTrackRow() {
  const fd = state.trackData;
  if (!fd) return null;
  return fd.tracks[state.primaryIndex]?.joints || null;
}

function jointNameAt(index) {
  return state.trackData?.jointNames?.[index] || '';
}

function isJointMissing(name) {
  if (!view.robot) return true;
  const joint = view.robot.joints ? view.robot.joints[name] : null;
  return !joint || joint.jointType === 'fixed';
}

function renderJointPanel() {
  const fd = state.trackData;
  const list = dom.jointList;
  list.textContent = '';
  state.jointRows = new Map();

  if (!fd || !fd.jointNames?.length) {
    dom.jointsCount.textContent = '0 个关节';
    dom.jointsFrame.textContent = '帧 - / -';
    dom.curveCaption.textContent = '选中关节后显示角度曲线';
    drawJointCurve(0);
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = '暂无可播放轨迹。从左侧资源库选择资源后，这里会显示每个关节的实时角度。';
    list.append(empty);
    return;
  }

  const names = fd.jointNames;
  if (!state.selectedJoint || !names.includes(state.selectedJoint)) state.selectedJoint = names[0];

  dom.jointsCount.textContent = `${names.length} 个关节`;

  for (let index = 0; index < names.length; index += 1) {
    const name = names[index];
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'joint-row';
    row.dataset.joint = name;
    row.title = name;
    if (name === state.selectedJoint) row.classList.add('selected');

    const nameEl = document.createElement('span');
    nameEl.className = 'j-name';
    nameEl.textContent = name;
    row.append(nameEl);

    if (isJointMissing(name)) {
      const missing = document.createElement('span');
      missing.className = 'j-missing';
      missing.textContent = '模型缺失';
      row.append(missing);
    }

    const bar = document.createElement('span');
    bar.className = 'j-bar';
    const barFill = document.createElement('i');
    bar.append(barFill);
    row.append(bar);

    const valueEl = document.createElement('span');
    valueEl.className = 'j-value';
    valueEl.textContent = '—';
    row.append(valueEl);

    list.append(row);
    state.jointRows.set(name, { row, value: valueEl, barFill });
  }

  updateJointReadout(currentFrameIndex());
  drawJointCurve(currentFrameIndex());
}

function updateJointReadout(frame) {
  const fd = state.trackData;
  const rows = jointTrackRow();
  if (!fd || !rows) return;
  const safeFrame = clamp(frame, 0, Math.max(0, play.frameCount - 1));
  const angles = rows[safeFrame] || [];
  const names = fd.jointNames;

  for (let index = 0; index < names.length; index += 1) {
    const name = names[index];
    const entry = state.jointRows.get(name);
    if (!entry) continue;
    const value = Number(angles[index]);
    if (!Number.isFinite(value)) {
      entry.value.textContent = '—';
      continue;
    }
    entry.value.textContent = `${(value * DEG).toFixed(1)}°`;
    const percent = clamp(50 + (value / Math.PI) * 50, 0, 100);
    if (entry.barFill) entry.barFill.style.left = `${percent.toFixed(1)}%`;
  }
  dom.jointsFrame.textContent = `帧 ${safeFrame + 1} / ${play.frameCount}`;
  if (state.selectedJoint) {
    const index = names.indexOf(state.selectedJoint);
    const value = Number(angles[index]);
    dom.curveCaption.innerHTML = `<span class="cname">${escapeHtml(state.selectedJoint)}</span><span>${Number.isFinite(value) ? `${(value * DEG).toFixed(2)}° / ${value.toFixed(4)} rad` : '—'}</span>`;
  }
}

function drawJointCurve(frame) {
  const canvas = dom.jointCurve;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cssWidth = canvas.clientWidth || 340;
  const cssHeight = canvas.clientHeight || 128;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(cssWidth * dpr) || canvas.height !== Math.round(cssHeight * dpr)) {
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = 'rgba(5, 11, 18, 0.72)';
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const fd = state.trackData;
  const selected = state.selectedJoint;
  const padding = { left: 34, right: 8, top: 10, bottom: 17 };
  const plotW = Math.max(20, cssWidth - padding.left - padding.right);
  const plotH = Math.max(20, cssHeight - padding.top - padding.bottom);

  if (!fd || !selected) {
    ctx.fillStyle = '#5b6e86';
    ctx.font = '11px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('选择资源并点击关节后显示角度曲线', cssWidth / 2, cssHeight / 2);
    return;
  }

  const nameIndex = fd.jointNames.indexOf(selected);
  const rows = jointTrackRow();
  if (nameIndex < 0 || !rows) return;
  const series = new Float32Array(rows.length);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < rows.length; i += 1) {
    const value = Number(rows[i]?.[nameIndex]);
    const safe = Number.isFinite(value) ? value : 0;
    series[i] = safe;
    if (safe < min) min = safe;
    if (safe > max) max = safe;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return;
  if (max - min < 0.15) {
    const mid = (max + min) / 2;
    min = mid - 0.1;
    max = mid + 0.1;
  }
  const padY = (max - min) * 0.12;
  min -= padY;
  max += padY;
  const toY = (value) => padding.top + plotH - ((value - min) / (max - min)) * plotH;
  const toX = (index) => padding.left + (rows.length <= 1 ? 0 : (index / (rows.length - 1)) * plotW);

  // 网格与刻度
  ctx.font = '9.5px "Consolas", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const gridCount = 4;
  for (let i = 0; i <= gridCount; i += 1) {
    const value = min + ((max - min) * i) / gridCount;
    const y = toY(value);
    ctx.strokeStyle = i === 0 || i === gridCount ? 'rgba(38, 58, 82, 0.9)' : 'rgba(26, 40, 57, 0.75)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#4f647c';
    ctx.fillText(`${(value * DEG).toFixed(0)}°`, padding.left - 5, y);
  }

  // 0 度参考线
  if (min < 0 && max > 0) {
    const zeroY = toY(0);
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(125, 211, 252, 0.35)';
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(padding.left + plotW, zeroY);
    ctx.stroke();
    ctx.restore();
  }

  // 曲线
  ctx.save();
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 1.6;
  ctx.shadowColor = 'rgba(34, 211, 238, 0.55)';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  for (let i = 0; i < rows.length; i += 1) {
    const x = toX(i);
    const y = toY(series[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();

  // 当前帧游标
  const safeFrame = clamp(Math.round(frame), 0, rows.length - 1);
  const cursorX = toX(safeFrame);
  ctx.strokeStyle = 'rgba(167, 139, 250, 0.75)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cursorX, padding.top);
  ctx.lineTo(cursorX, padding.top + plotH);
  ctx.stroke();
  ctx.fillStyle = '#c4b5fd';
  ctx.beginPath();
  ctx.arc(cursorX, toY(series[safeFrame]), 3.2, 0, Math.PI * 2);
  ctx.fill();

  // x 轴说明
  ctx.fillStyle = '#4f647c';
  ctx.font = '9.5px "Consolas", monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('0s', padding.left, cssHeight - 4);
  ctx.textAlign = 'right';
  const durationText = `${fmtTime(fd.duration)}s`;
  ctx.fillText(durationText, padding.left + plotW, cssHeight - 4);
}

// ===========================================================================
// 顶栏 / 事件绑定
// ===========================================================================

async function retryConnection() {
  if (SELFTEST) {
    toast('自检模式使用内置 mock 数据，无需连接后端', 'info');
    return;
  }
  setServerStatus('connecting', '连接中…');
  hideOffline();
  state.online = false;
  await initBackend();
}

function switchTab(name) {
  state.activeTab = name;
  for (const btn of dom.tabs.querySelectorAll('.tab')) {
    btn.classList.toggle('active', btn.dataset.tab === name);
  }
  $('tab-meta').hidden = name !== 'meta';
  $('tab-joints').hidden = name !== 'joints';
  if (name === 'joints') {
    renderJointPanel();
    updateJointReadout(currentFrameIndex());
    drawJointCurve(currentFrameIndex());
  }
}

function bindEvents() {
  dom.modelSelect.addEventListener('change', async () => {
    const robot = state.robots.find((item) => item.id === dom.modelSelect.value);
    if (!robot) {
      dom.modelSelect.value = state.modelId || '';
      return;
    }
    if (robot.id === state.modelId) return;
    const ok = await loadModel(robot);
    if (ok) {
      rebuildJointMap();
      rebuildGhost();
      if (state.trackData) applyFrame(currentFrameIndex());
      renderJointPanel();
      renderMetadata();
      toast(`已手动切换模型：${robot.name || robot.id}（下次加载资源仍会按 modelHint 自动选择）`, 'info', 3600);
    }
  });

  dom.btnRetry.addEventListener('click', retryConnection);
  dom.btnOfflineRetry.addEventListener('click', retryConnection);
  dom.btnRefresh.addEventListener('click', async () => {
    if (SELFTEST) {
      toast('自检模式：资源库为内置 mock 数据', 'info');
      await loadLibrary(state.root);
      return;
    }
    if (!state.online) {
      await retryConnection();
      return;
    }
    await loadLibrary(state.root);
  });

  dom.rootSelect.addEventListener('change', () => {
    loadLibrary(dom.rootSelect.value);
  });

  dom.searchInput.addEventListener('input', () => {
    state.search = dom.searchInput.value;
    renderLibrary();
  });

  dom.btnLoadPath.addEventListener('click', loadManualPath);
  dom.manualPathInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      loadManualPath();
    }
  });

  dom.entryList.addEventListener('click', (event) => {
    const button = event.target.closest('.entry');
    if (!button) return;
    const entry = state.entries.find((item) => item.path === button.dataset.path);
    if (entry) selectEntry(entry);
  });

  dom.ghostToggle.addEventListener('change', () => toggleGhost(dom.ghostToggle.checked));
  dom.btnResetView.addEventListener('click', resetView);
  dom.btnDismissError.addEventListener('click', hideViewError);

  dom.btnPlay.addEventListener('click', () => {
    if (!state.trackData) {
      toast('先选择一个可播放资源', 'warn');
      return;
    }
    setPlaying(!play.playing);
  });
  dom.btnReplay.addEventListener('click', () => {
    if (!state.trackData) return;
    play.time = 0;
    applyFrame(0);
    setPlaying(true);
  });

  dom.timeline.addEventListener('pointerdown', () => {
    if (!state.trackData) return;
    play.scrubbing = true;
    if (play.playing) setPlaying(false);
  });
  const endScrub = () => {
    if (!play.scrubbing) return;
    play.scrubbing = false;
    if (state.trackData) applyFrame(currentFrameIndex());
  };
  dom.timeline.addEventListener('pointerup', endScrub);
  dom.timeline.addEventListener('pointercancel', endScrub);
  dom.timeline.addEventListener('input', () => {
    if (!state.trackData) return;
    seekFrame(Number(dom.timeline.value));
  });
  dom.timeline.addEventListener('change', endScrub);

  dom.speedSelect.addEventListener('change', () => {
    play.speed = Number(dom.speedSelect.value) || 1;
    toast(`播放倍速：${play.speed}×`, 'info', 1400);
  });
  dom.loopToggle.addEventListener('change', () => {
    play.loop = dom.loopToggle.checked;
  });

  dom.tabs.addEventListener('click', (event) => {
    const tab = event.target.closest('.tab');
    if (tab) switchTab(tab.dataset.tab);
  });

  dom.jointList.addEventListener('click', (event) => {
    const row = event.target.closest('.joint-row');
    if (!row) return;
    state.selectedJoint = row.dataset.joint;
    for (const item of dom.jointList.querySelectorAll('.joint-row')) {
      item.classList.toggle('selected', item === row);
    }
    updateJointReadout(currentFrameIndex());
    drawJointCurve(currentFrameIndex());
  });

  // 点击后让 button 失焦，避免空格既触发 keydown 又触发 button click
  document.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (button) button.blur();
  });

  window.addEventListener('keydown', (event) => {
    const target = event.target;
    const tag = target && target.tagName ? target.tagName : '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (target && target.isContentEditable)) return;
    if (event.code === 'Space') {
      event.preventDefault();
      if (state.trackData) setPlaying(!play.playing);
    } else if (event.code === 'ArrowLeft') {
      event.preventDefault();
      if (!state.trackData) return;
      setPlaying(false);
      seekFrame(currentFrameIndex() - 1);
    } else if (event.code === 'ArrowRight') {
      event.preventDefault();
      if (!state.trackData) return;
      setPlaying(false);
      seekFrame(currentFrameIndex() + 1);
    } else if (event.code === 'KeyR') {
      event.preventDefault();
      resetView();
    }
  });

  window.addEventListener('error', (event) => {
    reportGlobalError(`页面错误：${event.message || '未知错误'}`);
  });
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    reportGlobalError(`未处理异常：${errorMessage(reason)}`);
  });
}

// ===========================================================================
// 初始化
// ===========================================================================

function markAppReady() {
  if (window.__APP_READY__) return;
  // ?path= 自动加载场景：等自动加载（inspect+tracks+模型）完成后再置位
  if (state.deferReady) return;
  window.__APP_READY__ = true;
}

function populateSelfTestShell() {
  setServerStatus('online', '自检模式');
  state.online = false;
  state.health = { ok: true, version: '0.1.0-selftest', roots: [{ path: MOCK_ROOT, label: `${MOCK_ROOT}（自检）` }] };
  state.roots = state.health.roots;
  state.root = '';
  populateRoots();
  state.robots = SELFTEST_ROBOTS.slice();
  populateModelSelect();
  state.entries = SELFTEST_ENTRIES.map(normalizeEntry);
  state.libraryLoaded = true;
  state.libraryError = '';
  renderLibrary();
}

async function initBackend() {
  try {
    const health = await api.health();
    state.online = true;
    state.health = health;
    state.roots = Array.isArray(health.roots) ? health.roots : [];
    state.health = { ...health, version: health.version || '?' };
    populateRoots();
    setServerStatus('online', `已连接 v${health.version || '?'}`);
    hideOffline();
  } catch (err) {
    state.online = false;
    setServerStatus('offline', '后端未连接');
    const msg = errorMessage(err);
    populateModelSelect();
    showOffline(`无法访问本地 ONNX 播放服务（默认 http://127.0.0.1:8766）。\n请先启动后端，然后点「重试连接」。\n错误：${msg}`);
    dom.libHint.className = 'lib-hint error';
    dom.libHint.textContent = '后端未连接，无法加载资源库。';
    markAppReady();
    return;
  }

  let robots = [];
  try {
    const data = await api.robots();
    robots = Array.isArray(data.robots) ? data.robots : [];
  } catch (err) {
    toast(`读取机器人模型失败：${errorMessage(err)}`, 'error', 4200);
  }
  state.robots = robots;
  populateModelSelect();
  if (!robots.length) toast('后端 /api/robots 没有返回可用模型', 'warn', 3800);

  // ?path= 自动加载时避免先白载一个默认模型（X2 约 89MB），等 tracks 的 modelHint 再载
  const modelPromise = robots.length && !AUTO_PATH ? loadModel(robots[0]) : Promise.resolve(false);
  const libraryPromise = loadLibrary('');
  await Promise.allSettled([modelPromise, libraryPromise]);
  markAppReady();
}

async function init() {
  try {
    initViewer();
    bindEvents();
    switchTab('meta');

    if (SELFTEST) {
      populateSelfTestShell();
      const preferred =
        state.entries.find((entry) => entry.playable && robotKey(entry.robot) === 'a3') ||
        state.entries.find((entry) => entry.playable);
      if (preferred) await selectEntry(preferred);
      state.deferReady = false;
      markAppReady();
    } else {
      await initBackend();
      // ?path=<绝对路径>：初始化完成后自动等价于点击资源库条目（inspect → tracks → 载模型 → 播放）
      if (AUTO_PATH) {
        let autoOk = false;
        try {
          autoOk = await loadResourceByPath(AUTO_PATH);
          if (!autoOk) {
            toast(`自动加载未生成轨迹：${AUTO_PATH}`, 'error', 5200);
            showViewError(`?path 自动加载失败：未生成播放轨迹。\n${AUTO_PATH}`);
          }
        } catch (err) {
          const msg = errorMessage(err);
          toast(`自动加载资源失败：${msg}`, 'error', 5200);
          showViewError(`?path 自动加载失败：${msg}\n${AUTO_PATH}`);
        }
        // 自动加载失败时兜底一个默认模型，避免 3D 视口空白
        if (!autoOk && !view.robot && !view.placeholder && state.robots.length) {
          await loadModel(state.robots[0]);
        }
        state.deferReady = false;
        markAppReady();
      }
    }
  } catch (err) {
    console.error('[ONNX 播放器] 初始化失败', err);
    reportGlobalError(`初始化失败：${errorMessage(err)}`);
    state.deferReady = false;
    markAppReady();
  } finally {
    // 兜底：无论后端/模型是否成功，初始化流程结束后都标记 ready
    setTimeout(markAppReady, 12000);
    // ?path= 自动加载若意外卡死，最多等 3 分钟也必须置位，避免 headless 集成一直等待
    setTimeout(() => {
      state.deferReady = false;
      markAppReady();
    }, 180000);
  }

  window.__APP = {
    THREE,
    selftest: SELFTEST,
    state,
    play,
    view,
    api,
    currentFrameIndex,
    applyFrame,
    seekFrame,
    setPlaying,
    resetView,
    loadModel,
    selectEntry,
    loadResourceByPath,
  };
}

init();
