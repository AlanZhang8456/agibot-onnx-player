# mjswan 迁移版 ONNX 播放器

本目录是把 `onnx-player` 从旧的自研 three.js/URDF + Python 仿真栈迁移到
[mjswan](https://github.com/ttktjmt/mjswan)（Apache-2.0）的成果：

- 渲染 / 物理：mjswan 内置（MuJoCo WASM + three.js）
- 推理：mjswan 内置 ONNX Runtime Web
- 参考动作：mjlab 格式 body-world motion clip，由 mjswan 原生 `TrackingCommand` 驱动
- 观测/动作：mjswan 的原生动作项 + 自定义 TS ObservationBinding（只写策略契约胶水）

已验证资源（`resources.json`）：灵犀 X2「特技连踢 / 劈叉落地」、远征 A3 T2.5「太极4-v03 /
金刚倒坠慢」。浏览器中策略闭环跟踪参考动作，不摔倒；截图见 `../docs/screenshots/mjswan-*.png`。

## 日常使用：按需打开一个文件（start.bat，推荐）

双击 `../start.bat` → 浏览器打开**打开文件页** <http://127.0.0.1:8767/library>：

- 「最近打开」：打开过的资源，点开即播（缓存，约 0.03 秒）。
- 「选择 ONNX 文件…」：Windows 原生文件框，选中后**只准备这一个文件**：
  A3 首次约 1~3 秒（CSV→参考轨迹 npz）、X2 首次约 0.2~1 秒（参考分支提取 + time_step
  循环包装 ONNX），之后直接复用缓存。
- 「资源库」：列出 `../config.json` 中识别到的可播放资源（当前 171 个），点击即播；
  也支持手动路径输入，或 `/open?path=<URL 编码绝对路径>`。
- 已打开的多个资源会进入同一个 mjswan manifest，可在播放器右上角 Scene 下拉切换。

实现要点（`player/`）：mjswan 前端、`plugins.js` 与机器人 MJCF（`templates/*_scene.mjz`）是固定
模板；打开文件时只增量准备该资源的 `policy.onnx` / `motion.npz`，追加到 `app/manifest.json`。
机器人模型通过 `../../templates/<kind>_scene.mjz` 共享，不会为每个资源复制 36MB 的 X2 模型；
因此**打开新文件不需要重建整站**。播放器运行时只需 `onnx-player/.pylibs`（onnx/onnxruntime/
numpy/mujoco），不需要 mjswan/torch/Node。

预构建静态站（固定资源，适合分享/离线）仍可用：`../start-static.bat`（8769）+
`../rebuild_mjswan.bat`（仅重建 `site/`）。

## 目录

```
mjswan/
├─ player/            # 按需播放器（start.bat）：单文件准备 + 动态 manifest，不需要重建整站
│  ├─ server.py       #   HTTP 服务：打开文件页 / 资源库 / 原生文件框 / 动态 manifest
│  ├─ dynamic.py      #   单资源准备：A3 CSV→npz、X2 参考提取+循环 ONNX、registry/manifest
│  ├─ templates/      #   固定的机器人 scene.mjz + 场景 JSON 模板
│  ├─ app/            #   mjswan 前端静态文件 + 已准备资源（policy/motion）
│  └─ runtime/        #   registry.json（已打开资源清单）与扫描索引（可删）
├─ build_site.py      # 构建预构建静态站：资源清单 -> 资产准备 -> mjswan Builder -> site/
├─ prepare_x2.py      # X2 ONNX -> 参考动作 npz + time_step 循环包装 ONNX + MJCF
├─ prepare_a3.py      # A3 ONNX + CSV -> 参考动作 npz
├─ resources.json     # 静态站资源清单（kind/name/onnx/csv）
├─ obs/AgibotA3Obs.ts # A3 157/137 维观测（TS，mjswan 自定义观测扩展点）
├─ obs/LingxiX2Obs.ts # X2 151/131 维观测
├─ mjswan_shim.py     # 静态站构建胶水（无 mjlab / Windows esbuild / 原生命令绑定）
├─ site/              # 预构建静态站产物（start-static.bat 托管）
└─ cache/             # 静态站构建的中间资产与资源库扫描缓存（可删）
```

## 运行

```bat
start.bat            :: 按需播放器（打开文件页 http://127.0.0.1:8767/library）
start-static.bat     :: 预构建静态站（http://127.0.0.1:8769/，固定 resources.json 资源）
start-old.bat        :: 旧播放器（过渡回退，http://127.0.0.1:8766/）
rebuild_mjswan.bat   :: 重建预构建静态站 site/（日常打开文件不需要）
rebuild_mjswan.bat --scan --limit 6
                     :: 扫描 config.json 资源库，每种机器人取 6 个可播放资源构建静态站
```

> 静态站 `--scan` 读取 `../config.json` 的 roots，用 ONNX 元数据识别可播放资源（X2: obs 151/131 +
> time_step 双输入；A3: obs 157/137 + 配套 CSV），结果缓存在 `cache/scan_index.json`。
> 按需播放器（start.bat）直接复用这份扫描索引作为资源库列表。

静态站只依赖 Python 标准库（`http.server`）即可托管；构建才需要 mjswan/torch。

### 构建环境

mjswan 0.13.1，Windows 可用。首次：

```bat
python -m venv mjswan\.venv
mjswan\.venv\Scripts\python.exe -m pip install mjswan onnxruntime ^
  -i https://pypi.tuna.tsinghua.edu.cn/simple
mjswan\.venv\Scripts\python.exe -m pip install torch==2.9.1 ^
  --index-url https://download.pytorch.org/whl/cpu
rebuild_mjswan.bat
```

> 构建只需要 CPU 版 torch（mjswan 内部 import 用），**不需要 mjlab/warp/CUDA**。
> 自定义 TS 观测需要 Node/esbuild：首次构建 mjswan 会自动装 Node 24（约 30 MB）并
> `npm install`；若 npm 缓存写入 `%LocalAppData%\npm-cache` 报 EPERM，设置
> `npm_config_cache` 到工作区即可（`rebuild_mjswan.bat` 已处理）。

## 资源契约（与旧播放器一致）

- **X2（obs 151 / 25-DOF 时 131，双输入）**：`obs[1,151] + time_step[1,1]` →
  `actions[1,29]`；参考分支只由 `time_step` 驱动，`time_step_total`/`motion_phase` 决定循环范围。
  观测布局：`command_dof_pos(29)+command_dof_vel(29)+projected_gravity(3)+base_ang_vel(3)`
  `+dof_pos_rel(29)+dof_vel*0.05(29)+last_action(29)`。
  `prepare_x2.py` 用 ONNX 参考分支提取 body-world 参考动作，并把 `time_step` 包装成
  `start + (t mod total)` 的循环版 ONNX，避免浏览器里无限增长的 time_step。
- **A3（obs 157 / 25-DOF 时 137，单输入）**：参考动作来自配套 CSV
  （root xyz + quat xyzw + 29 关节角，50Hz）；观测布局：
  `command(58)+projected_gravity(3)+base_ang_vel(3)+joint_pos_rel(29)+joint_vel*0.05(29)`
  `+last_action(29)+motion_anchor_ori_b(6)`。

动作统一为 `target = raw_action * action_scale + default_joint_pos`，PD 增益取 ONNX metadata：
A3 用 `joint_damping * 0.5`（与旧 sim 对齐），X2 用原值。

## 已知近似（PoC 验证时确定）

1. **X2 的重力/角速度用根（pelvis）IMU 而非实机的 torso IMU**：MuJoCo 的 torso_link 与实机
   IMU 安装系有偏差；用 torso 时闭环会摔（`poc_x2/sim_variants.py` 扫描：root 不摔、torso 摔），
   故沿用 root。真机部署仍以官方 torso IMU 为准。
2. **MuJoCo free joint 的 `qvel[3:6]` 角速度在 body 系**，观测里不能再做「世界→体系」旋转。
3. mjswan 构建器对「无 traced term 的纯声明式 MDP」仍要求 trace env / torch / mjlab 命令 reset 图，
   `mjswan_shim.py` 用占位 trace env + 原生 TrackingCommand 绑定绕过；正式长期方案可给 mjswan
   提上游 PR（无 traced term 时允许无 env）。
4. A3 需要配套 CSV；没有 CSV 的 A3 ONNX 无法重建参考动作，暂不进入站点（旧播放器同样如此）。
5. **重复 Scene 名称会导致 mjswan 整页黑屏**：mjswan 的 Scene 下拉框（Mantine Select）要求
   `name` 唯一，同一资源从两个不同路径打开会生成两个同名 scene。`dynamic.py` 现在准备资源与
   生成 manifest 时都会自动给重复名加 `[哈希后缀]`，并移除了会传空 `?scene=` 的入口链接；
   如果浏览器里还是旧的黑屏页，刷新（Ctrl+F5）即可。


## 打包成 Windows exe

已用 **PyInstaller** 打 Windows x64 便携包（目标电脑无需 Python）：
`dist_exe\AgiBotONNXPlayer-win64.zip`，解压后双击 `AgiBotONNXPlayer.exe` 即可。
选型、构建命令（`packaging/build_exe.py`）、数据目录回退与验证见
[`../packaging/README.md`](../packaging/README.md)。
## 许可

- mjswan：Apache-2.0（`site/THIRD_PARTY_LICENSES.md` 内含其依赖清单）。
- A3 / X2 模型、动作资源：来自智元官方 SDK/示例，版权归智元；随站点保留其许可证文件。
- 本目录 glue code（prepare_*.py、obs/*.ts、build_site.py、mjswan_shim.py）：遵循本仓库
  既有许可，并保留 mjswan Apache-2.0 出处。
