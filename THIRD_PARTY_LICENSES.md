# 第三方组件与许可

本仓库自有代码以 MIT 许可发布（见 `LICENSE`）。以下第三方组件按其各自许可使用：

| 组件 | 用途 | 许可 |
| --- | --- | --- |
| [mjswan](https://github.com/ttktjmt/mjswan) | 浏览器 MuJoCo WASM + ONNX Runtime Web 播放前端、TrackingCommand、静态站构建 | Apache-2.0 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 物理引擎（浏览器 WASM / 服务端 FK） | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ONNX 推理（浏览器 WASM / Python） | MIT |
| [onnx](https://github.com/onnx/onnx) | ONNX 模型解析与 metadata | Apache-2.0 |
| [three.js](https://github.com/mrdoob/three.js) | 旧播放器渲染 | MIT |
| [urdf-loader](https://github.com/gkjohnson/urdf-loaders) | 旧播放器 URDF 加载 | Apache-2.0 |
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | A3 tracking 观测构造思路参考 | Apache-2.0 |
| [PyInstaller](https://pyinstaller.org/) | Windows exe 打包 | GPL-2.0-or-later（附例外条款） |
| 智元官方 SDK / 模型 | X2 RL Deploy 契约、X2/A3 机器人模型与 MJCF | 见智元官方 SDK 与模型目录中的许可文件 |

## 不包含的内容

- 仓库不包含 X2/A3 机器人网格模型与 MJCF、mjswan 前端构建产物（体积大，且来自官方模型资源）。
- 仓库与 Release 不包含任何动作 ONNX 资源或用户数据；请使用你自己的合法资源。
- Release 中的便携 exe 内置了官方机器人模型/MJCF，其版权归智元或原提供方所有，仅用于播放器渲染。
