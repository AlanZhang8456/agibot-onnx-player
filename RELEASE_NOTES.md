# v0.1.0 · Windows x64 便携包

首个公开版本：AgiBot 灵犀 X2 / 远征 A3 动作 ONNX 浏览器播放器（mjswan 架构）。

## 下载
- 附件 `AgiBotONNXPlayer-win64.zip`：解压后双击 `AgiBotONNXPlayer.exe`，无需安装 Python。

## 功能
- 打开任意 X2 动作 ONNX（obs 151/131 + time_step 双输入）或 A3 动作 ONNX（obs 157/137，需配套 CSV），
  浏览器内 MuJoCo 物理闭环跟踪播放。
- 单文件按需准备：A3 首次约 1~3 秒、X2 约 0.2~1 秒，之后缓存秒开；不重建整站。
- 打开文件页支持最近打开、Windows 文件选择框、资源库扫描。
- 内置官方 X2/A3 机器人模型与 mjswan 播放前端；**不附带动作资源**。

## 说明
- 仅 Windows x64；未签名，SmartScreen 提示时选择“仍要运行”。
- 首次启动把播放器资源展开到 `%LOCALAPPDATA%\AgiBotONNXPlayer`（不可写时回退 exe 同目录 `data\`）。
- 使用你自己的合法动作资源；X2/A3 模型与动作资源版权归智元或原提供方。
