# 打包成 Windows exe（PyInstaller）

把 `mjswan/player/` 的按需播放器打包成双击即用、**不需要目标电脑装 Python** 的 Windows x64 便携包。

## 选型

| 工具 | 结论 |
| --- | --- |
| **[PyInstaller](https://pyinstaller.org/)**（本项目采用，6.22.3） | Windows 上最成熟的 Python 打包器，自带/社区 hook 覆盖 numpy、onnxruntime；`--onedir` 启动快、易排错；一条命令出一套 exe+依赖目录，再 zip 分发即可 |
| [Nuitka](https://nuitka.net/) | 编译成原生二进制、启动更快、体积可能更小，但需要 MSVC/MinGW 编译工具链，首次编译慢很多；本项目的体积主要在 onnxruntime/mujoco/前端资源，Nuitka 优势不明显 |
| cx_Freeze / Briefcase | 也能用，但 Windows + onnxruntime 的现成 hook 与社区资料不如 PyInstaller 多 |

参考：[PyInstaller 官方文档](https://pyinstaller.org/en/stable/)、[Python → exe 工具对比（PyInstaller/Nuitka/BeeWare）](https://codegym.cc/groups/posts/python-script-to-exe)、[PyInstaller 打包记录（资源/DLL 处理）](https://chaofanjiadan.github.io/2025/06/08/Pyinstaller%E6%89%93%E5%8C%85%E8%AE%B0%E5%BD%95-%E6%89%93%E5%8C%85%E6%8C%87%E5%8D%97/)、[Nuitka 中文整理](https://cloud.tencent.com.cn/developer/article/2712560)。

## 构建

构建机需要 Windows x64 + Python 3.12（本项目用 `onnx-player/.packenv` 虚拟环境）：

```bat
cd onnx-player
python312\python.exe -m venv .packenv
.packenv\Scripts\python.exe -m pip install pyinstaller onnx onnxruntime numpy mujoco ^
  -i https://pypi.tuna.tsinghua.edu.cn/simple
.packenv\Scripts\python.exe -X utf8 packaging\build_exe.py
```

产物：

```text
dist_exe\AgiBotONNXPlayer\AgiBotONNXPlayer.exe     # 双击启动（含控制台日志）
dist_exe\AgiBotONNXPlayer\使用说明.txt
dist_exe\AgiBotONNXPlayer-win64.zip                # 分发用：整个文件夹压缩包（约 126 MB）
```

## 打包内容

- Python 3.12 运行时 + onnx / onnxruntime / numpy / mujoco（`--collect-all` 收集 DLL 与数据）
- mjswan 播放器前端（`player/app/assets`，约 20 MB）与两台机器人的 MJCF 模板
  （A3 5 MB、X2 35 MB）
- A3 MJCF/网格（用于 A3 的 CSV→参考轨迹 FK）
- 播放器前端、两台机器人 MJCF 模板与 A3 模型网格
- 公开 Release 默认**不打包动作资源**（可用 `INCLUDE_SAMPLES=1` 在本机构建时打入自己的示例）

## 首次启动

1. 解压 zip，双击 `AgiBotONNXPlayer.exe`；
2. 选择可写数据目录（按顺序尝试）：`ONNX_PLAYER_HOME` →
   `%LOCALAPPDATA%\AgiBotONNXPlayer` → `%APPDATA%\AgiBotONNXPlayer` → exe 同目录 `data\` →
   用户目录 → 临时目录（不保留缓存），并展开约 90 MB 内置资源；
3. 自动生成 `config.json`（如构建时打入示例资源则加入；其他资源目录自行编辑 roots）；
4. 启动本地服务并打开浏览器「打开文件」页 <http://127.0.0.1:8767/library>。

> 打开单个 ONNX 仍按需准备（A3 约 1~3 秒、X2 约 0.2~1 秒，之后缓存秒开），不会重建整站。

## 已验证

- 在**干净数据目录**下以打包 exe 启动：资源展开、配置生成、资源库页面正常；
- 通过 exe 打开内置 A3 示例，1 秒完成准备并在浏览器中渲染出闭环跟踪画面；
- 默认数据目录选择在“受限环境”下会自动回退到 exe 同目录 `data\` 并正常启动。

## 限制

- 仅 Windows x64；包未做代码签名，SmartScreen 可能提示“未知发布者”（选择“仍要运行”）。
- 体积较大（zip 126 MB / 解压 265 MB），主要来自 onnxruntime、MuJoCo 与浏览器端 WASM/模型资源。
- 目标电脑无需 Python，但第一次启动会复制约 90 MB 资源到可写目录。
