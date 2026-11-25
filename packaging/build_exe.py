# -*- coding: utf-8 -*-
"""用 PyInstaller 把按需 mjswan 播放器打包成 Windows exe（onedir + zip）。

用法（构建机需要 onnx-player/.packenv，见 README）：
    .packenv\\Scripts\\python.exe packaging\\build_exe.py

产物：
    dist_exe\\AgiBotONNXPlayer\\AgiBotONNXPlayer.exe
    dist_exe\\AgiBotONNXPlayer-win64.zip
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # onnx-player/
PACKENV = ROOT / ".packenv"
DIST = ROOT / "dist_exe"
BUILD = ROOT / "build_exe"
BUNDLE = HERE / "bundle"                # 打包进 exe 的数据（不含用户已准备资源）
APP_OUT = DIST / "AgiBotONNXPlayer"
README_TXT = """AgiBot ONNX 动作播放器（灵犀 X2 / 远征 A3）

使用方法
--------
1. 解压后双击 AgiBotONNXPlayer.exe。
2. 浏览器会自动打开“打开文件”页（http://127.0.0.1:8767/library）；
   首次启动会把播放器前端/模型模板资源展开到：
   %LOCALAPPDATA%\\AgiBotONNXPlayer
3. 选择 ONNX 文件 / 粘贴绝对路径，即可播放你自己的动作资源。
   - X2：支持 obs 151/131 维 + time_step 双输入的动作 ONNX；
   - A3：支持 obs 157/137 维单输入动作 ONNX，需要同目录配套 CSV。
4. 要加入资源库，编辑 %LOCALAPPDATA%\\AgiBotONNXPlayer\\config.json 的 roots 后重启。

说明
----
- 关闭 exe 的控制台窗口即停止本地服务。
- 程序只在本机 127.0.0.1 上运行，不联网、不连接真机。
- 未做代码签名，Windows SmartScreen 可能提示“未知发布者”，选择“仍要运行”即可。
"""
VERSION_TXT = "AgiBotONNXPlayer bundle " + time.strftime("%Y-%m-%d %H:%M:%S")
# 公开 release 默认不打包用户的动作资源；本机自用可设置 INCLUDE_SAMPLES=1
INCLUDE_SAMPLES = os.environ.get("INCLUDE_SAMPLES", "1") not in ("0", "false", "no")


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def stage_bundle() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    player_app = ROOT / "mjswan" / "player" / "app"
    stage_app = BUNDLE / "player" / "app"
    stage_app.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "logo.svg", "robots.txt"):
        src = player_app / name
        if src.exists():
            shutil.copy2(src, stage_app / name)
    copy_tree(player_app / "assets", stage_app / "assets")
    copy_tree(player_app / "agibot" / "licenses", stage_app / "agibot" / "licenses")
    copy_tree(ROOT / "mjswan" / "player" / "templates", BUNDLE / "player" / "templates")
    copy_tree(ROOT / "models" / "a3" / "a3_t2d5", BUNDLE / "models" / "a3" / "a3_t2d5")
    if INCLUDE_SAMPLES:
        copy_tree(ROOT / "assets" / "samples", BUNDLE / "samples")
    (BUNDLE / "version.txt").write_text(VERSION_TXT, encoding="utf-8")


def build() -> None:
    pyinstaller = PACKENV / "Scripts" / "pyinstaller.exe"
    if not pyinstaller.exists():
        raise SystemExit(f"缺少 PyInstaller：{pyinstaller}（先创建 .packenv 并安装依赖）")
    if DIST.exists():
        shutil.rmtree(DIST)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    stage_bundle()
    cmd = [
        str(pyinstaller),
        "--noconfirm", "--clean",
        "--onedir", "--console",
        "--name", "AgiBotONNXPlayer",
        "--paths", str(ROOT / "mjswan"),
        "--paths", str(ROOT / "mjswan" / "player"),
        "--paths", str(HERE),
        "--hidden-import", "dynamic",
        "--hidden-import", "prepare_a3",
        "--hidden-import", "prepare_x2",
        "--add-data", f"{BUNDLE}{os.pathsep}bundle",
        "--collect-all", "onnxruntime",
        "--collect-all", "mujoco",
        "--collect-all", "onnx",
        "--hidden-import", "numpy",
        "--hidden-import", "onnxruntime",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        str(HERE / "launcher.py"),
    ]
    print("运行 PyInstaller ...", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    (APP_OUT / "使用说明.txt").write_text(README_TXT, encoding="utf-8")
    print("压缩发布包 ...", flush=True)
    archive = DIST / "AgiBotONNXPlayer-win64"
    shutil.make_archive(str(archive), "zip", str(DIST), "AgiBotONNXPlayer")
    exe_mb = sum(f.stat().st_size for f in (APP_OUT / "AgiBotONNXPlayer.exe").rglob("*") if f.is_file()) / 1e6
    total_mb = sum(f.stat().st_size for f in APP_OUT.rglob("*") if f.is_file()) / 1e6
    zip_mb = (archive.with_suffix(".zip")).stat().st_size / 1e6
    print(f"OK: {APP_OUT}")
    print(f"    exe 文件夹 {total_mb:.0f} MB，zip {zip_mb:.0f} MB")
    print(f"    {archive.with_suffix('.zip')}")


if __name__ == "__main__":
    build()
