# -*- coding: utf-8 -*-
"""打包版启动器：把随 exe 分发的资源展开到可写数据目录，然后启动 mjswan 按需播放器。

双击 exe 后：
1. 首次运行把内置的 app/templates/models/samples（约 90MB）复制到
   %LOCALAPPDATA%\\AgiBotONNXPlayer；
2. 生成默认 config.json（如随包附带示例资源则自动加入；其他资源目录可自行编辑 roots）；
3. 启动本地播放器（默认 http://127.0.0.1:8767/library），自动打开浏览器。
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path

APP_NAME = "AgiBotONNXPlayer"


def resource_root() -> Path:
    """PyInstaller onedir 下数据在 _MEIPASS/bundle；开发模式下在 packaging/bundle。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "bundle"


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def choose_home() -> Path:
    """依次尝试可写的数据目录（受权限策略限制的电脑也能启动）。"""
    candidates: list[tuple[str, Path]] = []
    env = os.environ.get("ONNX_PLAYER_HOME")
    if env:
        candidates.append(("环境变量", Path(env).expanduser().resolve()))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(("LOCALAPPDATA", Path(local) / APP_NAME))
    roaming = os.environ.get("APPDATA")
    if roaming:
        candidates.append(("APPDATA", Path(roaming) / APP_NAME))
    candidates.append(("exe 同目录 data", app_dir() / "data"))
    candidates.append(("用户目录", Path.home() / APP_NAME))
    tmp = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        candidates.append(("临时目录（不保留缓存）", Path(tmp) / APP_NAME))
    for label, path in candidates:
        if _writable(path):
            if not env:
                print(f"数据目录：{path}（来源：{label}）", flush=True)
            return path
    raise SystemExit("找不到可写的数据目录；请把 exe 解压到用户目录，或设置 ONNX_PLAYER_HOME 环境变量。")


def default_home() -> Path:
    return choose_home()


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def ensure_user_data(home: Path, bundle: Path) -> None:
    version_file = bundle / "version.txt"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "dev"
    marker = home / ".bundle-version"
    current = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    need_copy = current != version or not (home / "app" / "assets").is_dir()
    if need_copy:
        print(f"首次启动 / 更新资源：正在展开到 {home} ...", flush=True)
        home.mkdir(parents=True, exist_ok=True)
        copy_tree(bundle / "player" / "app", home / "app")
        copy_tree(bundle / "player" / "templates", home / "templates")
        copy_tree(bundle / "models", home / "models")
        copy_tree(bundle / "samples", home / "samples")
        (home / "runtime").mkdir(parents=True, exist_ok=True)
        marker.write_text(version, encoding="utf-8")
        print("资源展开完成。", flush=True)


def ensure_config(home: Path) -> None:
    import json

    cfg_path = home / "config.json"
    if cfg_path.exists():
        return
    roots = []
    samples = home / "samples"
    if samples.is_dir():
        roots.append({"path": str(samples), "label": "内置示例（随 exe 分发）"})
    cfg_path.write_text(json.dumps({"port": 8767, "roots": roots}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"已生成资源库配置：{cfg_path}", flush=True)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    bundle = resource_root()
    home = default_home()
    ensure_user_data(home, bundle)
    ensure_config(home)

    os.environ["ONNX_PLAYER_HOME"] = str(home)
    os.environ["ONNX_PLAYER_ROOT"] = str(home)
    os.environ["PYTHONUTF8"] = "1"
    os.chdir(home)

    # 把 exe 的命令行参数透传给播放器（如 --no-open / --port 8771）
    forwarded = [a for a in sys.argv[1:]]
    sys.argv = [str(Path(sys.argv[0]).name), *forwarded]
    print("=" * 64)
    print(f"  AgiBot ONNX 播放器 · 数据目录：{home}")
    print("  打开文件页：http://127.0.0.1:8767/library")
    print("  关闭此窗口即停止服务")
    print("=" * 64)
    sys.path.insert(0, str(Path(getattr(sys, "_MEIPASS", home))))
    from player import server  # noqa: E402

    server.main()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        try:
            input("启动失败，按回车退出...")
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1)
