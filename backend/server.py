# -*- coding: utf-8 -*-
"""ONNX 播放器本地 HTTP 服务。

用法：
    python -m backend.server [--port 8766] [--no-open] [--root PATH]
仅监听 127.0.0.1，供本机浏览器使用。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import library, motion_a3, motion_x2, onnx_meta as om, robots
from .paths import ASSETS_DIR, MODELS_DIR, PROJECT_DIR, VERSION, WEB_DIR, set_stdout_utf8

mimetypes.add_type("model/stl", ".stl")
mimetypes.add_type("model/vnd.collada+xml", ".dae")
mimetypes.add_type("application/vnd.ros.urdf", ".urdf")
mimetypes.add_type("text/xml", ".xml")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("image/svg+xml", ".svg")

MAX_BODY = 64 * 1024

INDEX_HTML = os.path.join(WEB_DIR, "index.html")


class Handler(BaseHTTPRequestHandler):
    server_version = "OnnxPlayer/" + VERSION
    protocol_version = "HTTP/1.1"

    # ---------------- 基础工具 ----------------
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send_bytes(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8",
                    extra: Optional[Dict[str, str]] = None) -> None:
        encoding = None
        if len(body) > 262144 and "gzip" in (self.headers.get("Accept-Encoding") or "").lower():
            try:
                import gzip
                body = gzip.compress(body, compresslevel=3)
                encoding = "gzip"
            except Exception:  # noqa: BLE001
                encoding = None
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if encoding:
            self.send_header("Content-Encoding", encoding)
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._send_json({"ok": False, "error": "文件不存在"}, 404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in (
            "application/json", "application/javascript", "image/svg+xml",
            "model/vnd.collada+xml", "application/vnd.ros.urdf", "text/xml",
        ):
            ctype += "; charset=utf-8"
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError as exc:
            self._send_json({"ok": False, "error": f"读取失败：{exc}"}, 500)
            return
        self._send_bytes(200, body, ctype)

    def _safe_join(self, base: str, url_path: str) -> Optional[str]:
        rel = urllib.parse.unquote(url_path).replace("/", os.sep).lstrip("\\/")
        if ".." in rel.split(os.sep):
            return None
        full = os.path.abspath(os.path.join(base, rel))
        base_abs = os.path.abspath(base)
        if not (full == base_abs or full.startswith(base_abs + os.sep)):
            return None
        return full

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("请求体为空或过大")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _query(self) -> Dict[str, str]:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        return {k: v[-1] for k, v in qs.items()}, parsed.path

    # ---------------- 路由 ----------------
    def do_GET(self) -> None:  # noqa: N802
        query, path = self._query()
        try:
            if path in ("/", "/index.html"):
                self._send_file(INDEX_HTML)
                return
            if path.startswith("/web/"):
                full = self._safe_join(WEB_DIR, path[len("/web/"):])
                self._send_file(full) if full else self._send_json({"ok": False, "error": "非法路径"}, 400)
                return
            if path.startswith("/models/"):
                full = self._safe_join(MODELS_DIR, path[len("/models/"):])
                self._send_file(full) if full else self._send_json({"ok": False, "error": "非法路径"}, 400)
                return
            if path.startswith("/assets/"):
                full = self._safe_join(ASSETS_DIR, path[len("/assets/"):])
                self._send_file(full) if full else self._send_json({"ok": False, "error": "非法路径"}, 400)
                return
            if path == "/api/health":
                self._send_json({
                    "ok": True,
                    "version": VERSION,
                    "projectDir": PROJECT_DIR,
                    "roots": library.get_roots(),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return
            if path == "/api/robots":
                self._send_json({"ok": True, "robots": robots.get_robots()})
                return
            if path == "/api/library":
                root = query.get("root")
                force = query.get("force") in ("1", "true", "yes")
                self._send_json(library.scan_library(root=root, force=force))
                return
            if path == "/api/inspect":
                p = query.get("path") or ""
                if not p:
                    self._send_json({"ok": False, "error": "缺少 path 参数"}, 400)
                    return
                self._send_json(om.inspect(p))
                return
            if path == "/api/tracks":
                self._send_json({"ok": False, "error": "请使用 POST"}, 405)
                return
            if path == "/favicon.ico":
                self._send_bytes(204, b"", "image/x-icon")
                return
            self._send_json({"ok": False, "error": f"未知接口：{path}"}, 404)
        except om.MetaError as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"服务器内部错误：{exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        query, path = self._query()
        try:
            if path == "/api/tracks":
                body = self._read_json()
                result = build_tracks(
                    body.get("path", ""),
                    include_policy=bool(body.get("includePolicy", True)),
                )
                self._send_json(result)
                return
            if path == "/api/refresh":
                body = self._read_json()
                root = body.get("root")
                self._send_json(library.scan_library(root=root, force=True))
                return
            self._send_json({"ok": False, "error": f"未知接口：{path}"}, 404)
        except om.MetaError as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": f"请求解析失败：{exc}"}, 400)
        except (motion_x2.UnsupportedResource, motion_a3.UnsupportedResource) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 200)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"服务器内部错误：{exc}"}, 500)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()


def build_tracks(path: str, include_policy: bool = True) -> Dict[str, Any]:
    if not path:
        return {"ok": False, "error": "缺少 path 参数"}
    meta = om.read_meta(path)
    if not meta.playable:
        return {
            "ok": False,
            "error": meta.note or "该 ONNX 资源暂不支持播放",
            "robot": meta.robot,
            "kind": meta.kind,
        }
    if meta.robot == "x2":
        result = motion_x2.build_x2_tracks(path, meta, include_policy=include_policy)
    elif meta.robot == "a3":
        result = motion_a3.build_a3_tracks(path, meta, include_policy=include_policy)
    else:
        return {"ok": False, "error": "未知机器人类型，无法生成轨迹"}
    result.update({
        "ok": True,
        "path": meta.path,
        "name": om.guess_display_name(meta.path),
        "robot": meta.robot,
        "kind": meta.kind,
        "modelHint": robots.model_for_robot(meta.robot),
        "size": meta.size,
        "summary": meta.summary(),
    })
    result["warnings"] = list(dict.fromkeys(list(meta.warnings) + list(result.get("warnings", []))))
    return result


def create_server(host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    for candidate in range(port, port + 30):
        try:
            httpd = ThreadingHTTPServer((host, candidate), Handler)
            httpd.daemon_threads = True
            return httpd
        except OSError:
            continue
    raise RuntimeError(f"端口 {port}~{port + 29} 都被占用")


def main(argv: Optional[list] = None) -> int:
    set_stdout_utf8()
    parser = argparse.ArgumentParser(description="ONNX 播放器")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--no-scan", action="store_true", help="启动时不预热资源库扫描")
    args = parser.parse_args(argv)

    httpd = create_server(args.host, args.port)
    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    print("=" * 64)
    print("  ONNX 播放器 · 灵犀 X2 / 远征 A3")
    print(f"  服务地址: {url}")
    print(f"  模型目录: {MODELS_DIR}")
    print("  按 Ctrl+C 停止")
    print("=" * 64)

    if not args.no_scan:
        threading.Thread(target=_warmup, daemon=True).start()
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止……")
    finally:
        httpd.server_close()
    return 0


def _warmup() -> None:
    try:
        t0 = time.time()
        data = library.scan_library()
        playable = sum(1 for e in data["entries"] if e.get("playable"))
        print(f"资源库预热完成：{len(data['entries'])} 个 ONNX（可播放 {playable}），"
              f"耗时 {time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"资源库预热失败（不影响使用）：{exc}")


if __name__ == "__main__":
    sys.exit(main())
