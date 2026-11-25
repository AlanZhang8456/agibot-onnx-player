# -*- coding: utf-8 -*-
"""动态 mjswan 播放器服务器：打开一个 ONNX 只准备这一个文件，不再整站重建。

路由：
  /                      mjswan 播放器（静态前端，动态 manifest）
  /manifest.json         当前已准备资源的 manifest（由 dynamic.py 维护）
  /library               资源库 / 打开文件页（资源列表、手动路径、选择文件、重扫）
  /open?path=<绝对路径>   准备该资源并跳转到 /?scene=<id>
  /choose                打开 Windows 原生文件选择框
  /rescan                重新扫描 config.json 资源目录
  /assets/*, /agibot/*    前端与已准备资源文件
"""
from __future__ import annotations

import html
import json
import mimetypes
import os
import sys
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dynamic  # noqa: E402

# PyInstaller 打包版：数据目录由 launcher 通过 ONNX_PLAYER_HOME 指向可写目录
APP = dynamic.APP
TEMPLATES = dynamic.TEMPLATES

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".mjz": "application/octet-stream",
    ".onnx": "application/octet-stream",
    ".npz": "application/octet-stream",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def _page(body: str, title: str = "ONNX 播放器") -> bytes:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
 body{{font:15px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#0f1720;color:#dbe4ee}}
 .wrap{{max-width:1180px;margin:0 auto;padding:26px 22px 60px}}
 h1{{font-size:24px;margin:0 0 4px}} .sub{{color:#8ba1b6;margin-bottom:18px}}
 h2{{font-size:17px;margin:26px 0 10px;color:#cfe0f0}}
 .card{{background:#17222d;border:1px solid #223244;border-radius:12px;padding:16px 18px;margin-bottom:16px}}
 .row{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
 input[type=text]{{flex:1;min-width:380px;background:#0f1720;border:1px solid #2b3f56;color:#e6eef7;border-radius:8px;padding:10px 12px;font:14px monospace}}
 button,.btn{{background:#1f6feb;color:#fff;border:0;border-radius:8px;padding:9px 14px;font-size:14px;cursor:pointer;text-decoration:none;display:inline-block;white-space:nowrap}}
 .btn.ghost{{background:#223244;color:#cfe0f0}} .btn:hover,button:hover{{filter:brightness(1.12)}}
 table{{width:100%;border-collapse:collapse;font-size:14px;table-layout:auto}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #223244;vertical-align:top}}
 th{{color:#8ba1b6;font-weight:600;position:sticky;top:0;background:#17222d;white-space:nowrap}}
 td:last-child{{width:1%;white-space:nowrap}}
 .src{{color:#7f95aa;font:12px monospace;word-break:break-all}}
 .tag{{display:inline-block;background:#223244;border-radius:6px;padding:1px 7px;font-size:12px;color:#9fc0e2}}
 .err{{background:#3a1d1d;border:1px solid #7a3030;color:#ffb4b4;padding:12px 14px;border-radius:10px;margin-bottom:14px;white-space:pre-wrap}}
 .ok{{background:#133528;border:1px solid #2f6b4f;color:#a9e8c6;padding:12px 14px;border-radius:10px;margin-bottom:14px}}
 .muted{{color:#7f95aa;font-size:13px}} .listwrap{{max-height:520px;overflow:auto;border-radius:10px}}
 .spin{{display:none;color:#9fc0e2;margin-left:8px}}
</style></head><body><div class="wrap">{body}</div></body></html>""".encode("utf-8")


def _library_page(error: str = "", message: str = "") -> bytes:
    prepared = dynamic.prepared_list()
    library = dynamic.load_library()
    rows_prepared = ""
    for item in prepared:
        rows_prepared.append if False else None
    parts = []
    for item in prepared:
        scene = urllib.parse.quote(item["scene"])
        name = html.escape(f"{dynamic.ROBOT_NAMES.get(item['kind'], item['kind'])} · {item['label']}")
        parts.append(
            f"<tr><td>{name}</td><td><code>{item['kind']}</code></td>"
            f"<td class=src>{html.escape(item['source'])}</td>"
            f"<td><a class=btn href='/?scene={scene}'>打开</a></td></tr>")
    rows_prepared = "".join(parts)
    if not rows_prepared:
        rows_prepared = "<tr><td colspan=4 class=muted>还没有打开过资源。用下面的「选择文件」或资源库列表开始。</td></tr>"

    lib_parts = []
    for item in library:
        q = urllib.parse.quote(item["source"], safe="")
        name = html.escape(item["name"])
        kind = item["kind"]
        lib_parts.append(
            f"<tr data-k='{kind}' data-n='{name.lower()}'><td>{name}</td><td><span class=tag>{kind}</span></td>"
            f"<td class=src>{html.escape(item['source'])}</td>"
            f"<td><a class=btn ghost href='/open?path={q}'>打开</a></td></tr>")
    rows_lib = "".join(lib_parts) or "<tr><td colspan=4 class=muted>资源库索引为空，点「重新扫描资源库」。</td></tr>"

    err = f"<div class=err>{html.escape(error)}</div>" if error else ""
    msg = f"<div class=ok>{html.escape(message)}</div>" if message else ""
    if prepared:
        player_href = "/?scene=" + urllib.parse.quote(prepared[0]["scene"])
        player_label = "打开最近资源"
    else:
        player_href = "/library"
        player_label = "先打开一个资源"
    body = f"""
<h1>ONNX 播放器 · 打开一个文件</h1>
<div class=sub>只准备当前文件（秒级缓存），不再重建整站；已打开过的资源可直接从「最近打开」进入。</div>
{err}{msg}
<div class=card>
  <h2 style="margin-top:0">最近打开 / 已准备</h2>
  <div class=listwrap><table><thead><tr><th>资源</th><th>类型</th><th>源文件</th><th></th></tr></thead>
  <tbody>{rows_prepared}</tbody></table></div>
</div>
<div class=card>
  <h2 style="margin-top:0">打开文件</h2>
  <div class=row>
    <a class=btn href="/choose">选择 ONNX 文件…</a>
    <span class=muted>或把文件绝对路径粘贴到这里：</span>
  </div>
  <form method="get" action="/open" style="margin-top:10px" class="row">
    <input type=text name=path placeholder="C:\\Users\\...\\policy.onnx" autocomplete=off>
    <button type=submit>打开</button>
  </form>
  <div class=muted style="margin-top:8px">A3 资源需要同目录下配套 CSV；支持 X2 151/131 维与 A3 157/137 维动作 ONNX。</div>
</div>
<div class=card>
  <h2 style="margin-top:0">资源库（{len(library)} 个可播放）</h2>
  <div class=row><input type=text id=q placeholder="按名字/路径筛选…" oninput="filt()">
  <a class="btn ghost" href="/rescan" onclick="this.textContent='扫描中…（可能要 1~2 分钟）'">重新扫描资源库</a></div>
  <div class=listwrap><table><thead><tr><th>资源</th><th>类型</th><th>路径</th><th></th></tr></thead>
  <tbody id=lib>{rows_lib}</tbody></table></div>
</div>
<div class=card>
  <h2 style="margin-top:0">其他</h2>
  <a class="btn ghost" href="{player_href}">{player_label}</a>
  <span class=muted>旧播放器：start-old.bat（8766）；修改资源清单/前端请见 mjswan/README.md。</span>
</div>
<script>
function filt(){{
  const q=document.getElementById('q').value.toLowerCase();
  document.querySelectorAll('#lib tr').forEach(tr=>{{
    const n=(tr.dataset.n||'')+' '+(tr.querySelector('.src')?.textContent||'').toLowerCase();
    tr.style.display = !q || n.includes(q) ? '' : 'none';
  }});
}}
</script>"""
    return _page(body, "ONNX 播放器 · 资源库")


class Handler(BaseHTTPRequestHandler):
    server_version = "AgiBotONNXPlayer/1.0"

    def log_message(self, fmt, *args):  # noqa: A003
        sys.stderr.write("[player] " + fmt % args + "\n")

    def _send(self, data: bytes, status: int = 200, ctype: str = "text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("text/html") else "public, max-age=3600")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _redirect(self, url: str):
        self.send_response(303)
        self.send_header("Location", url)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_app_file(self, path: Path):
        try:
            resolved = path.resolve()
            try:
                resolved.relative_to(APP.resolve())
            except ValueError:
                resolved.relative_to(TEMPLATES.resolve())
        except Exception:  # noqa: BLE001
            self._send(b"403", status=403, ctype="text/plain")
            return
        if not resolved.is_file():
            self._send(b"404", status=404, ctype="text/plain")
            return
        ctype = MIME.get(resolved.suffix.lower()) or mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        data = resolved.read_bytes()
        self._send(data, ctype=ctype)

    def do_GET(self):  # noqa: N802
        try:
            self._route()
        except Exception:  # noqa: BLE001
            detail = traceback.format_exc()
            self._send(_page(f"<h1>出错了</h1><div class=err>{html.escape(detail)}</div>"
                             f"<p><a class=btn href='/library'>返回资源库</a></p>"), status=500)

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def _route(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if route in ("/", "/index.html"):
            self._serve_app_file(APP / "index.html")
            return
        if route == "/manifest.json":
            dynamic.write_manifest()
            self._serve_app_file(APP / "manifest.json")
            return
        if route == "/library":
            self._send(_library_page(query.get("error", [""])[0], query.get("msg", [""])[0]))
            return
        if route == "/open":
            path = (query.get("path") or [""])[0].strip().strip('"')
            if not path:
                self._redirect("/library?error=" + urllib.parse.quote("没有提供文件路径"))
                return
            try:
                entry = dynamic.prepare(Path(path))
            except Exception as exc:  # noqa: BLE001
                self._redirect("/library?error=" + urllib.parse.quote(str(exc)))
                return
            scene = urllib.parse.quote(entry["scene"]["id"])
            self._redirect(f"/?scene={scene}")
            return
        if route == "/choose":
            chosen = dynamic.choose_file_dialog()
            if not chosen:
                self._redirect("/library")
                return
            self._redirect("/open?path=" + urllib.parse.quote(chosen, safe=""))
            return
        if route == "/rescan":
            try:
                count = dynamic.rescan()
                self._redirect("/library?msg=" + urllib.parse.quote(f"扫描完成：{count} 个可播放资源"))
            except Exception as exc:  # noqa: BLE001
                self._redirect("/library?error=" + urllib.parse.quote(f"扫描失败：{exc}"))
            return
        if route.startswith("/templates/"):
            self._serve_app_file(TEMPLATES / route[len("/templates/"):])
            return
        if route.startswith("/assets/") or route.startswith("/agibot/") or route in ("/logo.svg", "/robots.txt"):
            self._serve_app_file(APP / route.lstrip("/"))
            return
        self._send(b"404", status=404, ctype="text/plain")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    APP.mkdir(parents=True, exist_ok=True)
    dynamic.write_manifest()
    url = f"http://127.0.0.1:{args.port}/library"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("=" * 64)
    print(f"  ONNX Player (mjswan, on-demand) - http://127.0.0.1:{args.port}/")
    print(f"  Open a file: {url}")
    print("  Close this window or press Ctrl+C to stop.")
    print("=" * 64)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
