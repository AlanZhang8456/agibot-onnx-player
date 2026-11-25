// 用 Edge headless + CDP 做集成验证：
//  - 打开页面，等待 window.__APP_READY__
//  - 可选执行一段 JS（如自动加载资源），再等待自定义条件
//  - 输出页面状态 JSON + 截图 PNG + 控制台错误
// 用法: node tools/cdp_check.mjs <url> <out.png> [--wait "<js expr>"] [--timeout 60000] [--eval "<js>"]
// 依赖：Node >= 22（内置 WebSocket / fetch），无需 npm 包。

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const args = process.argv.slice(2);
const url = args[0] || 'http://127.0.0.1:8791/';
const outPng = args[1] || path.join(os.tmpdir(), 'cdp_shot.png');
let waitExpr = 'window.__APP_READY__ === true';
let timeout = 120000;
let preEval = null;
let afterEval = null;
let shots = null;
for (let i = 2; i < args.length; i++) {
  if (args[i] === '--wait') waitExpr = args[++i];
  else if (args[i] === '--timeout') timeout = parseInt(args[++i], 10);
  else if (args[i] === '--eval') preEval = args[++i];
  else if (args[i] === '--after') afterEval = args[++i];
  else if (args[i] === '--shots') shots = args[++i].split(',').map((v) => parseFloat(v));
}

const port = 9333 + Math.floor(Math.random() * 300);
const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'edge-cdp-'));
let proc = null;

function launchEdge() {
  proc = spawn(EDGE, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-extensions', '--hide-scrollbars', '--mute-audio',
    `--remote-debugging-port=${port}`, `--user-data-dir=${userData}`,
    '--window-size=1440,900', 'about:blank',
  ], { stdio: 'ignore' });
}
launchEdge();

let msgId = 0;
const pending = new Map();
const consoleMsgs = [];
const exceptions = [];
let ws = null;

function send(method, params = {}) {
  const id = ++msgId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    setTimeout(() => {
      if (pending.has(id)) { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }
    }, 30000);
  });
}

async function evaluate(expr, awaitPromise = true) {
  const r = await send('Runtime.evaluate', {
    expression: expr, awaitPromise, returnByValue: true, userGesture: true,
  });
  if (r.exceptionDetails) {
    throw new Error('JS 异常：' + JSON.stringify(r.exceptionDetails).slice(0, 800));
  }
  return r.result?.value;
}

async function getTargets() {
  for (let attempt = 0; attempt < 3; attempt++) {
    for (let i = 0; i < 60; i++) {
      try {
        const res = await fetch(`http://127.0.0.1:${port}/json/list`);
        const list = await res.json();
        const page = list.find((t) => t.type === 'page');
        if (page) return page;
      } catch { /* retry */ }
      await new Promise((r) => setTimeout(r, 250));
    }
    // 启动失败则重启一次 Edge（大量 Edge 实例时偶发端口未就绪）
    try { proc.kill(); } catch { /* ignore */ }
    await new Promise((r) => setTimeout(r, 800));
    launchEdge();
  }
  throw new Error('无法连接 Edge 调试端口');
}

async function main() {
  const page = await getTargets();
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = (e) => reject(new Error('WebSocket error'));
  });
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) p.reject(new Error(JSON.stringify(msg.error)));
      else p.resolve(msg.result);
    } else if (msg.method === 'Runtime.consoleAPICalled') {
      const text = (msg.params.args || []).map((a) => a.value ?? a.description ?? '').join(' ');
      consoleMsgs.push(`[${msg.params.type}] ${text}`);
    } else if (msg.method === 'Runtime.exceptionThrown') {
      exceptions.push(JSON.stringify(msg.params.exceptionDetails).slice(0, 1500));
    }
  };

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Log.enable');
  await send('Page.navigate', { url });
  await new Promise((r) => setTimeout(r, 800));

  if (preEval) {
    console.log('执行预置 JS:', preEval.slice(0, 200));
    console.log('结果:', JSON.stringify(await evaluate(preEval)).slice(0, 500));
  }

  const deadline = Date.now() + timeout;
  let lastErr = null;
  let ok = false;
  while (Date.now() < deadline) {
    try {
      ok = await evaluate(`Boolean(${waitExpr})`);
      if (ok) break;
    } catch (e) { lastErr = e; }
    await new Promise((r) => setTimeout(r, 500));
  }
  console.log(`等待条件 ${ok ? '满足' : '超时'}：${waitExpr}`);
  if (!ok && lastErr) console.log('最后一次等待错误:', String(lastErr).slice(0, 500));

  if (afterEval) {
    console.log('执行后置 JS:', afterEval.slice(0, 200));
    console.log('结果:', JSON.stringify(await evaluate(afterEval)).slice(0, 600));
  }

  // 稳定渲染几帧 / 或多时刻截图
  if (shots) {
    let prev = 0;
    for (const t of shots) {
      const waitMs = Math.max(0, (t - prev) * 1000);
      if (waitMs > 0) await new Promise((r) => setTimeout(r, waitMs));
      prev = t;
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      const p = outPng.replace(/\.png$/i, `_t${String(t).replace('.', 'p')}.png`);
      fs.writeFileSync(p, Buffer.from(shot.data, 'base64'));
      console.log(`截图 t=${t}s:`, p, fs.statSync(p).size, 'bytes');
    }
  }
  await new Promise((r) => setTimeout(r, 2500));

  const state = await evaluate(`(() => {
    const s = window.__APP?.state || {};
    const p = window.__APP?.play || {};
    return {
      title: document.title,
      ready: window.__APP_READY__ === true,
      hasCanvas: !!document.querySelector('canvas'),
      canvas: (() => { const c = document.querySelector('canvas'); return c ? [c.width, c.height] : null; })(),
      serverText: document.querySelector('#server-text')?.textContent?.trim(),
      playing: p.playing,
      time: p.time, fps: p.fps, frameCount: p.frameCount, duration: p.duration,
      trackCount: (s.trackData?.tracks || []).length,
      visibleTrack: (s.trackData?.tracks || []).findIndex((t) => t.defaultVisible !== false),
      jointNames: (s.trackData?.jointNames || []).length,
      robotId: s.robot?.id || s.modelId,
      loadState: s.loadingState || s.status,
      toasts: document.querySelector('#toast-stack')?.innerText?.slice(0, 400),
      viewError: document.querySelector('#view-error')?.textContent?.trim()?.slice(0, 300),
      offline: document.querySelector('#offline-overlay')?.style?.display,
    };
  })()`);
  console.log('页面状态:', JSON.stringify(state, null, 1).slice(0, 2500));

  // 截图
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(outPng, Buffer.from(shot.data, 'base64'));
  console.log('截图已保存:', outPng, fs.statSync(outPng).size, 'bytes');

  if (consoleMsgs.length) {
    console.log('控制台消息（前 20 条）:');
    consoleMsgs.slice(0, 20).forEach((m) => console.log('  ', m.slice(0, 300)));
  }
  if (exceptions.length) {
    console.log('页面异常:', exceptions.length);
    exceptions.slice(0, 5).forEach((e) => console.log('  ', e.slice(0, 600)));
  }
  return state.ready && state.hasCanvas ? 0 : 1;
}

main().then((code) => {
  try { ws?.close(); } catch { /* ignore */ }
  try { proc.kill(); } catch { /* ignore */ }
  setTimeout(() => process.exit(code), 300);
}).catch((e) => {
  console.error('CDP 检查失败:', e);
  try { ws?.close(); } catch { /* ignore */ }
  try { proc.kill(); } catch { /* ignore */ }
  process.exit(2);
});
