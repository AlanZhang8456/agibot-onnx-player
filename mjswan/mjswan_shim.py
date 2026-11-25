# -*- coding: utf-8 -*-
"""mjswan 构建的 Windows/无 mjlab shim。

mjswan 0.13.1 的声明式构建（自定义 TS 观测 + 原生动作/命令，无 traced term）本可不需要
mjlab，但构建器仍会：import torch、强制要求 trace env、在 Windows 上直接执行 POSIX 版
esbuild。这里集中处理这三件事，A3/X2/资源库构建脚本共用。
"""
from __future__ import annotations

import contextlib
import json
import subprocess
from types import SimpleNamespace

from mjswan.managers.command_manager import CommandBinding
from mjswan import register_command


def register_tracking_motion_command() -> None:
    """用原生 TrackingCommand 覆盖内置 MotionCommandCfg 绑定（去掉 mjlab reset jitter）。"""

    def _serialize(cfg) -> dict:
        viz = getattr(cfg, "viz", None)
        return {
            "anchor_body_name": cfg.anchor_body_name,
            "body_names": list(cfg.body_names),
            "sampling_mode": getattr(cfg, "sampling_mode", "start"),
            "pose_range": {k: list(v) for k, v in (getattr(cfg, "pose_range", {}) or {}).items()},
            "velocity_range": {k: list(v) for k, v in (getattr(cfg, "velocity_range", {}) or {}).items()},
            "joint_position_range": list(getattr(cfg, "joint_position_range", (0.0, 0.0))),
            "debug_vis": bool(getattr(cfg, "debug_vis", False)),
            "viz_mode": getattr(viz, "mode", "ghost"),
            "ghost_color": list(getattr(viz, "ghost_color", (0.5, 0.7, 0.5, 0.5))),
        }

    register_command(
        "MotionCommandCfg",
        CommandBinding(ts_name="TrackingCommand", serializer=_serialize),
    )


def patch_windows_esbuild() -> None:
    """Windows 下 `node_modules/.bin/esbuild` 是 POSIX 脚本，改用 node.exe 执行 JS 入口。"""
    import mjswan.build.frontend as _frontend

    def _build_plugins_module_win(self, dest) -> bool:
        terms = self._collect_custom_terms()
        if not terms:
            return False
        by_src: dict = {}
        for names in terms.values():
            for ts_name, src in names.items():
                by_src.setdefault(src, []).append(ts_name)
        lines = ["// Auto-generated plugin entry — do not edit."]
        for src, names in by_src.items():
            lines.append(f"import {{ {', '.join(sorted(set(names)))} }} from {json.dumps(str(src))};")
        for kind, names in terms.items():
            lines.append(f"export const {kind} = {{ {', '.join(sorted(names.keys()))} }};")
        entry = self.project_dir / "src" / ".plugins-entry.ts"
        entry.write_text("\n".join(lines) + "\n", encoding="utf-8")
        esbuild_js = self.project_dir / "node_modules" / "esbuild" / "bin" / "esbuild"
        if not esbuild_js.exists():
            self.create_env()
            self.install_dependencies()
        node = self._get_node_bin()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call([
                str(node), str(esbuild_js), str(entry),
                "--bundle", "--format=esm", "--platform=browser",
                f"--outfile={dest}", *self._plugin_alias_args(),
            ], cwd=self.project_dir)
        finally:
            entry.unlink(missing_ok=True)
        return True

    _frontend.ClientBuilder.build_plugins_module = _build_plugins_module_win


def make_trace_env_shim() -> SimpleNamespace:
    """占位 trace env：声明式 MDP（无 traced term）构建不会真的用它。"""
    return SimpleNamespace(
        num_envs=1, device="cpu", reward_manager=None,
        scene=SimpleNamespace(sensors={}),
    )


def null_mjlab_contexts() -> None:
    """空转 mjlab 的 action/command 上下文管理器（无 traced term 时不需要）。"""
    import mjswan.build.manifest as _manifest
    _manifest.policy_actions = lambda env, num_actions: contextlib.nullcontext()
    _manifest.mdp_commands = lambda env, widths: contextlib.nullcontext()
