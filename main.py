#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstrBot 本地 GPT-SoVITS 调音 TTS 插件
====================================
随 AstrBot 启动/停止本地 TTS 服务：
  1. GPT-SoVITS 推理引擎 (默认 9880)
  2. 调音包装服务 (默认 9881, 内置 /tts 与 /infer_single)

AstrBot WebUI 的「GSVI TTS(API)」语音合成源配置 api_base 指向包装服务地址即可。
"""
import asyncio
import os
import subprocess
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

PLUGIN_DIR = Path(__file__).resolve().parent

ENGINE_LOG = "/tmp/localtts_engine.log"
WRAPPER_LOG = "/tmp/localtts_wrapper.log"


class LocalTtsPlugin(Star):
    """随 AstrBot 启停本地 GPT-SoVITS 调音 TTS 服务。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._engine_proc: subprocess.Popen | None = None
        self._wrapper_proc: subprocess.Popen | None = None
        self._proc_lock = asyncio.Lock()

    # ---------------- 路径/配置辅助 ----------------
    def _svc(self) -> Path:
        return Path(str(self.config.get("service_dir", "") or "")).expanduser()

    def _python(self) -> Path:
        return self._svc() / ".venv" / "bin" / "python"

    def _tts_url(self) -> str:
        tts_port = int(self.config.get("tts_port", 9881))
        return str(self.config.get("public_url") or f"http://127.0.0.1:{tts_port}")

    def _load_ref_env(self) -> dict:
        """读取 service_dir/tts_ref.env, 得到默认参考音环境变量。"""
        env: dict = {}
        ref_file = self._svc() / "tts_ref.env"
        if ref_file.is_file():
            try:
                for line in ref_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip().strip('"').strip("'")
            except OSError as e:
                logger.warning("local_tts: 读取 tts_ref.env 失败: %s", e)
        return env

    # ---------------- 启停服务 ----------------
    async def _start_service(self) -> None:
        async with self._proc_lock:
            if self._engine_proc and self._engine_proc.poll() is None:
                logger.info("local_tts: TTS 服务已在运行, 跳过启动")
                return
            svc = self._svc()
            if not (svc / "GPT-SoVITS").exists():
                logger.error(f"local_tts: service_dir 无效或未部署: {svc}")
                return
            py = self._python()
            if not py.exists():
                logger.error(f"local_tts: 找不到 .venv 的 python: {py}")
                return
            engine_port = int(self.config.get("engine_port", 9880))
            tts_port = int(self.config.get("tts_port", 9881))
            s2 = str(self.config.get("engine_model_s2",
                "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"))
            s1 = str(self.config.get("engine_model_s1",
                "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/"
                "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"))

            # 1. GPT-SoVITS 推理引擎 (cwd 必须是 GPT-SoVITS/ 目录, 模型路径相对该目录)
            engine_cmd = [
                str(py), "api.py",
                "-d", "mps", "-a", "0.0.0.0", "-p", str(engine_port),
                "-s", s2, "-g", s1,
            ]
            engine_log = open(ENGINE_LOG, "ab")
            try:
                self._engine_proc = subprocess.Popen(
                    engine_cmd,
                    cwd=str(svc / "GPT-SoVITS"),
                    stdout=engine_log,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                logger.info(f"local_tts: 引擎已启动 PID={self._engine_proc.pid}")
            except Exception as e:
                logger.error(f"local_tts: 启动引擎失败: {e}")
                return

            # 2. 调音包装服务 (内置 /tts 与 /infer_single)
            ref_env = self._load_ref_env()
            env = os.environ.copy()
            env["GPT_SOVITS_ENGINE"] = f"http://127.0.0.1:{engine_port}"
            env["TTS_PORT"] = str(tts_port)
            env["TTS_PUBLIC_URL"] = self._tts_url()
            env["TTS_DEFAULT_SPEED"] = str(float(self.config.get("default_speed", 1.0)))
            env["TTS_DEFAULT_PITCH"] = str(float(self.config.get("default_pitch", 0.0)))
            env["TTS_DEFAULT_VOLUME"] = str(float(self.config.get("default_volume", 0.0)))
            for k in ("TTS_REF_PATH", "TTS_REF_TEXT", "TTS_REF_LANG"):
                if k in ref_env:
                    env[k] = ref_env[k]
            wrapper_log = open(WRAPPER_LOG, "ab")
            try:
                self._wrapper_proc = subprocess.Popen(
                    [str(py), str(PLUGIN_DIR / "astrbot_tts_server.py")],
                    cwd=str(svc),
                    stdout=wrapper_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                logger.info(f"local_tts: 包装服务已启动 PID={self._wrapper_proc.pid}")
            except Exception as e:
                logger.error(f"local_tts: 启动包装服务失败: {e}")

    async def _stop_service(self) -> None:
        async with self._proc_lock:
            for name, proc in (
                ("包装服务", self._wrapper_proc),
                ("引擎", self._engine_proc),
            ):
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=8)
                        except asyncio.TimeoutError:
                            proc.kill()
                        logger.info(f"local_tts: {name} 已停止")
                    except Exception as e:
                        logger.warning(f"local_tts: 停止{name}异常: {e}")
            self._engine_proc = None
            self._wrapper_proc = None

    # ---------------- 生命周期 ----------------
    async def initialize(self) -> None:
        """AstrBot 激活本插件时调用: 自动拉起 TTS 服务。"""
        if bool(self.config.get("auto_start", True)):
            await self._start_service()

    async def terminate(self) -> None:
        """AstrBot 禁用/重载本插件或关闭时调用: 停止 TTS 服务。"""
        await self._stop_service()

    # ---------------- 命令 ----------------
    @filter.command("tts_status")
    async def tts_status(self, event: AstrMessageEvent):
        eng = ("运行中(PID %s)" % self._engine_proc.pid) if (
            self._engine_proc and self._engine_proc.poll() is None
        ) else "未运行"
        wrap = ("运行中(PID %s)" % self._wrapper_proc.pid) if (
            self._wrapper_proc and self._wrapper_proc.poll() is None
        ) else "未运行"
        yield event.plain_result(
            f"本地 TTS 服务状态\n"
            f"引擎: {eng}\n"
            f"包装: {wrap}\n"
            f"TTS 接口: {self._tts_url()}/tts\n"
            f"AstrBot GSVI 配置 api_base: {self._tts_url()}"
        )

    @filter.command("tts_restart")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def tts_restart(self, event: AstrMessageEvent):
        yield event.plain_result("正在重启本地 TTS 服务...")
        await self._stop_service()
        await asyncio.sleep(2)
        await self._start_service()
        yield event.plain_result(
            f"本地 TTS 服务已重启。TTS 接口: {self._tts_url()}/tts "
            "(模型加载约 20-30 秒)"
        )
