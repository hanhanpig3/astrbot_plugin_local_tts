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
import tempfile
from pathlib import Path

import requests

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Plain, Record
from astrbot.api.star import Context, Star

PLUGIN_DIR = Path(__file__).resolve().parent

ENGINE_LOG = "/tmp/localtts_engine.log"
WRAPPER_LOG = "/tmp/localtts_wrapper.log"
SING_LOG = "/tmp/localtts_sing.log"


class LocalTtsPlugin(Star):
    """随 AstrBot 启停本地 GPT-SoVITS 调音 TTS 服务。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._engine_proc: subprocess.Popen | None = None
        self._wrapper_proc: subprocess.Popen | None = None
        self._sing_proc: subprocess.Popen | None = None
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
                    k = k.strip()
                    if k.startswith("export "):
                        k = k[len("export "):].strip()
                    env[k] = v.strip().strip('"').strip("'")
            except OSError as e:
                logger.warning("local_tts: 读取 tts_ref.env 失败: %s", e)
        return env

    # ---------------- 唱歌 (Seed-VC) 辅助 ----------------
    def _sing_dir(self) -> Path:
        return Path(str(self.config.get("sing_dir", "") or "")).expanduser()

    def _sing_python(self) -> Path:
        # Seed-VC 独立环境用 sing_dir/venv (无点, 与 GPT-SoVITS 的 .venv 区分)
        return self._sing_dir() / "venv" / "bin" / "python"

    def _sing_ready(self) -> bool:
        """唱歌服务是否可用 (HTTP 探测, 不依赖 _sing_proc)。"""
        try:
            return requests.get(f"{self._sing_url()}/health", timeout=2).status_code == 200
        except Exception:
            return False

    def _resolve_path(self, p: str) -> Path:
        """相对路径相对 service_dir 解析, 绝对路径原样返回。"""
        p = str(p or "").strip()
        if not p:
            return Path()
        path = Path(p).expanduser()
        if path.is_absolute():
            return path
        return self._svc() / path

    def _sing_url(self) -> str:
        port = int(self.config.get("sing_port", 9882))
        return f"http://127.0.0.1:{port}"

    def _sing_ref(self) -> Path:
        return self._resolve_path(str(self.config.get("sing_ref_path", "")))

    async def _start_sing_service(self) -> None:
        if not bool(self.config.get("sing_enable", False)):
            return
        async with self._proc_lock:
            # 端口已通 (外部/此前手动启动的服务) 则跳过, 避免重复占用
            if self._sing_ready():
                logger.info("local_tts: 唱歌服务已在 9882 可用, 跳过启动")
                return
            if self._sing_proc and self._sing_proc.poll() is None:
                logger.info("local_tts: 唱歌服务已在运行, 跳过启动")
                return
            sing_dir = self._sing_dir()
            py = self._sing_python()
            if not py.exists():
                logger.error(f"local_tts: 找不到唱歌 venv 的 python: {py}")
                return
            env = os.environ.copy()
            env["SEEDVC_DIR"] = str(sing_dir / "seed-vc")
            env["SVC_CHECKPOINT"] = str(self._resolve_path(str(self.config.get("sing_checkpoint", ""))))
            env["SVC_CONFIG"] = str(self._resolve_path(str(self.config.get("sing_config", ""))))
            env["SING_PORT"] = str(int(self.config.get("sing_port", 9882)))
            sing_log = open(SING_LOG, "ab")
            try:
                self._sing_proc = subprocess.Popen(
                    [str(py), str(PLUGIN_DIR / "sing_server.py")],
                    cwd=str(sing_dir),
                    stdout=sing_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                logger.info(f"local_tts: 唱歌服务已启动 PID={self._sing_proc.pid}")
            except Exception as e:
                logger.error(f"local_tts: 启动唱歌服务失败: {e}")
                self._sing_proc = None

    async def _stop_sing_service(self) -> None:
        async with self._proc_lock:
            proc = self._sing_proc
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=8)
                    except asyncio.TimeoutError:
                        proc.kill()
                    logger.info("local_tts: 唱歌服务已停止")
                except Exception as e:
                    logger.warning(f"local_tts: 停止唱歌服务异常: {e}")
            self._sing_proc = None

    def _find_audio_attachment(self, event: AstrMessageEvent) -> str | None:
        """从消息里找音频附件 (语音/文件), 返回本地路径或 url。"""
        for comp in getattr(event, "message", []) or []:
            if isinstance(comp, Record):
                return comp.file or comp.url or ""
            if isinstance(comp, File):
                return comp.file_ or comp.url or ""
        return None

    def _find_audio_attachments(self, event: AstrMessageEvent) -> list:
        """返回消息里所有音频附件 (语音/文件) 的本地路径或 url。"""
        atts = []
        for comp in getattr(event, "message", []) or []:
            if isinstance(comp, Record):
                v = comp.file or comp.url or ""
                if v:
                    atts.append(v)
            if isinstance(comp, File):
                v = comp.file_ or comp.url or ""
                if v:
                    atts.append(v)
        return atts

    def _download_temp(self, url: str, suffix: str = ".wav"):
        """下载 url 到临时文件, 返回 (tmp_path, tmp_file)。"""
        import tempfile as _tf
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        tmp = _tf.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name, tmp

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
            device = str(self.config.get("engine_device", "") or "mps").strip().lower()
            if device not in ("mps", "cuda", "cpu"):
                device = "mps"
            engine_cmd = [
                str(py), "api.py",
                "-d", device, "-a", "0.0.0.0", "-p", str(engine_port),
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
        await self._start_sing_service()

    async def terminate(self) -> None:
        """AstrBot 禁用/重载本插件或关闭时调用: 停止 TTS 服务。"""
        await self._stop_sing_service()
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

    # ---------------- 唱歌命令 ----------------
    @filter.command("sing")
    async def sing(self, event: AstrMessageEvent, source_path: str = "", ref: str = ""):
        """唱歌: 用参考音色(默认小鲸鱼, 或 ref/第二个附件)唱用户提供的干声。

        /sing <干声路径>            干声是本机 wav/mp3 路径
        /sing                       附带音频附件作为干声; 若消息带两个附件, 第二个作参考音
        可选: ref=<参考音路径/URL>  覆盖参考音色 (本机路径或 http 文件链接)
        """
        if not bool(self.config.get("sing_enable", False)):
            yield event.plain_result("唱歌模式未启用。请先在插件配置里开启 sing_enable 并重启 AstrBot。")
            return
        if not self._sing_ready():
            yield event.plain_result("唱歌服务未运行。请使用 /sing_restart 启动 (首次模型加载约 1-2 分钟)。")
            return

        atts = self._find_audio_attachments(event)
        source = (source_path or "").strip()
        if not source and atts:
            source = atts[0]
        if not source:
            yield event.plain_result(
                "请提供唱歌干声：/sing <本机 wav/mp3 路径>，或直接发送音频附件。\n"
                "可选 ref=<参考音路径/URL> 覆盖默认小鲸鱼参考音；发两个附件时第二个作参考音。"
            )
            return

        # 参考音: 显式 ref 参数 > 第二个附件 > 配置默认
        ref_path = self._sing_ref()
        ref_tmp = None
        ref_val = ""
        if ref:
            ref_val = ref.strip()
            if ref_val.startswith("ref="):
                ref_val = ref_val[len("ref="):].strip()
        elif len(atts) > 1:
            ref_val = atts[1]
        if ref_val:
            if str(ref_val).startswith("http"):
                try:
                    _, ref_tmp = self._download_temp(ref_val)
                    ref_path = Path(ref_tmp.name)
                except Exception as e:
                    yield event.plain_result(f"下载参考音失败: {e}")
                    return
            else:
                ref_path = self._resolve_path(ref_val)
        if not ref_path.exists():
            yield event.plain_result(f"参考音不存在: {ref_path}")
            return

        # 若干声是 http url, 下载到临时文件
        import tempfile as _tf
        tmp = None
        if str(source).startswith("http"):
            try:
                source, tmp = self._download_temp(source)
            except Exception as e:
                yield event.plain_result(f"下载干声失败: {e}")
                return

        yield event.plain_result("🎤 正在用小鲸鱼音色转换... (唱歌干声时长决定耗时, 请稍候)")
        try:
            resp = requests.post(
                f"{self._sing_url()}/sing",
                json={
                    "source": source,
                    "target": str(ref_path),
                    "diffusion_steps": int(self.config.get("sing_diffusion_steps", 25)),
                    "pitch_shift": float(self.config.get("sing_pitch_shift", 2)),
                },
                timeout=600,
            )
            if resp.status_code != 200:
                yield event.plain_result(f"唱歌转换失败: HTTP {resp.status_code} {resp.text[:200]}")
                return
            # 保存 wav 并返回语音
            out = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
            out.write(resp.content)
            out.close()
            yield event.chain_result([
                Plain(f"小鲸鱼唱歌完成: {len(resp.content)//1024}KB"),
                Record(file=out.name),
            ])
        except Exception as e:
            yield event.plain_result(f"唱歌转换异常: {e}")
        finally:
            for _t in (tmp, ref_tmp):
                if _t:
                    try:
                        os.unlink(_t)
                    except OSError:
                        pass

    @filter.command("sing_mix")
    @filter.command("sing_mix")
    async def sing_mix(self, event: AstrMessageEvent, source_path: str = "", ref: str = ""):
        """唱歌+伴奏: 用参考音色唱完整歌曲 (自动 demucs 分离人声+伴奏, 合成完整版)。

        /sing_mix <歌曲路径>        完整歌曲是本机 wav/mp3 路径 (带伴奏)
        /sing_mix                   附带音频附件作为完整歌曲; 两个附件时第二个作参考音
        可选: ref=<参考音路径/URL>  覆盖参考音色 (本机路径或 http 文件链接)
        """
        async for r in self._sing_mix_impl(event, source_path, ref):
            yield r

    async def _sing_mix_impl(self, event: AstrMessageEvent, source_path: str = "", ref: str = ""):
        """完整歌曲唱歌核心逻辑 (供 /sing_mix 命令与自动唱歌复用)。"""
        if not bool(self.config.get("sing_enable", False)):
            yield event.plain_result("唱歌模式未启用。请先在插件配置里开启 sing_enable 并重启 AstrBot。")
            return
        if not self._sing_ready():
            yield event.plain_result("唱歌服务未运行。请使用 /sing_restart 启动 (首次模型加载约 1-2 分钟)。")
            return

        atts = self._find_audio_attachments(event)
        source = (source_path or "").strip()
        if not source and atts:
            source = atts[0]
        if not source:
            yield event.plain_result(
                "请提供完整歌曲：/sing_mix <本机 wav/mp3 路径>，或直接发送音频附件。\n"
                "可选 ref=<参考音路径/URL> 覆盖默认小鲸鱼参考音；发两个附件时第二个作参考音。"
            )
            return

        ref_path = self._sing_ref()
        ref_tmp = None
        ref_val = ""
        if ref:
            ref_val = ref.strip()
            if ref_val.startswith("ref="):
                ref_val = ref_val[len("ref="):].strip()
        elif len(atts) > 1:
            ref_val = atts[1]
        if ref_val:
            if str(ref_val).startswith("http"):
                try:
                    _, ref_tmp = self._download_temp(ref_val)
                    ref_path = Path(ref_tmp.name)
                except Exception as e:
                    yield event.plain_result(f"下载参考音失败: {e}")
                    return
            else:
                ref_path = self._resolve_path(ref_val)
        if not ref_path.exists():
            yield event.plain_result(f"参考音不存在: {ref_path}")
            return

        import tempfile as _tf
        tmp = None
        if str(source).startswith("http"):
            try:
                source, tmp = self._download_temp(source)
            except Exception as e:
                yield event.plain_result(f"下载歌曲失败: {e}")
                return

        yield event.plain_result("🎤 正在用小鲸鱼唱完整歌曲... (分离人声+伴奏+转换, 约需 1-3 分钟)")
        try:
            resp = requests.post(
                f"{self._sing_url()}/sing_mix",
                json={
                    "source": source,
                    "target": str(ref_path),
                    "diffusion_steps": int(self.config.get("sing_diffusion_steps", 25)),
                    "pitch_shift": float(self.config.get("sing_pitch_shift", 2)),
                },
                timeout=900,
            )
            if resp.status_code != 200:
                yield event.plain_result(f"唱歌合成失败: HTTP {resp.status_code} {resp.text[:200]}")
                return
            out = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
            out.write(resp.content)
            out.close()
            yield event.chain_result([
                Plain(f"小鲸鱼唱完整歌曲完成: {len(resp.content)//1024}KB"),
                Record(file=out.name),
            ])
        except Exception as e:
            yield event.plain_result(f"唱歌合成异常: {e}")
        finally:
            for _t in (tmp, ref_tmp):
                if _t:
                    try:
                        os.unlink(_t)
                    except OSError:
                        pass

    @filter.event_message()
    async def _auto_sing(self, event: AstrMessageEvent):
        """自动唱歌: 收到音频附件(无命令) -> 自动唱完整歌曲 (可用 sing_auto=false 关闭)。"""
        if not bool(self.config.get("sing_auto", True)):
            return
        if not bool(self.config.get("sing_enable", False)):
            return
        if not self._sing_ready():
            return
        # 跳过命令消息 (命令由 command handler 处理, 避免双触发)
        msg = getattr(event, "message_str", "") or ""
        if msg.strip().startswith("/"):
            return
        atts = self._find_audio_attachments(event)
        if not atts:
            return
        async for r in self._sing_mix_impl(event, "", ""):
            yield r

    @filter.command("sing_status")
    async def sing_status(self, event: AstrMessageEvent):
        if not bool(self.config.get("sing_enable", False)):
            yield event.plain_result("唱歌模式未启用 (sing_enable=false)。")
            return
        st = "可用" if self._sing_ready() else "未运行"
        if self._sing_proc and self._sing_proc.poll() is None:
            st += f" (PID {self._sing_proc.pid})"
        ref = self._sing_ref()
        yield event.plain_result(
            f"唱歌(Seed-VC)服务状态\n"
            f"服务: {st}\n"
            f"接口: {self._sing_url()}/sing\n"
            f"参考音: {ref if ref.exists() else '缺失!'}\n"
            f"用法: /sing <干声路径> 或 附带音频附件"
        )

    @filter.command("sing_restart")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def sing_restart(self, event: AstrMessageEvent):
        yield event.plain_result("正在重启唱歌服务... (首次模型加载约 1-2 分钟)")
        await self._stop_sing_service()
        await asyncio.sleep(2)
        await self._start_sing_service()
        yield event.plain_result(
            f"唱歌服务已重启。接口: {self._sing_url()}/sing"
        )
