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
import json
import os
import re
import subprocess
import tempfile
from dataclasses import field
from pathlib import Path
from typing import Any

import requests

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Plain, Record
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic.dataclasses import dataclass

PLUGIN_DIR = Path(__file__).resolve().parent

ENGINE_LOG = "/tmp/localtts_engine.log"
WRAPPER_LOG = "/tmp/localtts_wrapper.log"
SING_LOG = "/tmp/localtts_sing.log"

# ---------------- 网易云点歌 (点歌并唱出来) 常量 ----------------
# 参考 astrbot_plugin_netease_pick 的官方直链方案 (免 key 免登录)。
NETEASE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
NETEASE_ID_RE = re.compile(r"[?&]id=(\d+)")
NETEASE_LINK_RE = re.compile(r"https?://[^\s\"'<>()（）【】]+")
NETEASE_OUTER_URL = "http://music.163.com/song/media/outer/url?id={sid}.mp3"
# 网易云音频体积下限: 真歌至少几百 KB; 错误页 ~104 KB / -460 JSON 83 字节。
NETEASE_MIN_AUDIO_BYTES = 200 * 1024

# ---- B站 (Bilibili) 音源: 用官方 API 免登录拿 DASH 音频流 (m4a) + ffmpeg 转 wav ----
BILI_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BILI_VIEW_URL = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
BILI_PLAYURL_URL = "https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&fnval=16&fnver=0&fourk=1"
BILI_BV_RE = re.compile(r"(BV[0-9A-Za-z]{8,12})")
BILI_REFERER = "https://www.bilibili.com/"


@dataclass
class SingSongTool(FunctionTool[AstrAgentContext]):
    """LLM 工具: 用小鲸鱼唱一首歌曲 (注册为 function calling, 供大模型调用)。

    call() 里 context.context.event 是触发工具的消息上下文, 用它把进度和最终音频
    发回当前聊天。plugin 为 LocalTtsPlugin 实例 (构造函数传入)。
    """

    __pydantic_config__ = {"arbitrary_types_allowed": True}

    plugin: Any = None
    name: str = "sing_song"
    description: str = (
        "用小鲸鱼唱一首歌曲的完整版(保留伴奏)。source 传歌曲来源: 本地音频路径、"
        "网易云歌曲链接、或 B站视频链接; ref 可选参考音色(音频路径, 留空用默认小鲸鱼)。"
        "唱完会把音频发到当前聊天。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "歌曲来源: 本地音频路径、网易云歌曲链接、或 B站视频链接",
                },
                "ref": {
                    "type": "string",
                    "description": "可选参考音色(音频路径); 留空用默认小鲸鱼参考音",
                },
            },
            "required": ["source"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], source: str = "", ref: str = ""
    ):
        event = getattr(context.context, "event", None)
        if event is None or self.plugin is None:
            yield "sing_song 工具无可用消息上下文或插件未初始化。"
            return
        async for r in self.plugin._llm_sing_tool_impl(event, source, ref):
            yield r


@dataclass
class SearchSongTool(FunctionTool[AstrAgentContext]):
    """LLM 工具: 按歌名在网易云 + B站 搜索歌曲候选 (点歌前的找歌环节)。

    call() 返回候选列表 JSON: [{platform, title, artist, id, url}, ...]。
    大模型把候选发给用户确认, 用户选定后再调 sing_song。
    """

    __pydantic_config__ = {"arbitrary_types_allowed": True}

    plugin: Any = None
    name: str = "search_song"
    description: str = (
        "按歌名(或 歌名+歌手)在网易云和B站搜索歌曲, 返回候选列表(平台/标题/歌手/链接)。"
        "用户想唱歌但没给链接时用; 把候选展示给用户确认, 选定后调用 sing_song。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "歌名或 歌名+歌手, 例如: 晴天 周杰伦",
                },
            },
            "required": ["keyword"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], keyword: str = ""):
        if self.plugin is None:
            yield "search_song 插件未初始化。"
            return
        try:
            cands = await self.plugin._search_song(keyword)
        except Exception as e:
            yield f"搜索失败: {e}"
            return
        yield json.dumps(cands, ensure_ascii=False)


class LocalTtsPlugin(Star):
    """随 AstrBot 启停本地 GPT-SoVITS 调音 TTS 服务。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._engine_proc: subprocess.Popen | None = None
        self._wrapper_proc: subprocess.Popen | None = None
        self._sing_proc: subprocess.Popen | None = None
        self._proc_lock = asyncio.Lock()
        # 延迟注册 LLM 工具: provider_manager 可能尚未就绪, 后台重试注册 sing_song
        self._llm_tool_task = asyncio.create_task(self._defer_register_llm_tool())

    async def _defer_register_llm_tool(self):
        """后台重试注册 sing_song / search_song LLM 工具 (等 provider_manager 就绪)。"""
        for attempt in range(10):
            try:
                self.context.add_llm_tools(
                    SingSongTool(plugin=self), SearchSongTool(plugin=self)
                )
                print("[local_tts] 🔧 唱歌已注册为 LLM 工具: sing_song / search_song", flush=True)
                return
            except Exception as e:
                if attempt == 9:
                    logger.warning(f"[local_tts] 注册 LLM 唱歌工具失败(已重试): {e}")
                    return
                await asyncio.sleep(1)

    async def _llm_sing_tool_impl(self, event, source="", ref=""):
        """LLM 工具处理器: 用小鲸鱼唱一首歌。event 为触发工具的消息上下文。

        与 /点歌 类似: 自动识别网易云/B站/本地, 下载后调 _sing_mix_impl, 进度+音频回给当前聊天。
        """
        if not bool(self.config.get("sing_enable", False)):
            yield event.plain_result("唱歌模式未启用 (sing_enable=false)。")
            return
        if not self._sing_ready():
            yield event.plain_result("唱歌服务未运行, 请先配置并启动 (sing_enable=true)。")
            return
        source = (source or "").strip()
        if not source:
            yield event.plain_result("sing_song 需要 source 参数 (本地音频路径 / 网易云链接 / B站链接)。")
            return
        src_path = None
        try:
            if "bilibili.com" in source or "b23.tv" in source or BILI_BV_RE.search(source):
                yield event.plain_result("🎵 LLM 调用唱歌: 检测到 B站链接, 提取音频中...")
                print(f"[local_tts] 🎵 LLM 工具 sing_song: B站 {source}", flush=True)
                src_path = await self._bilibili_download(source)
            elif "music.163.com" in source or "163cn.tv" in source or NETEASE_ID_RE.search(source):
                sid = await self._netease_resolve_id(source)
                if not sid:
                    yield event.plain_result("无法解析网易云歌曲 id。")
                    return
                print(f"[local_tts] 🎵 LLM 工具 sing_song: 网易云 id={sid}", flush=True)
                src_path = await self._netease_download(sid)
            else:
                src_path = str(Path(source).expanduser())
                if not Path(src_path).exists():
                    yield event.plain_result(f"本地文件不存在: {src_path}")
                    return
        except Exception as e:
            yield event.plain_result(f"下载歌曲失败: {e}")
            return
        try:
            async for r in self._sing_mix_impl(event, src_path, ref):
                yield r
        finally:
            try:
                if src_path:
                    os.unlink(src_path)
            except OSError:
                pass

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
        # AstrBot 4.29: event.message 已不存在, getattr(event,"message",[]) 恒为 []
        # 导致 _auto_sing 永远找不到附件而静默返回。改用 get_messages() / message_obj.message。
        chain = []
        if hasattr(event, "get_messages"):
            chain = event.get_messages() or []
        elif getattr(event, "message_obj", None) is not None:
            chain = getattr(event.message_obj, "message", []) or []
        else:
            chain = (
                getattr(event, "message_chain", None)
                or getattr(event, "message", None)
                or []
            )
        for comp in chain:
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
                    "diffusion_steps": int(self.config.get("sing_diffusion_steps", 6)),
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

        # 并发防护: 同一时刻只允许一个唱歌任务, 避免多个请求撞车/争抢资源。
        if getattr(self, "_sing_busy", False):
            yield event.plain_result("已有唱歌任务进行中, 请等它完成后再试。")
            return
        self._sing_busy = True
        try:
            yield event.plain_result("🎤 正在用小鲸鱼唱完整歌曲... (分离人声+伴奏+转换, 约需 1-3 分钟)")
            print("[local_tts] 🎤 开始用小鲸鱼唱完整歌曲 (分离人声+伴奏+转换)...", flush=True)
            # 异步化: POST 放后台线程, 主循环不被卡住; 同时轮询 /progress 把进度实时吐给用户。
            import threading, queue as _queue
            _result_q = _queue.Queue()
            _payload = {
                "source": source,
                "target": str(ref_path),
                "diffusion_steps": int(self.config.get("sing_diffusion_steps", 6)),
                "pitch_shift": float(self.config.get("sing_pitch_shift", 2)),
            }

            def _do_sing():
                try:
                    resp = requests.post(
                        f"{self._sing_url()}/sing_mix", json=_payload, timeout=900
                    )
                    _result_q.put(("ok", resp))
                except Exception as e:
                    _result_q.put(("err", e))

            threading.Thread(target=_do_sing, daemon=True).start()
            loop = asyncio.get_running_loop()
            last_stage, last_pct = "", -10
            kind, val = None, None
            while True:
                try:
                    kind, val = _result_q.get_nowait()
                    break
                except _queue.Empty:
                    pass
                try:
                    pr = await loop.run_in_executor(
                        None,
                        lambda: requests.get(f"{self._sing_url()}/progress", timeout=2).json(),
                    )
                    stage = str(pr.get("stage", ""))
                    pct = int(pr.get("pct", 0))
                    detail = str(pr.get("detail", ""))
                    if stage != last_stage or pct >= last_pct + 5:
                        msg = f"🎤 [{stage}] {pct}%"
                        if detail:
                            msg += f" {detail}"
                        yield event.plain_result(msg)
                        print(f"[local_tts] {msg}", flush=True)
                        last_stage, last_pct = stage, pct
                except Exception:
                    pass
                await asyncio.sleep(3)

            if kind == "err":
                print(f"[local_tts] ❌ 唱歌合成异常: {val}", flush=True)
                yield event.plain_result(f"唱歌合成异常: {val}")
            elif val.status_code != 200:
                print(f"[local_tts] ❌ 唱歌合成失败: HTTP {val.status_code} {val.text[:200]}", flush=True)
                yield event.plain_result(f"唱歌合成失败: HTTP {val.status_code} {val.text[:200]}")
            else:
                out = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
                out.write(val.content)
                out.close()
                print(f"[local_tts] ✅ 小鲸鱼唱完整歌曲完成: {len(val.content)//1024}KB", flush=True)
                yield event.chain_result([
                    Plain(f"小鲸鱼唱完整歌曲完成: {len(val.content)//1024}KB"),
                    Record(file=out.name),
                ])
                # 延迟删除输出音频, 避免堆积占用存储 (发送完成后清理)
                try:
                    asyncio.get_running_loop().call_later(60, self._safe_unlink, out.name)
                except Exception:
                    pass
        finally:
            self._sing_busy = False
            for _t in (tmp, ref_tmp):
                if _t:
                    try:
                        os.unlink(_t)
                    except OSError:
                        pass

    def _safe_unlink(self, path):
        """尽力删除文件 (清理唱歌输出, 避免堆积占用存储)。"""
        try:
            if path and os.path.exists(path):
                os.unlink(path)
                print(f"[local_tts] 🗑️ 已清理唱歌输出: {path}", flush=True)
        except OSError:
            pass

    @filter.command("点歌")
    async def dian_ge(self, event: AstrMessageEvent, source_path: str = "", ref: str = ""):
        """点歌并唱出来: 发网易云或 B站链接, 下载音频后用参考音色唱完整歌曲。

        /点歌 <网易云歌曲链接>    例如 /点歌 https://music.163.com/song?id=3413072220
                                  或 /点歌 https://163cn.tv/xxxx (QQ 分享短链, 先跟随 302 拿 id)
        /点歌 <B站视频链接>       例如 /点歌 https://www.bilibili.com/video/BV1xx411c7mD
                                  或 /点歌 https://b23.tv/xxxx (QQ 分享短链)
        需要 sing_enable=true 且唱歌服务运行 (9882)。网易云走官方直链免登录 (免费歌);
        B站用官方 API 免登录取 DASH 音频流 + ffmpeg 转 wav (几乎任意视频都能提音频, 不受版权限制)。
        """
        raw = getattr(event, "message_str", "") or ""
        m = NETEASE_LINK_RE.search(raw)
        if not m:
            # 兜底: 从命令参数里拼 (AstrBot 把消息按空格切分, 链接可能在任一 token)
            candidate = " ".join([source_path, ref])
            m = NETEASE_LINK_RE.search(candidate)
        if not m:
            yield event.plain_result(
                "用法: /点歌 <网易云或B站链接>\n"
                "例如: /点歌 https://music.163.com/song?id=3413072220&userid=xxx\n"
                "或 /点歌 https://163cn.tv/xxxx (QQ 分享短链)\n"
                "或 /点歌 https://www.bilibili.com/video/BV1xx411c7mD (B站视频/歌曲)\n"
                "或 /点歌 https://b23.tv/xxxx (B站 QQ 分享短链)"
            )
            return
        link = m.group(0).strip()
        is_bili = ("bilibili.com" in link or "b23.tv" in link or BILI_BV_RE.search(link))
        src_path = None
        try:
            if is_bili:
                yield event.plain_result("🎵 点歌: 检测到 B站链接, 正在提取音频...")
                print(f"[local_tts] 🎵 点歌: B站链接 {link}, 提取音频+唱歌中...", flush=True)
                src_path = await self._bilibili_download(link)
            else:
                sid = await self._netease_resolve_id(link)
                if not sid:
                    yield event.plain_result(
                        "无法从该链接解析到网易云歌曲 id。请发完整链接 (含 ?id= 或 163cn.tv 短链)。"
                    )
                    return
                yield event.plain_result(f"🎵 点歌: 歌曲 id={sid}, 正在下载并准备用小鲸鱼唱...")
                print(f"[local_tts] 🎵 点歌: 网易云歌曲 id={sid} 链接={link}, 下载+唱歌中...", flush=True)
                src_path = await self._netease_download(sid)
        except Exception as e:
            yield event.plain_result(f"下载歌曲失败: {e}")
            return
        try:
            async for r in self._sing_mix_impl(event, src_path, ""):
                yield r
        finally:
            try:
                if src_path:
                    os.unlink(src_path)
            except OSError:
                pass

    async def _netease_resolve_id(self, link: str):
        """从网易云链接解析歌曲 ID。支持 ?id= 直链 与 163cn.tv 短链(跟随 302, aiohttp 同 netease_pick)。"""
        link = (link or "").strip()
        if not link:
            return None
        m = NETEASE_ID_RE.search(link)
        if m:
            return m.group(1)
        if "163cn.tv" in link:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(
                        link,
                        headers={"User-Agent": NETEASE_UA},
                        allow_redirects=True,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as r:
                        m = NETEASE_ID_RE.search(str(r.url))
                        if m:
                            return m.group(1)
            except Exception as e:
                logger.warning(f"[local_tts] 网易云短链解析失败: {e}")
        return None

    async def _netease_download(self, song_id: str):
        """下载网易云歌曲为临时 mp3 (aiohttp, 同 netease_pick)。

        三判据校验 (netease_pick 踩过的坑): Content-Type 是 audio/*、
        最终 URL 落到 music.126.net CDN、体积 >= 200KB。否则多半是版权受限/VIP。
        """
        import aiohttp
        import tempfile as _tf
        url = NETEASE_OUTER_URL.format(sid=song_id)
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                url,
                headers={"User-Agent": NETEASE_UA},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                ctype = str(resp.headers.get("Content-Type", "") or "")
                final = str(resp.url)
                data = await resp.read()
        if not ctype.startswith("audio/"):
            raise RuntimeError(f"网易云没给音频 (Content-Type={ctype or '未知'})")
        if "music.126.net" not in final:
            raise RuntimeError("没跳到网易云 CDN (music.126.net), 多半是版权受限/VIP 或链接失效")
        if len(data) < NETEASE_MIN_AUDIO_BYTES:
            raise RuntimeError(f"音频过小 ({len(data)} 字节)")
        f = _tf.NamedTemporaryFile(suffix=".mp3", delete=False)
        f.write(data)
        f.close()
        return f.name

    async def _bilibili_download(self, link: str):
        """下载 B站视频音频为临时 wav (aiohttp 官方 API + ffmpeg 转码)。返回 wav 路径。

        B站免登录拿 DASH 音频流: 先访问主站拿 buvid cookie, view API 取 cid,
        playurl API 取 audio[0].baseUrl, 下载 m4a, ffmpeg 转 44.1k 单声道 wav。
        支持 BV 直链 (https://www.bilibili.com/video/BV...) 与 b23.tv 短链。
        """
        import aiohttp
        import tempfile as _tf
        import shutil
        bvid = None
        m = BILI_BV_RE.search(link)
        if m:
            bvid = m.group(1)
        elif "b23.tv" in link:
            async with aiohttp.ClientSession(headers={"User-Agent": BILI_UA}) as sess:
                async with sess.get(
                    link, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=15)
                ) as r:
                    mm = BILI_BV_RE.search(str(r.url))
                    if mm:
                        bvid = mm.group(1)
        if not bvid:
            raise RuntimeError("无法从该 B站链接解析到 BV id")
        print(f"[local_tts] 🅱️ B站: BV={bvid}, 取音频流...", flush=True)
        async with aiohttp.ClientSession(headers={"User-Agent": BILI_UA, "Referer": BILI_REFERER}) as sess:
            # 先访问主站拿 buvid cookie (DASH 音频 CDN 需要)
            try:
                async with sess.get(BILI_REFERER, timeout=aiohttp.ClientTimeout(total=8)) as _r:
                    await _r.read()
            except Exception:
                pass
            async with sess.get(BILI_VIEW_URL.format(bvid=bvid), timeout=aiohttp.ClientTimeout(total=15)) as r:
                vj = await r.json()
            if vj.get("code") != 0 or not vj.get("data"):
                raise RuntimeError(f"B站 view API 失败: code={vj.get('code')}")
            vd = vj["data"]
            cid = vd.get("cid")
            title = vd.get("title", bvid)
            if not cid:
                raise RuntimeError("B站视频没有 cid")
            async with sess.get(BILI_PLAYURL_URL.format(bvid=bvid, cid=cid), timeout=aiohttp.ClientTimeout(total=15)) as r:
                pj = await r.json()
            auds = (pj.get("data") or {}).get("dash", {}).get("audio", [])
            if not auds:
                raise RuntimeError(f"B站没给 DASH 音频流 (可能需登录/大会员): code={pj.get('code')}")
            url = auds[0]["baseUrl"]
            async with sess.get(url, headers={"Referer": BILI_REFERER}, timeout=aiohttp.ClientTimeout(total=120)) as r:
                data = await r.read()
            if len(data) < 200 * 1024:
                raise RuntimeError(f"B站音频过小 ({len(data)} 字节)")
            m4a = _tf.NamedTemporaryFile(suffix=".m4a", delete=False)
            m4a.write(data)
            m4a.close()
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        wav = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", m4a.name, "-ar", "44100", "-ac", "1", wav.name]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        try:
            os.unlink(m4a.name)
        except OSError:
            pass
        if proc.returncode != 0:
            try:
                os.unlink(wav.name)
            except OSError:
                pass
            raise RuntimeError(f"B站音频转码失败: {err.decode(errors='ignore')[:200]}")
        print(f"[local_tts] 🅱️ B站音频就绪: {title} ({len(data)//1024}KB -> wav)", flush=True)
        return wav.name

    # ---------------- 搜歌 (网易云 + B站) ----------------
    async def _netease_search(self, keyword: str):
        """网易云按歌名搜索候选。返回 [{platform,title,artist,id,url}, ...]。
        免登录: 先访问主站拿 session cookie, 再调 api/search/get (text/plain 但内容是 JSON)。
        """
        import aiohttp
        kw = (keyword or "").strip()
        if not kw:
            return []
        out = []
        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": NETEASE_UA, "Referer": "https://music.163.com/"}
            ) as sess:
                try:
                    async with sess.get("https://music.163.com/", timeout=aiohttp.ClientTimeout(total=8)) as _r:
                        await _r.read()
                except Exception:
                    pass
                async with sess.get(
                    "https://music.163.com/api/search/get",
                    params={"s": kw, "type": 1, "limit": 6, "offset": 0},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as r:
                    txt = await r.text()
            j = json.loads(txt)
        except Exception as e:
            print(f"[local_tts] ⚠️ 网易云搜索失败: {type(e).__name__} {e}", flush=True)
            return []
        for so in (j.get("result") or {}).get("songs", [])[:6]:
            sid = so.get("id")
            if not sid:
                continue
            artists = " / ".join(a.get("name", "") for a in so.get("artists", []) if a.get("name"))
            out.append(
                {
                    "platform": "netease",
                    "title": so.get("name", ""),
                    "artist": artists,
                    "id": str(sid),
                    "url": f"https://music.163.com/song?id={sid}",
                }
            )
        return out

    async def _bilibili_search(self, keyword: str):
        """B站按关键词搜索视频。返回 [{platform,title,artist,id,url}, ...]。"""
        import aiohttp
        kw = (keyword or "").strip()
        if not kw:
            return []
        out = []
        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": BILI_UA, "Referer": BILI_REFERER}
            ) as sess:
                try:
                    async with sess.get(BILI_REFERER, timeout=aiohttp.ClientTimeout(total=8)) as _r:
                        await _r.read()
                except Exception:
                    pass
                async with sess.get(
                    "https://api.bilibili.com/x/web-interface/search/type",
                    params={"search_type": "video", "keyword": kw, "page": 1},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as r:
                    j = await r.json()
        except Exception as e:
            print(f"[local_tts] ⚠️ B站搜索失败: {type(e).__name__} {e}", flush=True)
            return []
        for v in ((j.get("data") or {}).get("result", []) or [])[:6]:
            bvid = v.get("bvid")
            if not bvid:
                continue
            title = re.sub(r"<[^>]+>", "", v.get("title", ""))
            out.append(
                {
                    "platform": "bilibili",
                    "title": title,
                    "artist": v.get("author", ""),
                    "id": bvid,
                    "url": f"https://www.bilibili.com/video/{bvid}",
                }
            )
        return out

    async def _search_song(self, keyword: str):
        """网易云 + B站 双平台搜歌, 返回候选列表 (网易云优先, 再 B站)。"""
        ne, bi = await asyncio.gather(
            self._netease_search(keyword), self._bilibili_search(keyword)
        )
        print(f"[local_tts] 🔍 搜歌 '{keyword}': 网易云 {len(ne)} 条, B站 {len(bi)} 条", flush=True)
        return ne + bi

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def _auto_sing(self, event: AstrMessageEvent):
        """自动唱歌: 私聊收到音频附件(无命令) -> 自动唱完整歌曲 (可用 sing_auto=false 关闭)。

        边缘控制: 只对【私聊】消息自动唱歌。群聊里任何人的语音(含其他 bot 如 Molly)
        都可能触发, 导致误克隆别人音色 + 与你自己的任务撞车; 群聊请用 /sing_mix 手动触发。
        """
        if not bool(self.config.get("sing_auto", True)):
            return
        if not bool(self.config.get("sing_enable", False)):
            return
        if not self._sing_ready():
            return
        # 边缘控制: 仅私聊自动唱歌; 群聊语音(其他 bot / 他人)一律不触发。
        try:
            if not event.is_private_chat():
                return
        except Exception:
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
