#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调音 + 克隆 TTS 包装服务  (AstrBot 前端接口)
=========================================

架构
----
AstrBot  ->  本包装服务 (端口 9881)  ->  GPT-SoVITS 推理引擎 (端口 9880)

本服务是 AstrBot 唯一需要配置的接口，负责:
  1. 参数转发: text / text_language / speed / refer_wav_path / prompt_text ...
  2. 调音后处理: pitch (音调, 半音), volume (音量, dB)
  3. 音频转码: wav / ogg / mp3
  4. 默认参考音: 未传 refer_wav_path 时使用预设参考音(可通过 -r 指定)

调用示例 (AstrBot 或 curl)
--------------------------
POST /tts   JSON:
{
  "text": "你好，欢迎使用语音助手。",
  "text_language": "zh",
  "speed": 1.0,       # 语速, 0.5~2.0, 1.0=正常
  "pitch": 0.0,       # 音调, 单位=半音, -12~12, 0=正常
  "volume": 0.0,      # 音量, 单位=dB, -20~20, 0=正常
  "top_k": 15, "top_p": 1.0, "temperature": 1.0,
  "output_format": "wav"   # wav / ogg / mp3
}
返回: 音频二进制

GET /tts?text=你好&speed=1.0&pitch=2&output_format=ogg
  快速测试用

GET /health       健康检查
GET /             服务信息 + 当前默认参考音
"""

import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx
import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.responses import Response, JSONResponse

# ---------------- 配置 ----------------
ENGINE_URL = os.environ.get("GPT_SOVITS_ENGINE", "http://127.0.0.1:9880")

# 默认参考音 (模仿学习的基准音色): 路径 / 对应文本 / 语种
DEFAULT_REF_PATH = os.environ.get(
    "TTS_REF_PATH",
    str(Path(__file__).parent / "refs" / "base_ref.wav"),
)
DEFAULT_REF_TEXT = os.environ.get(
    "TTS_REF_TEXT",
    "你好，我是你的语音助手，很高兴认识你。",
)
DEFAULT_REF_LANG = os.environ.get("TTS_REF_LANG", "zh")

HOST = os.environ.get("TTS_BIND", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "9881"))

# AstrBot GSVI 兼容端点 (/infer_single) 的默认调音参数
DEFAULT_SPEED = float(os.environ.get("TTS_DEFAULT_SPEED", "1.0"))
DEFAULT_PITCH = float(os.environ.get("TTS_DEFAULT_PITCH", "0.0"))
DEFAULT_VOLUME = float(os.environ.get("TTS_DEFAULT_VOLUME", "0.0"))

# 对外可访问的根地址, 用于生成 /infer_single 返回的 audio_url
PUBLIC_URL = os.environ.get("TTS_PUBLIC_URL", f"http://127.0.0.1:{PORT}")
PUBLIC_URL = PUBLIC_URL.removesuffix("/")

# 生成的音频暂存目录 (供 /audio/{fname} 下载)
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "localtts_audio"
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_FORMATS = {"wav", "ogg", "mp3"}
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# AstrBot 传入的中文语种标签 -> 引擎语言码
_LANG_MAP = {"中文": "zh", "英文": "en", "自动": "auto", "中英混合": "auto"}

app = FastAPI(title="AstrBot 调音 TTS 服务", version="1.0")

# 单模型 TTS 引擎非线程安全: 用锁串行化所有请求
_tts_lock = threading.Lock()


def _engine_tts(payload: dict) -> bytes:
    """调用 GPT-SoVITS 引擎, 返回 wav 字节。

    trust_env=False: 忽略系统代理(HTTP_PROXY/macOS 系统代理), 直连本机引擎。
    """
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        resp = client.post(f"{ENGINE_URL}/", json=payload)
        resp.raise_for_status()
        return resp.content


def _apply_tuning(wav_bytes: bytes, pitch: float, volume_db: float):
    """对 wav 做 音调(pitch 半音) + 音量(volume dB) 后处理, 返回 wav 字节。"""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:  # 多声道转单声道
        data = data.mean(axis=1)

    if pitch != 0.0:
        data = librosa.effects.pitch_shift(data, sr=sr, n_steps=float(pitch))

    if volume_db != 0.0:
        gain = 10.0 ** (float(volume_db) / 20.0)
        data = data * gain
        maxv = float(np.abs(data).max()) if len(data) else 0.0
        if maxv > 1.0:  # 防削波
            data = data / maxv

    out = io.BytesIO()
    sf.write(out, data, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()


def _transcode(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """wav -> ogg/mp3 转码 (使用 ffmpeg)。返回 (audio_bytes, media_type)。"""
    if fmt == "wav":
        return wav_bytes, "audio/wav"
    if fmt not in ALLOWED_FORMATS:
        fmt = "wav"
    codec = {"ogg": "libopus", "mp3": "libmp3lame"}[fmt]
    ext = {"ogg": "ogg", "mp3": "mp3"}[fmt]
    # ogg/opus 用 48k 码率(语音质量), mp3 用 128k
    bitrate = "48k" if fmt == "ogg" else "128k"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(wav_bytes)
        wav_path = tf.name
    out_path = wav_path + "." + ext
    try:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", wav_path,
             "-vn", "-ac", "1", "-b:a", bitrate, "-c:a", codec, out_path],
            check=True, capture_output=True,
        )
        with open(out_path, "rb") as f:
            data = f.read()
    finally:
        for p in (wav_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
    media = "audio/ogg" if fmt == "ogg" else "audio/mpeg"
    return data, media


def _resolve_ref(refer_wav_path, prompt_text, prompt_language):
    """未提供参考音时回退到默认参考音。"""
    if not refer_wav_path or not prompt_text or not prompt_language:
        return DEFAULT_REF_PATH, DEFAULT_REF_TEXT, DEFAULT_REF_LANG
    return refer_wav_path, prompt_text, prompt_language


def _do_tts(text, text_language, speed, pitch, volume, refer_wav_path,
            prompt_text, prompt_language, top_k, top_p, temperature,
            cut_punc, output_format):
    if not text or not str(text).strip():
        return JSONResponse({"code": 400, "message": "text 不能为空"}, status_code=400)
    text = str(text).strip()

    rp, pt, pl = _resolve_ref(refer_wav_path, prompt_text, prompt_language)

    payload = {
        "refer_wav_path": rp,
        "prompt_text": pt,
        "prompt_language": pl,
        "text": text,
        "text_language": text_language or "zh",
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "speed": float(speed),
    }
    if cut_punc:
        payload["cut_punc"] = cut_punc

    # 串行化: 同一时刻只有一个请求进入推理引擎
    with _tts_lock:
        try:
            wav_bytes = _engine_tts(payload)
        except Exception as e:
            return JSONResponse(
                {"code": 502, "message": f"TTS 引擎调用失败: {e}"}, status_code=502
            )
        wav_bytes = _apply_tuning(wav_bytes, float(pitch or 0.0), float(volume or 0.0))

    fmt = (output_format or "wav").lower()
    if fmt not in ALLOWED_FORMATS:
        fmt = "wav"
    audio, media = _transcode(wav_bytes, fmt)
    return Response(content=audio, media_type=media)


@app.post("/tts")
async def tts_post(request: Request):
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    def g(key, default=None):
        v = body.get(key, default)
        return default if v in (None, "") else v

    return await asyncio.to_thread(
        _do_tts,
        text=g("text"),
        text_language=g("text_language", "zh"),
        speed=g("speed", 1.0),
        pitch=g("pitch", 0.0),
        volume=g("volume", 0.0),
        refer_wav_path=g("refer_wav_path"),
        prompt_text=g("prompt_text"),
        prompt_language=g("prompt_language"),
        top_k=g("top_k", 15),
        top_p=g("top_p", 1.0),
        temperature=g("temperature", 1.0),
        cut_punc=g("cut_punc"),
        output_format=g("output_format", "wav"),
    )


@app.get("/tts")
async def tts_get(
    text: str = Query(...),
    text_language: str = Query("zh"),
    speed: float = Query(1.0),
    pitch: float = Query(0.0),
    volume: float = Query(0.0),
    refer_wav_path: str = Query(None),
    prompt_text: str = Query(None),
    prompt_language: str = Query(None),
    top_k: int = Query(15),
    top_p: float = Query(1.0),
    temperature: float = Query(1.0),
    cut_punc: str = Query(None),
    output_format: str = Query("wav"),
):
    return await asyncio.to_thread(
        _do_tts, text, text_language, speed, pitch, volume, refer_wav_path,
        prompt_text, prompt_language, top_k, top_p, temperature,
        cut_punc, output_format,
    )


@app.get("/health")
async def health():
    import socket
    from urllib.parse import urlparse
    engine_ok = False
    try:
        u = urlparse(ENGINE_URL)
        with socket.create_connection((u.hostname, u.port or 9880), timeout=2.0):
            engine_ok = True
    except Exception:
        engine_ok = False
    return {"status": "ok", "engine": "up" if engine_ok else "down",
            "engine_url": ENGINE_URL}


@app.get("/")
async def info():
    return {
        "service": "AstrBot 调音 TTS 服务",
        "version": "1.0",
        "engine_url": ENGINE_URL,
        "default_reference": {
            "path": DEFAULT_REF_PATH,
            "text": DEFAULT_REF_TEXT,
            "language": DEFAULT_REF_LANG,
        },
        "params": {
            "text": "必填, 要合成的文本",
            "speed": "语速 0.5~2.0",
            "pitch": "音调(半音) -12~12",
            "volume": "音量(dB) -20~20",
            "refer_wav_path": "参考音频(模仿学习源), 可选",
            "prompt_text": "参考音频对应文本, 可选",
            "output_format": "wav/ogg/mp3",
        },
    }


# ---------------- AstrBot GSVI TTS 兼容端点 ----------------

def _clean_old_audio(max_age_sec: int = 3600):
    """清理暂存音频目录里超过 max_age_sec 的旧文件。"""
    import time
    now = time.time()
    try:
        for f in TEMP_AUDIO_DIR.iterdir():
            try:
                if f.is_file() and (now - f.stat().st_mtime) > max_age_sec:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _norm_lang(lang: str | None) -> str:
    """把 AstrBot 传入的 '中文'/'英文'/'auto' 映射成引擎语言码 zh/en/auto。"""
    if not lang:
        return "zh"
    lang = str(lang).strip()
    return _LANG_MAP.get(lang, lang if lang in ("zh", "en", "auto") else "zh")


@app.post("/infer_single")
async def infer_single(request: Request):
    """AstrBot 内置 'GSVI TTS(API)' 源调用的端点。

    AstrBot 发送: {text, text_lang, prompt_text_lang, prompt_text?, model_name,
                   emotion, version, dl_url}
    期望返回: {"msg": "合成成功", "audio_url": "http://.../audio/xxx.wav"}
    """
    body = await request.json()
    text = str(body.get("text", "")).strip()
    if not text:
        return JSONResponse({"msg": "合成失败", "reason": "text 不能为空"}, status_code=400)

    # AstrBot 的 prompt_text_lang 是 '中文'/'英文', text_lang 同理
    text_lang = _norm_lang(body.get("text_lang", "中文"))
    prompt_text = body.get("prompt_text") or DEFAULT_REF_TEXT
    prompt_lang = _norm_lang(body.get("prompt_text_lang", "中文"))
    refer_path = body.get("refer_wav_path") or DEFAULT_REF_PATH

    audio = await asyncio.to_thread(
        _do_tts,
        text=text,
        text_language=text_lang,
        speed=DEFAULT_SPEED,
        pitch=DEFAULT_PITCH,
        volume=DEFAULT_VOLUME,
        refer_wav_path=refer_path,
        prompt_text=prompt_text,
        prompt_language=prompt_lang,
        top_k=15, top_p=1.0, temperature=1.0,
        cut_punc=None, output_format="wav",
    )
    if isinstance(audio, JSONResponse):  # 出错时是 JSONResponse, 直接返回错误
        return audio

    import uuid
    _clean_old_audio()
    fname = f"gsvi_{uuid.uuid4().hex}.wav"
    with (TEMP_AUDIO_DIR / fname).open("wb") as f:
        f.write(audio.body)
    return {"msg": "合成成功", "audio_url": f"{PUBLIC_URL}/audio/{fname}"}


@app.get("/audio/{fname}")
async def get_audio(fname: str):
    """供 AstrBot 下载 /infer_single 生成的音频。"""
    safe = Path(fname).name  # 防路径穿越
    fpath = TEMP_AUDIO_DIR / safe
    if not fpath.is_file():
        return JSONResponse({"msg": "音频不存在"}, status_code=404)
    data = fpath.read_bytes()
    media = "audio/wav" if safe.endswith(".wav") else "application/octet-stream"
    return Response(content=data, media_type=media)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
