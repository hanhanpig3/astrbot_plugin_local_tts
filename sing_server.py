"""Seed-VC 音色转换常驻 HTTP 服务 (SVC 模式, f0-condition=True, 音调保留)。

用于 AstrBot 本地 TTS 插件的「唱歌」功能:
  1. 用 Seed-VC SVC 模型 (DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2)
     做零样本音色转换, f0 强制保留 (音调对齐参考音中位数, 音调不乱)
  2. source = 唱歌干声(无伴奏人声), target = 参考音色 (如小鲸鱼说话参考音)
  3. POST /sing  {source, target}                     -> 纯人声 wav
     POST /sing_mix {source, target}                  -> 完整歌曲 wav (demucs 分离人声+伴奏, SVC 转人声, 混合伴奏)

依赖环境变量:
  SEEDVC_DIR   : seed-vc 仓库目录
  SVC_CHECKPOINT: SVC 主权重 .pth (DiT_seed_v2_uvit_whisper_base_f0_44k_..._v2.pth)
  SVC_CONFIG   : SVC config .yml (config_dit_mel_seed_uvit_whisper_base_f0_44k.yml)
  SING_PORT    : 服务端口 (默认 9882)
"""

import os
import sys
import json
import time
import tempfile
import threading
import argparse
import subprocess
import numpy as np
import torch
import torchaudio
import librosa
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEDVC_DIR = os.environ["SEEDVC_DIR"]
sys.path.insert(0, SEEDVC_DIR)
os.chdir(SEEDVC_DIR)

import inference as seedvc  # noqa: E402  # 加载 seed-vc 模块 (设置 device)


def _load_model():
    args = argparse.Namespace(
        checkpoint=os.environ.get("SVC_CHECKPOINT") or os.environ["VC_CHECKPOINT"],
        config=os.environ.get("SVC_CONFIG") or os.environ["VC_CONFIG"],
        fp16=True,
        f0_condition=True,
        diffusion_steps=int(os.environ.get("SVC_DIFFUSION_STEPS", 25)),
        length_adjust=float(os.environ.get("SVC_LENGTH_ADJUST", 1.0)),
        inference_cfg_rate=float(os.environ.get("SVC_CFG_RATE", 0.7)),
    )
    print("[sing] 加载 Seed-VC SVC 模型 (f0=True)...", flush=True)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_args = seedvc.load_models(args)
    print(f"[sing] 模型加载完成 sr={mel_args['sampling_rate']} device={seedvc.device}", flush=True)
    return model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_args["sampling_rate"]


MODEL, SEMANTIC, F0FN, VOCODER, CAMPPLUS, MELFN, SR = _load_model()
DEVICE = seedvc.device
HOP = 256 if SR == 22050 else 512
MAX_CTX = SR // HOP * 30
OVERLAP_FRAME = 16
OVERLAP_WAVE = OVERLAP_FRAME * HOP
_convert_lock = threading.Lock()

# ---- 进度追踪 (供 /progress 轮询, 让插件把唱歌进度实时吐给用户) ----
PROGRESS = {"stage": "空闲", "pct": 0, "detail": "", "t0": time.time()}
_progress_lock = threading.Lock()


def _set_progress(stage: str, pct: int, detail: str = ""):
    with _progress_lock:
        PROGRESS["stage"] = stage
        PROGRESS["pct"] = max(0, min(100, int(pct)))
        PROGRESS["detail"] = detail
        PROGRESS["t0"] = time.time()


def _get_progress():
    with _progress_lock:
        return dict(PROGRESS)


@torch.no_grad()
def convert(source, target, diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7, pitch_shift=0.0,
            progress_base=0, progress_span=100):
    """Seed-VC SVC 转换: source 干声 -> target 参考音色。pitch_shift>0 移高音调(女声化)。返回 (wave, sr)。

    progress_base/progress_span: 把本函数的进度映射到全局 [base, base+span] 区间。
    """
    _set_progress("音色转换", progress_base, "加载音频...")
    with _convert_lock:
        source_audio = librosa.load(source, sr=SR)[0]
        ref_audio = librosa.load(target, sr=SR)[0]
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(DEVICE)
        ref_audio = torch.tensor(ref_audio[: SR * 25]).unsqueeze(0).float().to(DEVICE)

        converted_waves_16k = torchaudio.functional.resample(source_audio, SR, 16000)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = SEMANTIC(converted_waves_16k)
        else:
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat(
                        [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                        dim=-1)
                S_alt = SEMANTIC(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, SR, 16000)
        S_ori = SEMANTIC(ori_waves_16k)
        mel = MELFN(source_audio.to(DEVICE).float())
        mel2 = MELFN(ref_audio.to(DEVICE).float())
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = CAMPPLUS(feat2.unsqueeze(0))

        # SVC: 提取 F0。保留源 F0 (auto_f0_adjust=False, 音调=原歌), 不对齐到参考
        F0_ori = F0FN(ori_waves_16k[0], thred=0.03)
        F0_alt = F0FN(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).float().to(DEVICE)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(DEVICE)[None]
        # 保留源 F0 (音调=原歌), 可选 pitch_shift 上移(女声化)
        shifted_f0_alt = F0_alt * (2 ** (pitch_shift / 12.0)) if pitch_shift else F0_alt

        cond, _, _, _, _ = MODEL.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
        prompt_condition, _, _, _, _ = MODEL.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

        max_source_window = MAX_CTX - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        _total_frames = cond.size(1)
        while processed_frames < cond.size(1):
            _set_progress(
                "音色转换",
                progress_base + int(processed_frames / max(_total_frames, 1) * progress_span),
                f"转换中 {processed_frames}/{_total_frames} 帧",
            )
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            if DEVICE.type == "mps":
                import contextlib
                _ctx = contextlib.nullcontext()
            else:
                _ctx = torch.autocast(device_type=DEVICE.type, dtype=torch.float16)
            with _ctx:
                vc_target = MODEL.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = VOCODER(vc_target.float()).squeeze()
            vc_wave = vc_wave[None, :]
            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                generated_wave_chunks.append(vc_wave[0, :-OVERLAP_WAVE].cpu().numpy())
                previous_chunk = vc_wave[0, -OVERLAP_WAVE:]
                processed_frames += vc_target.size(2) - OVERLAP_FRAME
            elif is_last_chunk:
                generated_wave_chunks.append(seedvc.crossfade(
                    previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), OVERLAP_WAVE))
                processed_frames += vc_target.size(2) - OVERLAP_FRAME
                break
            else:
                generated_wave_chunks.append(seedvc.crossfade(
                    previous_chunk.cpu().numpy(), vc_wave[0, :-OVERLAP_WAVE].cpu().numpy(), OVERLAP_WAVE))
                previous_chunk = vc_wave[0, -OVERLAP_WAVE:]
                processed_frames += vc_target.size(2) - OVERLAP_FRAME

        wave = np.concatenate(generated_wave_chunks)
        return wave.astype(np.float32), SR


def _demucs_vocals_and_accompaniment(audio_path: str, workdir: str):
    """demucs 分离人声+伴奏。返回 (vocals_path, accompaniment_path)。"""
    out_dir = os.path.join(workdir, "demucs")
    os.makedirs(out_dir, exist_ok=True)
    # demucs 用本 venv python (已装 demucs); -d mps 用 Apple GPU 加速 (M4 上接近实时)
    cmd = [
        sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
        "--two-stems=vocals", "-o", out_dir, "-d", "mps", audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(out_dir, "htdemucs", base)
    return os.path.join(stem_dir, "vocals.wav"), os.path.join(stem_dir, "no_vocals.wav")


def _mix(whale_wav: str, acc_wav: str, out_wav: str, vocal_gain=1.0, acc_gain=0.85):
    """混合小鲸鱼人声 + 伴奏 -> 完整歌曲。"""
    voc, sr1 = librosa.load(whale_wav, sr=None, mono=True)
    acc, sr2 = librosa.load(acc_wav, sr=None, mono=True)
    sr = min(sr1, sr2)
    voc = librosa.resample(voc, orig_sr=sr1, target_sr=sr) if sr1 != sr else voc
    acc = librosa.resample(acc, orig_sr=sr2, target_sr=sr) if sr2 != sr else acc
    n = min(len(voc), len(acc))
    voc, acc = voc[:n], acc[:n]
    mixed = vocal_gain * voc + acc_gain * acc
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    sf_write(out_wav, mixed.astype(np.float32), sr)
    return sr


def sf_write(path, data, sr):
    import soundfile as _sf
    _sf.write(path, data, sr)


def convert_and_mix(source, target, diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7,
                    vocal_gain=1.0, acc_gain=0.85):
    """完整流程: demucs 分离 -> SVC 转小鲸鱼 -> 混合伴奏。返回 (out_wav_path, sr)。"""
    _set_progress("分离人声+伴奏", 5, "demucs 分离中 (约1-2分钟)...")
    with tempfile.TemporaryDirectory() as workdir:
        vocals_path, acc_path = _demucs_vocals_and_accompaniment(source, workdir)
        _set_progress("分离人声+伴奏", 20, "demucs 完成")
        wave, sr = convert(vocals_path, target, diffusion_steps, length_adjust, inference_cfg_rate,
                           progress_base=20, progress_span=70)
        _set_progress("混合伴奏", 92, "混合中...")
        whale_wav = os.path.join(workdir, "whale.wav")
        import soundfile as _sf
        _sf.write(whale_wav, wave, sr)
        out_wav = os.path.join(workdir, "full.wav")
        _mix(whale_wav, acc_path, out_wav, vocal_gain, acc_gain)
        _set_progress("完成", 100, "")
        with open(out_wav, "rb") as f:
            data = f.read()
        return data, sr


class SingHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/progress"):
            body = json.dumps(_get_progress()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _send_wav(self, wav_bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        # 分块写, 避免大 WAV 单次 write 触发 BrokenPipe
        CHUNK = 1024 * 1024
        for i in range(0, len(wav_bytes), CHUNK):
            self.wfile.write(wav_bytes[i:i + CHUNK])
        try:
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def _send_error(self, e):
        import traceback
        traceback.print_exc()
        try:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            body = json.dumps({"error": str(e)}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_POST(self):
        try:
            # 精确匹配 /sing, 否则 /sing_mix 会被 startswith("/sing") 截走
            if self.path == "/sing":
                data = self._read_json()
                source = data["source"]
                target = data["target"]
                diffusion_steps = int(data.get("diffusion_steps", 25))
                length_adjust = float(data.get("length_adjust", 1.0))
                cfg_rate = float(data.get("inference_cfg_rate", 0.7))
                pitch_shift = float(data.get("pitch_shift", 0.0))
                t0 = time.time()
                _set_progress("音色转换", 0, "开始...")
                wave, sr = convert(source, target, diffusion_steps, length_adjust, cfg_rate, pitch_shift)
                _set_progress("完成", 100, "")
                import io
                buf = io.BytesIO()
                torchaudio.save(buf, torch.from_numpy(wave).unsqueeze(0), sr, format="wav")
                wav_bytes = buf.getvalue()
                print(f"[sing] /sing 转换完成 {time.time()-t0:.1f}s {len(wave)/sr:.1f}s audio", flush=True)
                self._send_wav(wav_bytes)
            elif self.path.startswith("/sing_mix"):
                data = self._read_json()
                source = data["source"]
                target = data["target"]
                diffusion_steps = int(data.get("diffusion_steps", 25))
                length_adjust = float(data.get("length_adjust", 1.0))
                cfg_rate = float(data.get("inference_cfg_rate", 0.7))
                t0 = time.time()
                _set_progress("准备", 1, "接收歌曲...")
                wav_bytes, sr = convert_and_mix(source, target, diffusion_steps, length_adjust, cfg_rate)
                print(f"[sing] /sing_mix 完成 {time.time()-t0:.1f}s (demucs+SVC+mix)", flush=True)
                self._send_wav(wav_bytes)
            else:
                self.send_error(404)
        except Exception as e:
            self._send_error(e)


if __name__ == "__main__":
    port = int(os.environ.get("SING_PORT", 9882))
    server = ThreadingHTTPServer(("127.0.0.1", port), SingHandler)
    print(f"[sing] Seed-VC 服务监听 http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
