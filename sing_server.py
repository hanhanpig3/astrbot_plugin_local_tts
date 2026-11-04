"""Seed-VC 音色转换常驻 HTTP 服务 (VC 模式, f0-condition=False)。

用于 AstrBot 本地 TTS 插件的「唱歌」功能:
  1. 用 Seed-VC VC 模型 (DiT_seed_v2_uvit_whisper_small_wavenet) 做零样本音色转换
  2. source = 唱歌干声, target = 参考音色 (如小鲸鱼说话参考音)
  3. POST /sing {source, target} -> wav bytes

依赖环境变量:
  SEEDVC_DIR   : seed-vc 仓库目录
  VC_CHECKPOINT: VC 主权重 .pth 路径
  VC_CONFIG    : VC config .yml 路径
  SING_PORT    : 服务端口 (默认 9882)
"""

import os
import sys
import json
import time
import threading
import argparse
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
        checkpoint=os.environ["VC_CHECKPOINT"],
        config=os.environ["VC_CONFIG"],
        fp16=True,
        f0_condition=False,
        diffusion_steps=int(os.environ.get("VC_DIFFUSION_STEPS", 25)),
        length_adjust=float(os.environ.get("VC_LENGTH_ADJUST", 1.0)),
        inference_cfg_rate=float(os.environ.get("VC_CFG_RATE", 0.7)),
    )
    print("[sing] 加载 Seed-VC VC 模型...", flush=True)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_args = seedvc.load_models(args)
    print(f"[sing] 模型加载完成 sr={mel_args['sampling_rate']} device={seedvc.device}", flush=True)
    return model, semantic_fn, vocoder_fn, campplus_model, mel_fn, mel_args["sampling_rate"]


MODEL, SEMANTIC, VOCODER, CAMPPLUS, MELFN, SR = _load_model()
DEVICE = seedvc.device
HOP = 256 if SR == 22050 else 512
MAX_CTX = SR // HOP * 30
OVERLAP_FRAME = 16
OVERLAP_WAVE = OVERLAP_FRAME * HOP
_convert_lock = threading.Lock()


@torch.no_grad()
def convert(source, target, diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7):
    """Seed-VC VC 转换: source 干声 -> target 参考音色。返回 (wave float32 numpy, sr)。"""
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

        cond, _, _, _, _ = MODEL.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)
        prompt_condition, _, _, _, _ = MODEL.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)

        max_source_window = MAX_CTX - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        while processed_frames < cond.size(1):
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
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/sing"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                source = data["source"]
                target = data["target"]
                diffusion_steps = int(data.get("diffusion_steps", 25))
                length_adjust = float(data.get("length_adjust", 1.0))
                cfg_rate = float(data.get("inference_cfg_rate", 0.7))
                t0 = time.time()
                wave, sr = convert(source, target, diffusion_steps, length_adjust, cfg_rate)
                # 写 wav 到内存
                import io
                buf = io.BytesIO()
                torchaudio.save(buf, torch.from_numpy(wave).unsqueeze(0), sr, format="wav")
                wav_bytes = buf.getvalue()
                print(f"[sing] 转换完成 {time.time()-t0:.1f}s {len(wave)/sr:.1f}s audio", flush=True)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                body = json.dumps({"error": str(e)}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        else:
            self.send_error(404)


if __name__ == "__main__":
    port = int(os.environ.get("SING_PORT", 9882))
    server = ThreadingHTTPServer(("127.0.0.1", port), SingHandler)
    print(f"[sing] Seed-VC 服务监听 http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
