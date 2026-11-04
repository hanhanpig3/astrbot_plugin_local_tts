# astrbot_plugin_local_tts

**AstrBot 本地 GPT-SoVITS 调音与音色克隆 TTS 插件** —— 随 AstrBot 自动启停本地语音合成后端，支持**语速/音调/音量实时调音**，并通过少量样本（few-shot）进行**专属音色克隆**，让你的机器人拥有独一无二的声音。

![平台](https://img.shields.io/badge/平台-macOS%20%7C%20Linux%20%7C%20Windows-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.5.7-green)
![License](https://img.shields.io/badge/license-MIT-purple)

---

## 🤖 小白配合 AI 辅助安装指南（非程序员看这里）

如果你不是程序员、没有代码或终端基础，**完全不用担心**！你可以直接把本篇 README 全选复制（`Cmd+A` / `Ctrl+A`），丢给任意大模型 AI（如 **DeepSeek / Claude / ChatGPT / 豆包**），并发送以下提示词：

> **“你好！我想把这个开源项目安装在我的电脑上，给我的 AstrBot 机器人配置专属声音。我的电脑系统是【macOS Apple Silicon / Windows 11 / Ubuntu Linux】，我没有任何编程基础。请仔细阅读这份 README，像一名手把手教学的耐心理工科导师一样，一次只给我一条清晰明确的命令或操作指示，等我做完并反馈给你之后，你再告诉我下一步怎么做。”**

AI 会根据你的操作系统，一步一步帮你建虚拟环境、拉取模型和配置插件！

---

## 一、核心原理与架构

```
用户发送消息 ──▶ AstrBot (QQ/微信等) 
                     │
                     ▼ 内置 GSVI TTS(API) 插件源
           POST /infer_single (带文本)
                     │
                     ▼
          ┌───────────────────────────────────┐
          │ 包装管理服务 (端口 9881)           │
          │ 负责: 语速/音调调音、格式转换、缓存   │
          └─────────────────┬─────────────────┘
                            │ 转发推理请求 (带默认参考音色)
                            ▼
          ┌───────────────────────────────────┐
          │ GPT-SoVITS 推理引擎 (端口 9880)    │
          │ v2 底模 + Few-Shot 音色克隆        │
          └───────────────────────────────────┘
```

- **全本地运行**：不花一分钱 API 费、保护隐私、不依赖任何云端语音接口。
- **随 AstrBot 启停**：启动 AstrBot 时后台自动拉起引擎，退出 AstrBot 时自动杀掉进程，不占后台内存。
- **跨平台兼容**：
  - macOS (Apple Silicon M1~M4)：使用 Metal (`engine_device: mps`) 硬件加速，单句延迟 0.8~1.2 秒。
  - Linux / Windows (NVIDIA 独显)：使用 CUDA (`engine_device: cuda`) 硬件加速。
  - 无独显设备 / AMD CPU：使用纯 CPU (`engine_device: cpu`) 推理。

---

## 二、第一步：如何获取音色与制作参考音频？

> 💡 **原理解释**：GPT-SoVITS 采用 Few-shot 克隆技术，只需一段 **3 ~ 10 秒** 的清晰人声音频（`.wav`）以及**音频里所说文字的原原本本文本**，就能高保真克隆这个人的音色与说话腔调。

### 推荐方案 A：零成本制作高品质音色（MiniMax 音色设计 + F12 抓包法）

如果你没有现成的录音，推荐使用大厂的免费音色设计工具定制：

1. **进入音色设计界面**：
   - 电脑浏览器访问 **MiniMax 开放平台** 或 **海螺 AI** 的免费“音色设计 / 声音定制”功能。
   - 输入你期望的音色提示词（例如：“温柔治愈的知性大姐姐”、“活泼开朗的小萝莉”、“磁性沉稳的少年音”等）。
2. **输入试听文本**：
   - 在生成试听文本框中，输入一句**自然、简短**的话（建议 8~15 个字），例如：
     > *“你好，我是小鲸鱼，很高兴认识你呀。”*
   - ⚠️ **切记**：将这句文本复制并保存下来！后续配置中的 `TTS_REF_TEXT` 必须与此**一字不差（标点都要一致）**。
3. **F12 浏览器抓取音频**：
   - 按键盘 `F12`（或右键网页任意处选择“检查”）打开开发者工具。
   - 切换到顶部的 **Network（网络）** 标签页。
   - 在筛选器（Filter）中选中 **Media**，或者在搜索框输入 `mp3` 或 `audio`。
   - 回到网页点击“生成试听”或“播放”。
   - 在网络请求列表中，会立即弹出一个音频文件请求（通常是以 `.mp3` 结尾）。
   - 右键该请求 ➔ 选择 **Open in new tab（在新标签页打开）** 或 **Copy URL**，将音频另存为本地文件，命名为 `sample.mp3`。
4. **转换为标准单声道 WAV 格式**：
   - GPT-SoVITS 推荐使用采样率为 16000Hz 或 22050Hz 的单声道 WAV 格式。
   - 终端一行转换命令（需安装 ffmpeg）：
     ```bash
     ffmpeg -i sample.mp3 -ar 22050 -ac 1 sample.wav
     ```
   - 或者使用任意在线音频转换网站将 `sample.mp3` 转换为 `sample.wav`。

### 方案 B：自行录制或使用现有音频

- 用手机录音机录制一段自己或朋友的声音，或者从喜欢的动漫、播客、影视剧里剪辑出一段 **无背景音乐、无混响、咬字清晰** 的干声（3~8 秒最佳）。
- 将说的那句话原原本本记录成文字。
- 同理导出为单声道 `.wav` 格式。

---

## 三、第二步：下载 GPT-SoVITS 模型与上游源码

你需要一个本地目录（本指南假设为 `~/localtts`，你可以自定义）。完整目录结构如下：

```
~/localtts/
├── .venv/                      # Python 3.10 虚拟环境
├── GPT-SoVITS/                 # GPT-SoVITS 官方源码
│   ├── api.py                  # 推理引擎入口
│   └── GPT_SoVITS/pretrained_models/
│       ├── gsv-v2final-pretrained/
│       │   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
│       │   └── s2G2333k.pth
│       ├── chinese-hubert-base/
│       │   ├── config.json
│       │   ├── preprocessor_config.json
│       │   └── pytorch_model.bin
│       └── chinese-roberta-wwm-ext-large/
│           ├── config.json
│           ├── tokenizer.json
│           └── pytorch_model.bin
└── tts_ref.env                 # 默认参考音色配置文件
```

### 1. 拉取 GPT-SoVITS 源码与创建虚拟环境

```bash
mkdir -p ~/localtts && cd ~/localtts

# 拉取 GPT-SoVITS 源码 (国内镜像加速)
git clone https://ghproxy.net/https://github.com/RVC-Boss/GPT-SoVITS.git GPT-SoVITS

# 创建 Python 3.10 虚拟环境 (推荐用 uv 或 conda)
python3.10 -m venv .venv
source .venv/bin/activate

# 安装推理依赖
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  # macOS/CPU环境
# 若是 NVIDIA 显卡环境请使用: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn librosa soundfile httpx numpy
```

### 2. 模型下载全渠道说明（模型去哪找？）

GPT-SoVITS v2 运行总共需要下载 **3 组预训练模型**（共约 1.1GB）：

| 模型组件 | 文件路径 | 官方下载链接 (HuggingFace) | 国内高速镜像 (免翻墙) |
|---------|---------|---------------------------|---------------------|
| **GPT 语义模型** | `GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` (155MB) | [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained) | [hf-mirror.com/lj1995/GPT-SoVITS](https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt) |
| **SoVITS 声码器** | `GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth` (106MB) | [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained) | [hf-mirror.com/lj1995/GPT-SoVITS](https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth) |
| **中文 HuBERT** | `GPT_SoVITS/pretrained_models/chinese-hubert-base/` (约 180MB) | [TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) | [hf-mirror.com/TencentGameMate/chinese-hubert-base](https://hf-mirror.com/TencentGameMate/chinese-hubert-base) |
| **中文 RoBERTa** | `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/` (约 620MB) | [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | [hf-mirror.com/hfl/chinese-roberta-wwm-ext-large](https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large) |

#### 快速下载脚本（终端一键运行）：

在 `~/localtts` 目录下执行下列命令，自动通过国内镜像完成所有模型下载：

```bash
cd ~/localtts/GPT-SoVITS/GPT_SoVITS/pretrained_models

# 1. 下载 gsv-v2final 核心底模
mkdir -p gsv-v2final-pretrained
curl -L -o gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
  https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
curl -L -o gsv-v2final-pretrained/s2G2333k.pth \
  https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth

# 2. 下载 chinese-hubert-base
mkdir -p chinese-hubert-base
curl -L -o chinese-hubert-base/config.json https://hf-mirror.com/TencentGameMate/chinese-hubert-base/resolve/main/config.json
curl -L -o chinese-hubert-base/preprocessor_config.json https://hf-mirror.com/TencentGameMate/chinese-hubert-base/resolve/main/preprocessor_config.json
curl -L -o chinese-hubert-base/pytorch_model.bin https://hf-mirror.com/TencentGameMate/chinese-hubert-base/resolve/main/pytorch_model.bin

# 3. 下载 chinese-roberta-wwm-ext-large
mkdir -p chinese-roberta-wwm-ext-large
curl -L -o chinese-roberta-wwm-ext-large/config.json https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/resolve/main/config.json
curl -L -o chinese-roberta-wwm-ext-large/tokenizer.json https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/resolve/main/tokenizer.json
curl -L -o chinese-roberta-wwm-ext-large/pytorch_model.bin https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin
```

### 3. 配置默认参考音色（`tts_ref.env`）

在 `~/localtts/tts_ref.env` 中写入你的参考音频路径和对应台词：

```bash
# ~/localtts/tts_ref.env
export TTS_REF_PATH="/Users/你的用户名/localtts/sample.wav"
export TTS_REF_TEXT="你好，我是小鲸鱼，很高兴认识你呀。"
export TTS_REF_LANG="zh"
```

---

## 四、第三步：安装与启用 AstrBot 插件

### 1. 将插件安装到 AstrBot

把本插件克隆或复制到 AstrBot 的 `plugins` 目录下：

```bash
# 假设你的 AstrBot 数据目录在 ~/data
git clone https://github.com/hanhanpig3/astrbot_plugin_local_tts.git ~/data/plugins/astrbot_plugin_local_tts
```

### 2. 插件参数配置

打开 AstrBot WebUI ➔ **插件管理** ➔ 找到 **`astrbot_plugin_local_tts`** ➔ 点击配置：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `service_dir` | `/Users/你的用户名/localtts` | 刚刚部署的引擎根目录（必须是**绝对路径**） |
| `engine_device` | `mps` / `cuda` / `cpu` | Mac 用 `mps`；N卡用 `cuda`；其余用 `cpu` |
| `engine_port` | `9880` | 引擎端口（保持默认） |
| `tts_port` | `9881` | 包装服务端口（保持默认） |
| `auto_start` | `true` | AstrBot 启动时自动拉起服务 |
| `default_speed` | `0.85` | 默认语速（1.0 是原速，0.85 稍显慢速更自然） |

保存配置后，**请完整重启 AstrBot 进程**（注意：Python 模块级更新需要完全重启进程生效，而不是单纯页面重载）。

### 3. 在 AstrBot WebUI 启用语音合成 (TTS)

1. 进入 AstrBot WebUI ➔ **设置** ➔ **语音合成（TTS）**（`provider_tts_settings`）。
2. 将 **启用 TTS（enable）** 开关打开。
3. 在 Provider 下拉菜单中选择 **`GSVI TTS(API)`**。
4. 在对应配置项中填写：
   - **`api_base`**：`http://127.0.0.1:9881`（必须指向包装服务的端口 9881）
   - `text_lang`：`中文`
   - `prompt_text_lang`：`中文`
   - `version`：`v4`
5. 启用该 Provider 并保存设置。

现在，在群聊或私聊中向机器人发送任意文字，机器人就会用你配置的克隆音色实时发语音啦！

---

## 唱歌模式（Seed-VC 音色转换）

文字 TTS（GPT-SoVITS）是「文本→说话」，没有旋律输入，**无法直接唱歌**。本项目用 **Seed-VC 零样本音色转换**补上唱歌能力：把一段「唱歌干声」的音色换成小鲸鱼（或任何参考音色），保留干声的旋律。

> 原理：输入「唱歌干声 + 参考音色」→ 输出「参考音色唱干声的旋律」。

### 1. 准备唱歌环境（一次即可）

需要**独立的 Python 3.10 环境**（不与 GPT-SoVITS 的 `.venv` 混用，避免 torch 版本冲突）：

```bash
mkdir -p ~/localtts/sing_lab && cd ~/localtts/sing_lab
python3.10 -m venv venv
./venv/bin/pip install "torch==2.4.0" "torchaudio==2.4.0"
./venv/bin/pip install "scipy==1.13.1" "librosa==0.10.2" "numpy==1.26.4" \
  "huggingface_hub>=0.28.1" "munch==4.0.0" "einops==0.8.0" \
  "descript-audio-codec==1.0.0" "pydub==0.25.1" "resemblyzer" "jiwer==3.0.3" \
  "transformers==4.46.3" "soundfile==0.12.1" "hydra-core==1.3.2" "pyyaml" \
  "python-dotenv" "accelerate"
```

下载 Seed-VC 源码 + 权重（国内走 hf-mirror.com）：

```bash
cd ~/localtts/sing_lab
# Seed-VC 源码 (GitHub 被墙用 curl 下载 zip; 也可用代理 git clone)
curl -L -o seed-vc.zip "https://github.com/Plachtaa/seed-vc/archive/refs/heads/main.zip"
unzip -q seed-vc.zip && mv seed-vc-main seed-vc

# SVC 主权重 (f0=True 音调保留) + config (hf-mirror)
cd seed-vc
mkdir -p checkpoints models/bigvgan_44k models/whisper-small
curl -L -o checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth \
  "https://hf-mirror.com/Plachta/Seed-VC/resolve/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
# bigvgan 44k 声码器 + whisper-small (hf-mirror)
curl -L -o models/bigvgan_44k/config.json "https://hf-mirror.com/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/config.json"
curl -L -o models/bigvgan_44k/bigvgan_generator.pt "https://hf-mirror.com/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt"
curl -L -o models/whisper-small/config.json "https://hf-mirror.com/openai/whisper-small/resolve/main/config.json"
curl -L -o models/whisper-small/model.safetensors "https://hf-mirror.com/openai/whisper-small/resolve/main/model.safetensors"
curl -L -o models/whisper-small/preprocessor_config.json "https://hf-mirror.com/openai/whisper-small/resolve/main/preprocessor_config.json"
```

> 本插件配套的 `seed-vc/inference.py` 与 `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` 已改为**加载本地 `checkpoints/models/` 目录**（绕过 huggingface_hub 直连 HF 被墙）。下载后按上述路径放置即可。

### 2. 在 AstrBot 插件配置里启用唱歌

| 配置项 | 说明 |
| --- | --- |
| `sing_enable` | `true`（默认关闭） |
| `sing_dir` | 唱歌环境目录，如 `~/localtts/sing_lab` |
| `sing_checkpoint` | SVC 主权重 `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` |
| `sing_config` | SVC `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`（f0=True 音调保留） |
| `sing_ref_path` | 唱歌参考音色，默认小鲸鱼参考音 `demos/小鲸鱼_默认参考_0.85.wav` |
| `sing_diffusion_steps` | 25（标准），30~50 更高质量但更慢 |

改完保存后**完整重启 AstrBot**（插件重载不重新加载 Python 模块）。

### 3. 使用唱歌

- `/sing <唱歌干声路径>`：用小鲸鱼唱本机**干声**（无伴奏人声，wav/mp3），音调=原歌、音色=小鲸鱼
- `/sing_mix <歌曲路径>`：用小鲸鱼唱**完整歌曲**（自动 demucs 分离人声+伴奏，唱完合成伴奏，输出完整版）
- `/sing` / `/sing_mix` + **发送一段音频附件**：用附件作输入
- `/sing_status`：查看唱歌服务状态
- `/sing_restart`：重启唱歌服务（首次模型加载约 1-2 分钟）
- 可选 `ref=<参考音路径>`：覆盖默认小鲸鱼参考音

唱歌干声建议：**无伴奏人声**（清唱或伴奏分离后的干声），wav，采样率 16k~48k 均可。`/sing_mix` 则直接给**带伴奏的完整歌曲**即可。

---

## 五、在 DSH (DeepSeek Harness) 中使用

如果你使用的宿主不是 AstrBot，而是 **DeepSeek Harness (DSH)**，无需更换底层服务：

- 本项目包含的 `astrbot_tts_server.py` 是一个通用的 FastAPI 调音包装层，对外直接提供标准 HTTP 接口：
  - `POST /tts`（通用合成、支持格式切换）
  - `POST /infer_single`（GSVI 兼容接口）
  - `GET /health`（健康检查）
- 你只需在本项目基础上，为 DSH 编写一个简单的插件接入模块（实现 DSH 对应的生命周期管理，启动 9880 和 9881 两个子进程），即可让 DSH 享受完全相同的音色克隆与调音能力。

---

## 六、常见问题与避坑指南 (FAQ)

### Q1: 机器人发出的语音报 `Input does not contain a known duration` 或提示音频为空？
- **原因**：这是由于引擎刚刚启动还在加载 1GB 模型权重（冷启动），或者 Metal/CUDA 正在即时编译（JIT）。
- **解决**：本插件最新版已增加 `_has_audio_data()` 校验和自动重试机制。通常等待 10~20 秒引擎完全暖机后即可恢复正常。

### Q2: 英文或中英混合句子合成报错（502 错误）？
- **原因**：GPT-SoVITS 推理英文需要 NLTK 的发音字典（`cmudict` 与 `averaged_perceptron_tagger_eng`）。
- **解决**：在你的 `.venv` 虚拟环境中运行：
  ```bash
  python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
  ```
  如果国内网络访问 GitHub 失败，可手动将这两个数据包下载解压到虚拟环境的 `nltk_data` 目录下。

### Q3: 为什么克隆出来的声音不像、或者吐字含糊？
- 参考音频尽量保证 **干声（无背景音、无混响、无回声）**。
- `tts_ref.env` 里的 `TTS_REF_TEXT` **必须与音频中的实际说话内容 100% 完全匹配**，哪怕多一个字或少一个标点都会导致音素对齐错误。
- 音频时长建议在 **3 到 8 秒** 之间，过短信息不足，过长模型容易注意力涣散。

---

## 许可证与致谢

- 遵循 **MIT License** 开源。
- 感谢 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 团队开源的高性能零样本 TTS 框架。
- 感谢 [AstrBot](https://github.com/Soulter/AstrBot) 提供的优良机器人框架与插件生态。
