# astrbot_plugin_local_tts

**AstrBot 本地 GPT-SoVITS 调音 TTS 插件** —— 随 AstrBot 一起启停本地语音合成服务，支持**语速/音调/音量调音**与**参考音频音色克隆**。

![平台](https://img.shields.io/badge/平台-macOS%20Apple%20Silicon-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.5.7-green)

---

## 一、来源与开发原因

### 来源（上游是什么）
- **语音合成基座**：[GPT-SoVITS v2](https://github.com/RVC-Boss/GPT-SoVITS)（RVC-Boss 开源项目），一个"少量样本音色克隆"（few-shot voice cloning）的 TTS 模型，由 4 个组件构成：
  - **GPT 语义模型** `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`
  - **SoVITS 声码器** `s2G2333k.pth`
  - **中文 HuBERT** `chinese-hubert-base`（提取参考音色）
  - **中文 RoBERTa** `chinese-roberta-wwm-ext-large`（理解文本语义）
- **AstrBot 对接**：本插件让 AstrBot WebUI 内置的 **「GSVI TTS(API)」语音合成源**直接使用上面的本地服务（`/infer_single` 协议）。

### 开发原因
1. **AstrBot 默认 TTS 走云端**（如 Edge-TTS / 云厂商），音色固定、不可定制、且受网络限制。
2. 用户希望拥有**专属音色**：先调音（语速/音调/音量），再用调好的音频做**音色克隆**，让机器人用自己独有的声音说话。
3. 需要**跟随 AstrBot 一起启停**，避免手动起两个后台服务、以及忘记关闭导致的资源占用。
4. 因此把本地 GPT-SoVITS 服务封装成符合 AstrBot 插件规范的插件，随 AstrBot 生命周期自动管理。

---

## 二、开发环境

| 项 | 值 |
|----|----|
| 操作系统 | **macOS 26.5.2**（Apple M4 / 16GB / ARM64） |
| 硬件加速 | **MPS（Metal）**，非 NVIDIA GPU |
| Python | 3.10.20（`.venv` 虚拟环境，uv 创建） |
| 依赖 | torch 2.14.1 / torchaudio 2.11.1（MPS 可用）、fastapi、uvicorn、librosa、soundfile、httpx、ffmpeg |
| 模型来源 | HuggingFace 镜像（`hf-mirror.com`），代码经 `gh-proxy.com` 下载 |
| AstrBot | **v4.29.4**（Star 插件 API，`astrbot_version >= 4.5.7`） |

> 本插件**只在 macOS Apple Silicon 上验证过**（引擎以 `-d mps` 运行）。其它平台需改为 CPU/CUDA 推理。

### 性能参考（M4 / MPS）
- 稳态合成延迟：**0.8 ~ 1.2 秒**（CPU 为 45 秒/句，MPS 提速约 40 倍）
- 首次调用（MPS 编译）：约 5~12 秒
- 输出：ogg/Opus 单句约 14~20KB；wav 约 150~240KB

---

## 三、安装

### 3.1 前置：本地部署 GPT-SoVITS 引擎

本插件**不打包** GPT-SoVITS 模型（体积大）与你的**私有参考音样本**（音色隐私）。请先在 `service_dir` 目录准备好引擎：

```
service_dir/
├── .venv/                      # Python 3.10 虚拟环境(含 torch MPS, librosa, soundfile, httpx, fastapi)
├── GPT-SoVITS/                 # 官方源码
│   ├── api.py                  # 推理引擎(FastAPI)
│   └── GPT_SoVITS/pretrained_models/
│       ├── gsv-v2final-pretrained/
│       │   ├── s2G2333k.pth
│       │   └── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
│       ├── chinese-hubert-base/
│       └── chinese-roberta-wwm-ext-large/
└── tts_ref.env                 # 默认参考音配置(可选)
```

`tts_ref.env` 示例（指定克隆音色，`TTS_REF_TEXT` 必须与音频内容一字不差）：
```bash
export TTS_REF_PATH="/绝对路径/你的参考音.wav"
export TTS_REF_TEXT="参考音频里说的那句话"
export TTS_REF_LANG="zh"
```

### 3.2 安装插件
把本插件目录放到 AstrBot 插件目录：
```bash
cp -r astrbot_plugin_local_tts ~/data/plugins/
```
或在 AstrBot WebUI 的插件市场/管理页安装。

### 3.3 配置
在 AstrBot WebUI 的插件配置里设置：
- `service_dir`：上述 GPT-SoVITS 部署目录的**绝对路径**
- `engine_port` / `tts_port`：默认 9880 / 9881
- `default_speed` / `default_pitch` / `default_volume`：默认调音参数（如 `0.85`/`0`/`0`）
- `auto_start`：默认开，AstrBot 启动即拉起服务

---

## 四、使用

### 4.1 插件随 AstrBot 启停
插件 `initialize()` 时自动启动引擎+包装服务；`terminate()`（禁用/重载/关闭 AstrBot）时自动停止。无需手动起服务。

命令：
- `/tts_status`：查看服务状态、TTS 接口地址、AstrBot 应填的 `api_base`
- `/tts_restart`（管理员）：重启 TTS 服务

### 4.2 在 AstrBot WebUI 配置语音合成
AstrBot 内置 **「GSVI TTS(API)」** 语音合成源，本插件完全兼容其协议：

1. 打开 AstrBot WebUI → 语音合成（TTS）配置
2. 选择/新增 **`GSVI TTS(API)`**
3. 把 **`api_base`** 填成插件显示的地址（默认 `http://127.0.0.1:9881`）
4. 启用该 TTS 源

之后 AstrBot 每次要说话时，会 `POST {api_base}/infer_single` 并下载返回的音频，走的是**本地调好音色的声音**。

### 4.3 直接调用（非 AstrBot 也可用）

```bash
# 调音合成
curl -X POST http://127.0.0.1:9881/tts -H "Content-Type: application/json" \
  -d '{"text":"你好","speed":0.85,"pitch":0,"volume":0,"output_format":"ogg"}' -o a.ogg

# 音色克隆(指定参考)
curl -X POST http://127.0.0.1:9881/tts -H "Content-Type: application/json" \
  -d '{"text":"要说的话","refer_wav_path":"/绝对路径/参考.wav","prompt_text":"参考音频内容","output_format":"ogg"}' -o b.ogg
```

### 4.4 调音与克隆说明
| 参数 | 说明 |
|------|------|
| `speed` | 语速 0.5~2.0（1.0 正常，0.85 略慢） |
| `pitch` | 音调，单位半音，-12~12 |
| `volume` | 音量，单位 dB，-20~20 |
| `refer_wav_path`+`prompt_text` | 音色克隆源；`prompt_text` 必须与参考音频内容**一字不差** |

---

## 五、隐私与安全

- **本仓库不含任何音频样本**：你的参考音/克隆音色是机器人独有资产，`*.wav / *.ogg / *.mp3`、`refs/`、`demos/`、`tts_ref.env` 均被 `.gitignore` 排除。
- 插件只提供**运行方法**（启停逻辑 + HTTP 协议），不包含私有声音。
- 服务绑定 `0.0.0.0`，局域网内其它设备也可调用；若不需要请将 `tts_port` 改绑 `127.0.0.1`（`TTS_BIND` 环境变量）。

---

## 六、常见问题

| 现象 | 解决 |
|------|------|
| AstrBot 报 TTS 请求失败 | 检查 `service_dir` 是否有效、模型是否已部署；`/tts_status` 看服务是否在运行 |
| 首次调用慢 | MPS 编译，5~12 秒后恢复正常 |
| 音色不像参考 | `prompt_text` 与参考音频内容不一致；参考音建议 3~10 秒、清晰、单声道 |
| 英文/中英混排报错 | 需安装 NLTK `cmudict` 与 `averaged_perceptron_tagger_eng`（部署时一次性） |

---

*MIT License · © 2026 hanhanpig3*
