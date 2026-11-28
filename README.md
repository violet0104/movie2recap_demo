# Movie2Recap

在 Linux GPU 服务器上运行的单用户电影解说 Demo。上传 MP4 或 MKV 原片（可选上传 UTF-8 SRT 对白字幕），生成带英文旁白和烧录字幕的 720p MP4，并下载独立 SRT。自动选段的总时长上限为原片的 8%。技术方案见 [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)。

## 安装

准备 uv、Python 3.11、Node.js 22.12+、ffprobe，以及带 `subtitles`、`libx264` 和 AAC 编码器的 FFmpeg。应用依赖和 spaCy 英文模型由 `uv.lock` 固定。Kokoro 在 CPU 上使用 `af_heart`，支持通过 Misaki 内置的 eSpeak NG 回退发音。

在可联网的节点上，从共享项目目录安装依赖并下载模型。Slurm 集群可在登录节点执行这一步。

```bash
uv sync --locked --python 3.11
npm ci --prefix frontend
npm run build --prefix frontend
.venv/bin/hf download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/af_heart.pt
```

模型服务使用独立环境。以下是 Linux x86-64、CUDA 12.9 的安装示例，所用 vLLM 和 PyTorch 分别为 0.31.0 和 2.13.0。按 GPU 驱动选用对应的 CUDA 构建。建议从 40 GB 显存的 GPU 开始运行这一配置。[vLLM GPU 安装](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/)

```bash
uv venv --python 3.11 .vllm
UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4 \
  uv pip install --python .vllm/bin/python --torch-backend=cu129 \
  'torch==2.13.0+cu129' \
  'vllm @ https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
.venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

如果 cuDNN 等大依赖下载超时，可直接重跑 `uv pip install` 这一步。uv 会复用已缓存的依赖，已下载的模型权重可直接使用。[uv 缓存文档](https://docs.astral.sh/uv/concepts/cache/)

## 启动

在同一台 GPU 服务器上从项目根目录分别启动三个进程。使用 Slurm 时，在分配到 GPU 的计算节点运行这些命令。

```bash
PATH="$PWD/.vllm/bin:$PATH" .vllm/bin/vllm serve models/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 8001 \
  --trust-request-mm-kwargs \
  --limit-mm-per-prompt '{"image":0,"video":1}' \
  --max-model-len 65536 --max-num-seqs 1
```

启动命令将 `.vllm/bin` 加入该进程的 `PATH`，供 FlashInfer 首次预热时调用环境里的 `ninja`。

```bash
.venv/bin/python -m uvicorn recap.api:app --host 127.0.0.1 --port 8080
```

模型服务就绪后启动任务进程：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m recap.worker
```

Web 服务和任务进程直接使用安装好的 `.venv`。任务进程从已下载的 Hugging Face 缓存加载 Kokoro。

在服务器浏览器打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)。如果浏览器在自己的电脑上，从该电脑的本地终端启动 SSH 转发。SSH 的最终目标是运行 Web 服务的节点。

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:127.0.0.1:8080 <用户>@<服务器>
```

如果经登录节点访问 Slurm 计算节点，可用 `-J` 指定登录节点。

```bash
ssh -N -o ExitOnForwardFailure=yes -J <用户>@<登录节点>:<SSH端口> \
  -L 127.0.0.1:18080:127.0.0.1:8080 <用户>@<GPU节点>
```

保持 SSH 终端打开，在本机浏览器打开 [http://127.0.0.1:18080](http://127.0.0.1:18080)。

| 环境变量 | 默认值 |
| --- | --- |
| `RECAP_DATA_DIR` | 项目根目录下的 `data` |
| `RECAP_FFMPEG` / `RECAP_FFPROBE` | `ffmpeg` / `ffprobe` |
| `RECAP_MODEL_URL` | `http://127.0.0.1:8001/v1`（用于画面描述、片段评分与旁白撰写） |
| `RECAP_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` |
| `RECAP_MODEL_MAX_RETRIES` | `3`，本地模型服务请求失败重试次数 |
| `RECAP_SCORE_BATCH_SIZE` | `40`，每次向本地模型请求评分的片段数（保留全片上下文） |
| `RECAP_SCORE_CONTEXT_CHARS` | `24000`，评分请求附带的全片上下文最大字符数（超限时自动折半降级） |
| `RECAP_CLEANUP_FRAMES` | `0`，设为 `1` 时在全片描述完成后自动清理 `analysis/frames/` |
| `RECAP_SKIP_PREFLIGHT` | `0`，设为 `1` 时跳过任务进程启动自检 |

可通过 `GET /api/health` 查看 Web、Worker 与模型服务的连通状态，通过 `DELETE /api/projects/{id}` 清理闲置项目数据；各阶段耗时和错误摘要会记录在项目的 `analysis/run.log`。

Web 服务和任务进程使用同一数据目录。媒体工具路径可设置为可执行文件的绝对路径。迁移时携带源码、锁文件、`frontend/dist/`、项目数据、模型权重和 Hugging Face 缓存，然后在目标机重建环境。

在 Rocky Linux 8 等内置 eSpeak NG 库无法加载的环境中，启动任务进程前，将 `PHONEMIZER_ESPEAK_LIBRARY` 和 `PHONEMIZER_ESPEAK_DATA_PATH` 分别设为本机兼容的 `libespeak-ng.so` 和 `espeak-ng-data` 的绝对路径。可将 eSpeak NG 构建并安装到账号的可写目录。[eSpeak NG 构建文档](https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md)

## 使用

1. 上传原片，可选上传 UTF-8 SRT 对白字幕。上传中断后，在原项目重新选择同一原片即可续传。
2. 在项目页查看分析阶段、原片进度和已识别事件数。保留项目 URL 可在刷新后继续查看。任务进程重启后，从下一批帧恢复描述。项目的 `analysis/` 目录保存 `caption.json`、`caption_score.json` 和 `selected_segments.json`。
3. 播放成片，选择段落编辑英文旁白、候选镜头与入点和出点。保存修改会更新该段配音及后续时间码。点击「生成当前预览和导出」渲染新版本，MP4 和 SRT 可独立下载。
