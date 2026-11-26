# 电影解说 Demo 技术方案

用户在远程服务器的浏览器打开 Demo，上传一部电影和对应的 SRT 字幕。页面显示生成进度，并提供在线审片、修改和下载。系统为每部电影生成新的英文解说，自动选段的总时长上限为原片的 **8%**。交付物是带英文字幕的 **720p MP4** 和独立的 **SRT**。

`avatar-recap.mp4` 用作剪辑节奏和画面呈现的参考。旁白由 Kokoro 使用固定的预设英文音色生成。剧情分析模型和处理程序运行在同一台远程 Linux GPU 服务器上。Demo 采用单用户、单任务进程和项目目录存储。

## 1. 技术架构

| 组件 | 技术 | 职责 |
| --- | --- | --- |
| 网页 | React、TypeScript、Vite | 上传、查看进度、在线播放、逐段编辑和下载。 |
| Web 服务 | Python、FastAPI | 提供打包后的网页和 API，管理项目文件和任务状态，提供视频播放接口。 |
| 任务进程 | Python、FFmpeg、ffprobe、Kokoro | 按固定规则抽帧和组窗，调用模型服务并保存结果，生成旁白和成片。 |
| 模型服务 | Qwen3-VL-8B-Instruct、vLLM、DeepSeek | Qwen 描述画面并写旁白，DeepSeek 根据全片描述给每个片段评分。 |
| 存储 | 服务器磁盘、JSON | 保存上传素材、分析结果、时间线、配音和导出文件。 |
| 运行环境 | Linux、NVIDIA GPU、uv | 用普通用户权限安装 Python 环境，在服务器上启动 Web 服务、任务进程和模型服务。 |

浏览器和 API 使用同一个地址。模型服务在服务器回环地址监听，由任务进程调用。电影、字幕、抽帧和模型推理结果都保存在服务器上。输入准备模块按固定规则处理文件和时间码，任务进程使用 0-1 背包算法按评分和时长选片。

```text
服务器浏览器 ↔ FastAPI ↔ 项目目录
任务进程 ↔ 项目目录
任务进程 → vLLM / Qwen3-VL
任务进程 → DeepSeek API（片段描述和时间码）
任务进程 → Kokoro / FFmpeg
```

## 2. 上传和素材读取

首版接收 **MP4 或 MKV 原片**和 **SRT 字幕**。用户创建项目后，网页分别上传原片和字幕。服务端给项目分配 ID，用固定的内部文件名保存素材，并保留原文件名供网页显示。

- **大文件上传。** 浏览器分块发送电影。API 返回服务器已提交的字节数。连接中断后，浏览器从该偏移继续上传。服务端写完分块后推进偏移，全部上传完成后发布原片文件。
- **读取视频信息。** 用 `ffprobe` 读取原片时长、分辨率和轨道信息，供分窗和渲染使用。
- **解析字幕。** 从 SRT 读取字幕编号、起止时间和文本。用户可以设置全局字幕偏移。窗口匹配使用偏移后的字幕时间。

## 3. 分批生成事件描述

FFmpeg 按每秒一帧生成分析帧。Qwen3-VL 每次读取 300 帧和同一时间范围的 SRT 对白，片尾只发送剩余的帧。各批按原片顺序连续推进。模型按时间顺序描述该批内看到的事件，短事件用两句话、较长事件用三句话，人物可用外观或行为描述。

每批写入项目内的 `analysis/caption.json`。每个片段包含 `segment_id`、原片秒数 `start` 和 `end`、事件描述 `caption`。文件同时记录已处理的帧数，供重试时定位下一批。

## 4. 评分和选段

任务进程调用 `utils/score_caption.py` 读入全部片段描述，让 DeepSeek 参考全片内容为每段评分，并保存 `analysis/caption_score.json`。评分请求发送文本和时间码。任务进程通过 `DEEPSEEK_API_KEY` 认证。[DeepSeek Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)

`utils/select_segments_knapsack.py` 以原片时长的 8% 为预算，在预算内最大化所选片段的总分，并按起点排序，保存为 `analysis/selected_segments.json`。Qwen 结合选中片段的描述和对白撰写连续的英文旁白，每个片段对应一段旁白。网页保留原片入点和出点，供审看和编辑。

## 5. 配音、字幕和成片时间线

1. **生成英文旁白。** Kokoro 使用固定的 `af_heart` 音色连续合成每段英文旁白，保留段内的自然停顿。保存音频和实际时长。[Kokoro 文档](https://github.com/hexgrad/kokoro)
2. **计算字幕时间。** 按 Kokoro 返回的词时间戳切分显示字幕，按音频采样数累计成片时间。字幕换行处的语音连续播放。输入 SRT 提供原片对白，导出 SRT 由新配音的时间线生成。
3. **匹配画面长度。** 成片按原片顺序保留选中的片段。在每段区间内安排旁白，超出区间时请模型缩短旁白。
4. **保存时间线。** 每个段落记录英文文本、语音文件、事件编号、原片入点和出点，以及成片上的起止时间。原片 `01:07–01:13` 可以安排在成片 `04:12–04:18`，该句旁白和英文字幕使用成片的区间。
5. **渲染成片。** FFmpeg 从原片精确截取选中的片段，统一分辨率和帧率，按时间线拼接并加入旁白。保留原画幅比例，输出 1280 × 720 横屏 MP4。视频使用 H.264，音频使用 AAC，英文字幕用 `libass` 烧录。独立 SRT 从同一时间线导出。[FFmpeg 时间定位文档](https://ffmpeg.org/ffmpeg.html) · [FFmpeg 字幕滤镜文档](https://ffmpeg.org/ffmpeg-filters.html#subtitles-1)

## 6. 网页操作和在线播放

- **上传页**接收电影和 SRT，显示上传进度和字幕偏移。用户确认素材后启动生成。
- **进度页**显示当前阶段、已读取的原片位置和已提取事件数。网页定期读取任务状态。刷新页面后可继续查看同一项目。
- **项目页**使用浏览器原生视频播放器。已经生成的解说可以直接播放、暂停、拖动进度条和全屏。英文字幕随画面显示。播放器旁边显示各段旁白、原片时间和成片位置。
- **编辑**支持更换候选镜头、调整入点和出点、修改英文旁白。改稿后重新合成该段语音，并更新后续时间码。页面显示预览对应的时间线版本，用户可以从当前版本重新生成预览或导出。
- **播放接口**返回 `video/mp4`，支持 HTTP Range。有效的字节范围请求返回 `206 Partial Content` 和对应的 `Content-Range`，让浏览器可以拖到任意位置。MP4 使用 `faststart` 将索引放到文件前部。项目页提供 MP4 和 SRT 的独立下载按钮。[FFmpeg MP4 封装文档](https://ffmpeg.org/ffmpeg-formats.html)

### API 草案

| 接口 | 用途 |
| --- | --- |
| `POST /api/projects` | 创建项目，返回项目 ID。 |
| `GET /api/projects/{id}/upload` | 查询原片上传偏移。 |
| `PUT /api/projects/{id}/movie` | 上传一个分块，携带偏移和文件总长度。 |
| `PUT /api/projects/{id}/subtitles` | 上传 SRT。 |
| `POST /api/projects/{id}/generate` | 按字幕偏移和 8% 选段预算启动生成。 |
| `GET /api/projects/{id}` | 获取任务状态、进度和可播放产物。 |
| `GET /api/projects/{id}/timeline` | 读取英文稿、剪辑区间和成片时间。 |
| `PATCH /api/projects/{id}/timeline` | 保存段落修改并生成新的时间线版本。 |
| `GET /api/projects/{id}/clips/{clip_id}/preview` | 播放候选原片片段。 |
| `POST /api/projects/{id}/exports` | 根据指定版本生成预览或成片。 |
| `GET /api/projects/{id}/media/{asset_id}` | 播放或下载已完成的项目产物。 |

## 7. 项目文件和恢复机制

```text
data/projects/<project_id>/
  job.json
  source/
  analysis/
  audio/
  timeline.json
  exports/
```

`job.json` 记录上传信息、当前步骤、已读取的原片位置、已提取的事件数和最近的错误。`analysis/` 保存分段描述、评分、选片清单和解说稿。`timeline.json` 保存当前时间线及其版本。`source/`、`audio/` 和 `exports/` 分别保存用户素材、分段旁白和播放产物。

任务进程依次完成素材读取、分段描述、全片评分、选段、写稿、配音和渲染。JSON 先写同目录的临时文件，再原子替换。分析中断后，从已保存的帧数继续读取下一批。评分完成后，重试复用已保存的评分。重新编辑只更新受影响的旁白段落和时间线。渲染使用指定版本的快照，完成后发布最终文件和播放地址。

## 8. 在远程服务器用 uv 部署

### 环境准备

将应用、工具、模型权重和项目数据放在用户可写的目录。服务器提供当前账号可用的 NVIDIA GPU、ffprobe，以及带 `subtitles` 滤镜和 H.264、AAC 编码器的 FFmpeg。eSpeak NG 提供 Kokoro 的生词发音回退。应用通过配置指定媒体工具路径、模型地址和项目数据目录。[Kokoro 安装与配音示例](https://github.com/hexgrad/kokoro)

### 环境安装和服务启动

1. **应用环境。** FastAPI、媒体处理依赖和 Kokoro 写入应用的 `pyproject.toml`。uv 使用 `uv.lock` 固定依赖，部署时执行 `uv sync --locked`。[uv 锁定与同步文档](https://docs.astral.sh/uv/concepts/projects/sync/)
2. **模型环境。** 用 uv 建立独立的 vLLM 环境，固定与目标服务器 GPU 驱动兼容的 vLLM、PyTorch 版本和下载源。用本地权重启动 Qwen3-VL，模型服务监听 `127.0.0.1:8001`。[vLLM GPU 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [vLLM 服务参数](https://docs.vllm.ai/en/latest/cli/serve/)
3. **前端产物。** 用 Vite 构建 `dist/`，放到 FastAPI 静态文件目录。FastAPI 监听 `127.0.0.1:8080`。模型服务就绪后启动任务进程。
4. **打开 Demo。** 在服务器上的浏览器打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)，上传素材并在项目页直接播放解说视频。

迁移到另一台服务器时，复制源码、uv 锁文件、前端产物、模型权重、配置和项目数据目录。在目标服务器重建 uv 环境，再按同一顺序启动服务。
