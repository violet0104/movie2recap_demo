from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .media import preview_clip, read_subtitles
from .store import DIST, PROJECTS, enqueue, lock, movie_path, read_json, version_path, write_json

app = FastAPI(title="Movie to Recap")
CHUNK_SIZE = 8 * 1024 * 1024


class Generate(BaseModel):
    target_minutes: int = Field(default=15, ge=10, le=20)
    subtitle_offset: float = Field(default=0, allow_inf_nan=False)


class Edit(BaseModel):
    version: int
    segment_id: str
    text: str
    candidate_id: str
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)


class Export(BaseModel):
    version: int


def project_path(project_id: UUID):
    project = PROJECTS / project_id.hex
    if not (project / "job.json").is_file():
        raise HTTPException(404, "项目不存在")
    return project


def editable_job(project):
    job = read_json(project / "job.json")
    if job["status"] in {"queued", "running"}:
        raise HTTPException(409, "项目正在处理")
    return job


def upload_job(project):
    job = editable_job(project)
    if job["status"] != "uploading":
        raise HTTPException(409, "项目已开始生成")
    return job


def current_upload(project, job):
    upload = job["movie"]
    if upload and not upload["complete"] and movie_path(project, upload).is_file():
        # The final rename may have completed before the metadata was written.
        upload.update(offset=upload["total"], complete=True)
        write_json(project / "job.json", job)
    return upload


@app.post("/api/projects", status_code=201)
def create_project():
    project_id = uuid4().hex
    project = PROJECTS / project_id
    for directory in ("source", "analysis", "audio", "exports", "timelines"):
        (project / directory).mkdir(parents=True)
    write_json(project / "job.json", {
        "id": project_id, "status": "uploading", "stage": "uploading",
        "movie": None, "subtitles": None, "source_time": 0, "event_count": 0,
        "duration": None, "timeline_version": None, "asset_version": None, "error": None, "task": None,
    })
    return {"id": project_id}


@app.get("/api/projects/{project_id}/upload")
def get_upload(project_id: UUID):
    project = project_path(project_id)
    with lock(project / ".lock"):
        job = read_json(project / "job.json")
        return current_upload(project, job) or {"offset": 0, "complete": False}


def commit_chunk(project, name, offset, total, chunk):
    with lock(project / ".lock"):
        job = upload_job(project)
        upload = current_upload(project, job)
        if not upload:
            if offset != 0:
                raise HTTPException(409, "请从字节 0 开始上传")
            upload = job["movie"] = {"name": name, "offset": 0, "total": total, "complete": False}
            write_json(project / "job.json", job)
        if (name, total) != (upload["name"], upload["total"]):
            raise HTTPException(409, "请选择此项目开始上传时的原片")
        if offset != upload["offset"] or upload["complete"]:
            raise HTTPException(409, f"服务器已接收 {upload['offset']} 字节")
        partial = project / "source" / "movie.part"
        with partial.open("r+b" if partial.exists() else "wb") as file:
            file.seek(offset)
            file.write(chunk)
            file.truncate()
        upload["offset"] += len(chunk)
        if upload["offset"] == total:
            partial.replace(movie_path(project, upload))
            upload["complete"] = True
        write_json(project / "job.json", job)
        return upload


@app.put("/api/projects/{project_id}/movie")
async def put_movie(project_id: UUID, name: str, offset: int, total: int, request: Request):
    project = project_path(project_id)
    name = Path(name).name
    if Path(name).suffix.lower() not in {".mp4", ".mkv"}:
        raise HTTPException(422, "原片须为 MP4 或 MKV")
    chunk = bytearray()
    async for data in request.stream():
        chunk.extend(data)
        if len(chunk) > CHUNK_SIZE:
            raise HTTPException(413, "分块大小不能超过 8 MiB")
    if not chunk or offset < 0 or offset + len(chunk) > total:
        raise HTTPException(422, "分块偏移或长度无效")
    return await run_in_threadpool(commit_chunk, project, name, offset, total, chunk)


def save_subtitles(project, name, content):
    try:
        read_subtitles(content)
    except ValueError as error:
        raise HTTPException(422, "字幕须为 UTF-8 编码的有效 SRT") from error
    with lock(project / ".lock"):
        job = upload_job(project)
        temporary = project / "source" / "dialogue.tmp"
        temporary.write_bytes(content)
        temporary.replace(project / "source" / "dialogue.srt")
        job["subtitles"] = Path(name).name
        write_json(project / "job.json", job)
    return {"name": job["subtitles"]}


@app.put("/api/projects/{project_id}/subtitles")
async def put_subtitles(project_id: UUID, name: str, request: Request):
    if Path(name).suffix.lower() != ".srt":
        raise HTTPException(422, "字幕须为 SRT")
    return await run_in_threadpool(save_subtitles, project_path(project_id), name, await request.body())


@app.post("/api/projects/{project_id}/generate", status_code=202)
def generate(project_id: UUID, settings: Generate):
    project = project_path(project_id)
    with lock(project / ".lock"):
        job = upload_job(project)
        upload = current_upload(project, job)
        if not upload or not upload["complete"] or not job["subtitles"]:
            raise HTTPException(409, "请上传原片和字幕")
        job["settings"] = settings.model_dump()
        enqueue(project, job, {"kind": "generate"})
    return job


@app.post("/api/projects/{project_id}/retry", status_code=202)
def retry(project_id: UUID):
    project = project_path(project_id)
    with lock(project / ".lock"):
        job = editable_job(project)
        if job["status"] != "error":
            raise HTTPException(409, "当前没有失败的任务")
        task = job["task"]
        if task["kind"] == "edit":
            raise HTTPException(409, "请在时间线中调整后重新保存")
        enqueue(project, job, task)
    return job


@app.get("/api/projects/{project_id}")
def get_project(project_id: UUID):
    return read_json(project_path(project_id) / "job.json")


def current_timeline(project):
    path = project / "timeline.json"
    if not path.is_file():
        raise HTTPException(404, "时间线尚未生成")
    return read_json(path)


@app.get("/api/projects/{project_id}/timeline")
def get_timeline(project_id: UUID):
    return current_timeline(project_path(project_id))


@app.patch("/api/projects/{project_id}/timeline", status_code=202)
def edit_timeline(project_id: UUID, edit: Edit):
    project = project_path(project_id)
    with lock(project / ".lock"):
        job = editable_job(project)
        timeline = current_timeline(project)
        if edit.version != timeline["version"]:
            raise HTTPException(409, "时间线已更新，请重新加载")
        segment = next((item for item in timeline["segments"] if item["id"] == edit.segment_id), None)
        if segment is None:
            raise HTTPException(404, "旁白段落不存在")
        if not any(shot["id"] == edit.candidate_id for shot in segment["candidates"]):
            raise HTTPException(404, "候选镜头不存在")
        if not edit.text.strip() or not 0 <= edit.start < edit.end <= job["duration"]:
            raise HTTPException(422, "请填写旁白并检查原片入点和出点")
        enqueue(project, job, {"kind": "edit", **edit.model_dump()})
    return job


@app.post("/api/projects/{project_id}/exports", status_code=202)
def export(project_id: UUID, selection: Export):
    project = project_path(project_id)
    with lock(project / ".lock"):
        job = editable_job(project)
        if not version_path(project, selection.version).is_file():
            raise HTTPException(404, "时间线版本不存在")
        enqueue(project, job, {"kind": "export", "version": selection.version})
    return job


@app.get("/api/projects/{project_id}/clips/{clip_id}/preview")
def get_clip_preview(project_id: UUID, clip_id: str, start: float | None = None, end: float | None = None):
    project = project_path(project_id)
    timeline = current_timeline(project)
    for segment in timeline["segments"]:
        clip = next((item for item in segment["candidates"] if item["id"] == clip_id), None)
        if clip:
            break
    else:
        raise HTTPException(404, "候选镜头不存在")
    job = read_json(project / "job.json")
    if start is None and end is None:
        start, end = clip["start"], clip["end"]
    if start is None or end is None or not 0 <= start < end <= job["duration"]:
        raise HTTPException(422, "原片入点或出点无效")
    directory = project / "exports" / "clips"
    directory.mkdir(exist_ok=True)
    path = directory / f"{clip_id}-{start}-{end}.mp4"
    if not path.is_file():
        with TemporaryDirectory(dir=directory) as scratch:
            temp = Path(scratch) / "clip.mp4"
            preview_clip(movie_path(project, job["movie"]), start, end, temp)
            temp.replace(path)
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/projects/{project_id}/media/{asset_id}")
def get_media(project_id: UUID, asset_id: str, download: bool = False):
    project = project_path(project_id)
    version, _, kind = asset_id.partition(".")
    media = {"mp4": "video/mp4", "srt": "application/x-subrip"}
    if not (version.isascii() and version.isdigit()) or kind not in media:
        raise HTTPException(404, "产物不存在")
    path = project / "exports" / version / f"recap.{kind}"
    if not path.is_file():
        raise HTTPException(404, "产物尚未生成")
    filename = f"{project_id.hex[:8]}-recap-v{version}.{kind}" if download else None
    return FileResponse(path, media_type=media[kind], filename=filename)


app.mount("/", StaticFiles(directory=DIST, html=True, check_dir=False))
