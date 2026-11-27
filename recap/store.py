import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = Path(os.environ.get("RECAP_DATA_DIR", ROOT / "data")) / "projects"
DIST = ROOT / "frontend" / "dist"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def lock(path: Path, *, wait=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flock(handle, LOCK_EX if wait else LOCK_EX | LOCK_NB)
        yield


def worker_alive() -> bool:
    lock_path = PROJECTS / ".worker.lock"
    if not lock_path.is_file():
        return False
    try:
        with lock(lock_path, wait=False):
            return False
    except BlockingIOError:
        return True


def append_run_log(project: Path, message: str):
    log_path = project / "analysis" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def movie_path(project, upload):
    return project / "source" / f"movie{Path(upload['name']).suffix.lower()}"


def enqueue(project, job, task):
    job.update(status="queued", stage="queued", error=None, stage_progress=None, task=task)
    write_json(project / "job.json", job)


def version_path(project, version):
    return project / "timelines" / f"{version}.json"


def publish_timeline(project, timeline):
    write_json(version_path(project, timeline["version"]), timeline)
    write_json(project / "timeline.json", timeline)
