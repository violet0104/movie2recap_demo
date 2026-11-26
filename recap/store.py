import json
import os
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = Path(os.environ.get("RECAP_DATA_DIR", ROOT / "data")) / "projects"
DIST = ROOT / "frontend" / "dist"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def lock(path: Path, *, wait=True):
    with path.open("a") as handle:
        flock(handle, LOCK_EX if wait else LOCK_EX | LOCK_NB)
        yield


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
