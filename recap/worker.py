import logging
import time

from .pipeline import process
from .store import PROJECTS, lock, read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run():
    for path in sorted(PROJECTS.glob("*/job.json"), key=lambda item: item.stat().st_mtime):
        project = path.parent
        with lock(project / ".lock"):
            job = read_json(path)
            if job["status"] not in {"queued", "running"}:
                continue
            job["status"] = "running"
            write_json(path, job)
        logging.info("%s %s", project.name, job["task"]["kind"])
        try:
            process(project, job)
        except Exception as error:
            logging.exception("%s", project.name)
            job.update(status="error", error=str(error))
        else:
            job.update(status="ready", stage="ready", last_task=job["task"]["kind"], task=None, error=None)
        write_json(path, job)
        return True
    return False


def main():
    PROJECTS.mkdir(parents=True, exist_ok=True)
    try:
        with lock(PROJECTS / ".worker.lock", wait=False):
            while True:
                if not run():
                    time.sleep(1)
    except BlockingIOError:
        raise SystemExit("已有任务进程在运行")


if __name__ == "__main__":
    main()
