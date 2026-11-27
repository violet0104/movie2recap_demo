import logging
import os
import time

from . import model
from .media import check_media_tools
from .pipeline import kokoro, process
from .store import PROJECTS, append_run_log, lock, read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def preflight():
    errors = []
    media = check_media_tools()
    errors.extend(media["errors"])
    for warning in media["warnings"]:
        logging.warning("preflight: %s", warning)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        errors.append("未设置 DEEPSEEK_API_KEY 环境变量")
    vllm = model.check_model_server(timeout=5.0)
    if not vllm["ok"]:
        errors.append(f"模型服务不可达 ({vllm['url']}): {vllm['error']}")
    elif not vllm["target_present"]:
        logging.warning("preflight: 模型服务未列出目标模型 %s (当前: %s)", model.MODEL, vllm["models"])
    try:
        kokoro()
    except Exception as error:
        errors.append(f"Kokoro TTS 初始化失败: {error}")
    if errors:
        raise RuntimeError("任务进程启动自检失败:\n- " + "\n- ".join(errors))
    logging.info("preflight checks passed")


def run():
    for path in sorted(PROJECTS.glob("*/job.json"), key=lambda item: item.stat().st_mtime):
        project = path.parent
        with lock(project / ".lock"):
            job = read_json(path)
            if job["status"] not in {"queued", "running"}:
                continue
            job["status"] = "running"
            write_json(path, job)
        kind = job["task"]["kind"]
        logging.info("%s %s", project.name, kind)
        append_run_log(project, f"task {kind} started")
        started = time.monotonic()
        try:
            process(project, job)
        except Exception as error:
            logging.exception("%s", project.name)
            job.update(status="error", error=str(error))
            append_run_log(project, f"task {kind} failed ({time.monotonic() - started:.1f}s): {error}")
        else:
            job.update(status="ready", stage="ready", last_task=kind, task=None, error=None)
            append_run_log(project, f"task {kind} finished ({time.monotonic() - started:.1f}s)")
        with lock(project / ".lock"):
            write_json(path, job)
        return True
    return False


def main():
    PROJECTS.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RECAP_SKIP_PREFLIGHT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        try:
            preflight()
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    try:
        with lock(PROJECTS / ".worker.lock", wait=False):
            while True:
                if not run():
                    time.sleep(1)
    except BlockingIOError:
        raise SystemExit("已有任务进程在运行")


if __name__ == "__main__":
    main()
