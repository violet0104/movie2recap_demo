import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from recap import api, media, model, pipeline, store, worker
from utils import score_caption, select_segments_knapsack


def _make_response(status_code: int, payload: dict | str):
    request = httpx.Request("POST", "http://test/chat/completions")
    if isinstance(payload, str):
        return httpx.Response(status_code, text=payload, request=request)
    return httpx.Response(status_code, json=payload, request=request)


def test_knapsack_cli_main_runs_without_name_error(tmp_path):
    input_path = tmp_path / "caption_score.json"
    output_path = tmp_path / "selected.json"
    input_path.write_text(json.dumps({
        "video_path": "movie.mp4",
        "duration": 100.0,
        "segments": [
            {"segment_id": 0, "start": 0.0, "end": 5.0, "caption": "Intro", "score": 0.2},
            {"segment_id": 1, "start": 10.0, "end": 16.0, "caption": "Climax", "score": 0.95},
            {"segment_id": 2, "start": 20.0, "end": 22.0, "caption": "Ending", "score": 0.7},
        ],
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(Path(select_segments_knapsack.__file__)),
         "--input", str(input_path), "--output", str(output_path)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Selected 2 segments" in proc.stdout
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [s["segment_id"] for s in saved["selected_segments"]] == [1, 2]


def test_score_caption_uses_qwen_env_retries_fence_and_clamping(monkeypatch):
    monkeypatch.setenv("RECAP_MODEL_URL", "http://127.0.0.1:9001/v1/")
    monkeypatch.setenv("RECAP_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    calls = []
    responses = [
        _make_response(429, "rate limited"),
        _make_response(200, {"choices": [{"finish_reason": "stop", "message": {"content": '{"scores": [0.5]}'}}]}),
        _make_response(200, {"choices": [{"finish_reason": "stop", "message": {"content": '```json\n{"scores": [-0.2, 1.456]}\n```'}}]}),
    ]

    def fake_post(url, *, json, timeout, trust_env):
        calls.append((url, json, trust_env))
        return responses.pop(0)

    monkeypatch.setattr(score_caption.httpx, "post", fake_post)
    monkeypatch.setattr(score_caption.time, "sleep", lambda _: None)

    captions = [
        {"segment_id": 0, "start": 0.0, "end": 3.0, "caption": "a"},
        {"segment_id": 1, "start": 3.0, "end": 6.0, "caption": "b"},
    ]
    scored = score_caption.score(captions, max_retries=3, backoff=0.0)
    assert len(calls) == 3
    assert calls[0][0] == "http://127.0.0.1:9001/v1/chat/completions"
    assert calls[0][1]["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert calls[0][1]["response_format"]["type"] == "json_schema"
    assert calls[0][2] is False
    assert [item["score"] for item in scored] == [0.0, 1.0]


def test_score_caption_does_not_retry_400(monkeypatch):
    count = 0

    def fake_post(*args, **kwargs):
        nonlocal count
        count += 1
        return _make_response(400, "bad request")

    monkeypatch.setattr(score_caption.httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        score_caption.score([{"segment_id": 0, "start": 0.0, "end": 1.0, "caption": "x"}], max_retries=3, backoff=0.0)
    assert count == 1


def test_model_ask_retries_transient_503_but_not_length(monkeypatch):
    monkeypatch.setattr(model.time, "sleep", lambda _: None)
    seq = [
        _make_response(503, "busy"),
        _make_response(200, {"choices": [{"finish_reason": "stop", "message": {"content": '{"text": "ok"}'}}]}),
    ]
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return seq.pop(0)

    monkeypatch.setattr(model.httpx, "post", fake_post)
    out = model.ask(model.Narration, "p", {"a": 1}, max_retries=3, backoff=0.0)
    assert out.text == "ok"
    assert calls == 2

    length_calls = 0

    def fake_post_len(*args, **kwargs):
        nonlocal length_calls
        length_calls += 1
        return _make_response(200, {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]})

    monkeypatch.setattr(model.httpx, "post", fake_post_len)
    with pytest.raises(model.ModelOutputTooLong):
        model.ask(model.Narration, "p", {"a": 1}, max_retries=3, backoff=0.0)
    assert length_calls == 1


def test_caption_movie_bisects_on_length_and_cleans_frames(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "analysis").mkdir(parents=True)
    frames_dir = project / "analysis" / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(60):
        (frames_dir / f"{i:06}.jpg").write_bytes(b"jpg")

    class FakeFrames:
        def __init__(self, movie, directory, duration):
            self.directory = directory
            self.total = 60

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def through(self, stop):
            return min(stop, self.total)

        def path(self, index):
            return self.directory / f"{index:06}.jpg"

    seen_windows = []

    def fake_caption_batch(frames, duration, start, end, dialogue):
        seen_windows.append((start, end))
        if end - start > 30:
            raise model.ModelOutputTooLong("too long")
        return [model.Caption(start=float(start), end=float(end), caption=f"event {start}-{end}")]

    monkeypatch.setattr(pipeline, "AnalysisFrames", FakeFrames)
    monkeypatch.setattr(pipeline.model, "caption_batch", fake_caption_batch)
    monkeypatch.setenv("RECAP_CLEANUP_FRAMES", "1")

    job = {"duration": 60.0}
    caption_path = pipeline.caption_movie(project, job, tmp_path / "movie.mp4", [])
    saved = store.read_json(caption_path)
    assert seen_windows == [(0, 60), (0, 30), (30, 60)]
    assert saved["complete"] is True
    assert saved["processed_frames"] == 60
    assert len(saved["segments"]) == 2
    assert list(frames_dir.glob("*.jpg")) == []


def test_voice_segment_trims_gap_and_fit_narration_recovers_stale_cache(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "analysis").mkdir(parents=True)
    (project / "audio").mkdir(parents=True)

    # 3.0s footage = 90 frames = 72,000 samples
    segment = {
        "id": "segment_001",
        "text": "too long original text with many words",
        "candidates": [{"id": "segment_001-1", "event_id": "event_001", "start": 0.0, "end": 3.0}],
        "candidate_id": "segment_001-1",
        "source": {"start": 0.0, "end": 3.0},
        "event_ids": ["event_001"],
    }

    # Case 1: speech is 71,200 samples + 3,840 GAP = 75,040 (> 72,000), trimming gap makes it fit (71,200 <= 72,000)
    monkeypatch.setattr(
        pipeline,
        "speak",
        lambda proj, text: [{"text": text, "audio": "a.wav", "audio_offset": 0, "samples": 71_200, "gap_after": pipeline.GAP}],
    )
    voiced = pipeline.voice_segment(project, segment, full_source=True, allow_trim_gap=True)
    assert voiced["cues"][0]["gap_after"] == 0

    # Case 2: stale rewrite cache is still too long; fit_narration invalidates it and runs progressive shorten rounds
    cache_path = project / "analysis" / "selected-rewrite-0.json"
    store.write_json(cache_path, {"text": "stale cached text still too long"})

    sample_map = {
        "too long original text with many words": 96_000,  # 4.0s
        "stale cached text still too long": 84_000,         # 3.5s
        "round1 text": 76_000,                              # 3.17s (still > 3.0s)
        "round2 ok": 60_000,                                # 2.5s (fits)
    }
    monkeypatch.setattr(
        pipeline,
        "speak",
        lambda proj, text: [{"text": text, "audio": "a.wav", "audio_offset": 0, "samples": sample_map[text], "gap_after": pipeline.GAP}],
    )
    shorten_factors = []
    shorten_outputs = ["round1 text", "round2 ok"]

    def fake_shorten(text, duration, available, *, factor=0.9):
        shorten_factors.append(factor)
        return shorten_outputs.pop(0)

    monkeypatch.setattr(pipeline.model, "shorten", fake_shorten)
    fitted = pipeline.fit_narration(project, segment, cache_path, full_source=True)
    assert fitted["text"] == "round2 ok"
    assert shorten_factors == [0.9, 0.8]
    assert store.read_json(cache_path) == {"text": "round2 ok"}


def test_store_write_json_concurrent_atomic(tmp_path):
    target = tmp_path / "job.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.write_json(target, {"n": i}), range(40)))
    data = store.read_json(target)
    assert isinstance(data["n"], int)
    assert list(tmp_path.glob("*.tmp")) == []


def test_api_subtitles_encoding_size_preview_dedup_delete_and_health(tmp_path, monkeypatch):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(store, "PROJECTS", projects_dir)
    monkeypatch.setattr(api, "PROJECTS", projects_dir)
    monkeypatch.setattr(api.model, "check_model_server", lambda timeout=2.0: {"ok": True, "url": "http://127.0.0.1:8001/v1", "models": ["m"], "target_present": True})

    client = TestClient(api.app)

    # Health endpoint
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["vllm"]["ok"] is True
    assert body["worker_running"] is False
    assert "deepseek_key_configured" not in body

    # Create project
    pid = client.post("/api/projects").json()["id"]

    # Non-UTF-8 (GBK) subtitle returns 422 instead of 500
    gbk_srt = "1\n00:00:01,000 --> 00:00:02,000\n你好世界\n".encode("gbk")
    res = client.put(f"/api/projects/{pid}/subtitles?name=test.srt", content=gbk_srt)
    assert res.status_code == 422

    # Oversized subtitle returns 413
    monkeypatch.setattr(api, "MAX_SUBTITLE_SIZE", 16)
    res = client.put(f"/api/projects/{pid}/subtitles?name=big.srt", content=b"a" * 32)
    assert res.status_code == 413
    monkeypatch.setattr(api, "MAX_SUBTITLE_SIZE", 16 * 1024 * 1024)

    # Valid UTF-8 subtitle succeeds
    utf8_srt = "1\n00:00:01,000 --> 00:00:02,000\nHello\n".encode("utf-8")
    res = client.put(f"/api/projects/{pid}/subtitles?name=ok.srt", content=utf8_srt)
    assert res.status_code == 200

    # Concurrent clip preview deduplication + normalized float filename
    proj = projects_dir / pid
    (proj / "source" / "movie.mp4").write_bytes(b"fake-mp4")
    job = store.read_json(proj / "job.json")
    job.update(status="ready", duration=100.0, movie={"name": "m.mp4", "offset": 8, "total": 8, "complete": True})
    store.write_json(proj / "job.json", job)
    store.publish_timeline(proj, {
        "version": 1,
        "duration": 5.0,
        "duration_budget": 8.0,
        "segments": [{
            "id": "segment_001", "text": "Hi", "start": 0.0, "end": 5.0,
            "source": {"start": 1.0, "end": 6.0},
            "candidate_id": "segment_001-1",
            "candidates": [{"id": "segment_001-1", "event_id": "event_001", "start": 1.0, "end": 6.0}],
            "clips": [{"start": 1.0, "end": 6.0, "frames": 150}],
            "event_ids": ["event_001"],
            "cues": [],
        }],
    })

    preview_calls = 0
    preview_lock = threading.Lock()

    def fake_preview_clip(movie, start, end, destination):
        nonlocal preview_calls
        with preview_lock:
            preview_calls += 1
        time.sleep(0.05)
        Path(destination).write_bytes(b"clip-bytes")

    monkeypatch.setattr(api, "preview_clip", fake_preview_clip)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda q: client.get(f"/api/projects/{pid}/clips/segment_001-1/preview?{q}"),
            ["start=1&end=6", "start=1.0&end=6.0", "start=1.000&end=6.000", "start=1.0&end=6"],
        ))
    assert all(r.status_code == 200 for r in results)
    assert preview_calls == 1
    assert (proj / "exports" / "clips" / "segment_001-1-1.000-6.000.mp4").is_file()

    # Delete refuses running project (409), succeeds when idle (200)
    job["status"] = "running"
    store.write_json(proj / "job.json", job)
    assert client.delete(f"/api/projects/{pid}").status_code == 409

    job["status"] = "ready"
    store.write_json(proj / "job.json", job)
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert not proj.exists()


def test_worker_preflight_checks_vllm_without_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(worker, "check_media_tools", lambda: {"ok": True, "errors": [], "warnings": []})
    monkeypatch.setattr(worker, "kokoro", lambda: object())

    monkeypatch.setattr(worker.model, "check_model_server", lambda timeout=5.0: {"ok": True, "url": "u", "models": [], "target_present": True})
    worker.preflight()

    monkeypatch.setattr(worker.model, "check_model_server", lambda timeout=5.0: {"ok": False, "url": "u", "error": "down"})
    with pytest.raises(RuntimeError, match="模型服务不可达"):
        worker.preflight()


def test_worker_run_writes_run_log_and_pipeline_edit_enforces_budget(tmp_path, monkeypatch):
    projects_dir = tmp_path / "projects"
    proj = projects_dir / "p1"
    for sub in ("source", "analysis", "audio", "exports", "timelines"):
        (proj / sub).mkdir(parents=True)
    monkeypatch.setattr(store, "PROJECTS", projects_dir)
    monkeypatch.setattr(worker, "PROJECTS", projects_dir)

    (proj / "source" / "movie.mp4").write_bytes(b"mp4")
    store.publish_timeline(proj, {
        "version": 1,
        "duration": 5.0,
        "duration_budget": 8.0,
        "segments": [{
            "id": "segment_001", "text": "Old", "start": 0.0, "end": 5.0,
            "source": {"start": 0.0, "end": 5.0},
            "candidate_id": "segment_001-1",
            "candidates": [{"id": "segment_001-1", "event_id": "event_001", "start": 0.0, "end": 5.0}],
            "clips": [{"start": 0.0, "end": 5.0, "frames": 150}],
            "event_ids": ["event_001"],
            "cues": [],
        }],
    })
    store.write_json(proj / "job.json", {
        "id": "p1", "status": "queued", "stage": "queued",
        "movie": {"name": "movie.mp4", "offset": 3, "total": 3, "complete": True},
        "subtitles": "a.srt", "source_time": 0, "event_count": 1,
        "duration": 100.0, "timeline_version": 1, "asset_version": None, "error": None,
        "task": {"kind": "edit", "version": 1, "segment_id": "segment_001", "text": "New",
                 "candidate_id": "segment_001-1", "start": 0.0, "end": 9.0},
    })
    monkeypatch.setattr(
        pipeline,
        "speak",
        lambda p, t: [{"text": t, "audio": "a.wav", "audio_offset": 0, "samples": 24_000, "gap_after": pipeline.GAP}],
    )

    # Editing to 9.0s exceeds 8.0s budget -> worker marks error and records run.log
    assert worker.run() is True
    job = store.read_json(proj / "job.json")
    assert job["status"] == "error"
    assert "8%" in job["error"]
    run_log = (proj / "analysis" / "run.log").read_text(encoding="utf-8")
    assert "task edit started" in run_log
    assert "task edit failed" in run_log

    # Editing within budget (7.0s <= 8.0s) succeeds and publishes v2
    job.update(status="queued", stage="queued", error=None, task={
        "kind": "edit", "version": 1, "segment_id": "segment_001", "text": "New",
        "candidate_id": "segment_001-1", "start": 0.0, "end": 7.0,
    })
    store.write_json(proj / "job.json", job)
    assert worker.run() is True
    job = store.read_json(proj / "job.json")
    assert job["status"] == "ready"
    assert job["timeline_version"] == 2
    assert store.read_json(proj / "timeline.json")["duration"] == pytest.approx(7.0)
    run_log = (proj / "analysis" / "run.log").read_text(encoding="utf-8")
    assert "task edit finished" in run_log


def test_generate_works_without_subtitles_and_without_offset(tmp_path, monkeypatch):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(store, "PROJECTS", projects_dir)
    monkeypatch.setattr(api, "PROJECTS", projects_dir)

    client = TestClient(api.app)
    pid = client.post("/api/projects").json()["id"]

    # Cannot generate before movie upload completes
    assert client.post(f"/api/projects/{pid}/generate").status_code == 409

    # Upload movie only (no subtitles)
    res = client.put(f"/api/projects/{pid}/movie?name=clip.mp4&offset=0&total=8", content=b"12345678")
    assert res.status_code == 200
    assert res.json()["complete"] is True

    # Generate succeeds without subtitles and without subtitle_offset
    gen = client.post(f"/api/projects/{pid}/generate")
    assert gen.status_code == 202
    assert gen.json()["status"] == "queued"
    assert gen.json()["subtitles"] is None

    # pipeline.generate passes empty dialogue list when dialogue.srt is absent
    proj = projects_dir / pid
    job = store.read_json(proj / "job.json")
    seen_dialogue = []

    monkeypatch.setattr(pipeline, "probe", lambda movie: {"duration": 100.0, "width": 1280, "height": 720})

    def fake_caption_movie(project, job_arg, movie, dialogue):
        seen_dialogue.append(dialogue)
        cap_path = project / "analysis" / "caption.json"
        store.write_json(cap_path, {"segments": []})
        return cap_path

    monkeypatch.setattr(pipeline, "caption_movie", fake_caption_movie)
    monkeypatch.setattr(pipeline, "select_captions", lambda p, j, c: {"selected_segments": []})
    monkeypatch.setattr(pipeline, "write_selected_timeline", lambda p, j, d, s, dlg, v: {"version": v, "dialogue_len": len(dlg)})

    out = pipeline.generate(proj, job, proj / "source" / "movie.mp4", 1)
    assert seen_dialogue == [[]]
    assert out["dialogue_len"] == 0
