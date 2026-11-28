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


def test_score_caption_uses_qwen_fixed_key_schema_retries_and_clamping(monkeypatch):
    monkeypatch.setenv("RECAP_MODEL_URL", "http://127.0.0.1:9001/v1/")
    monkeypatch.setenv("RECAP_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    calls = []
    responses = [
        _make_response(429, "rate limited"),
        _make_response(200, {"choices": [{"finish_reason": "stop", "message": {"content": '{"0": 50}'}}]}),
        _make_response(200, {"choices": [{"finish_reason": "stop", "message": {"content": '```json\n{"0": -5, "1": 145}\n```'}}]}),
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
    schema = calls[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["0", "1"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["0"] == {"type": "integer", "minimum": 0, "maximum": 100}
    assert calls[0][2] is False
    assert [item["score"] for item in scored] == [0.0, 1.0]


def test_score_caption_batches_and_bisects_on_length(monkeypatch):
    monkeypatch.setattr(score_caption.time, "sleep", lambda _: None)
    seen_batches = []

    def fake_post(url, *, json, timeout, trust_env):
        keys = json["response_format"]["json_schema"]["schema"]["required"]
        seen_batches.append(keys)
        if len(keys) > 20:
            return _make_response(200, {"choices": [{"finish_reason": "length", "message": {"content": "{truncated"}}]})
        return _make_response(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"content": json_lib.dumps({k: int(k) * 2 for k in keys})},
        }]})

    json_lib = json
    monkeypatch.setattr(score_caption.httpx, "post", fake_post)

    captions = [
        {"segment_id": i, "start": float(i), "end": float(i + 1), "caption": f"scene {i}"}
        for i in range(45)
    ]
    scored = score_caption.score(captions, batch_size=30, max_retries=2, backoff=0.0)
    # Batch 1 (30 items) hits length -> bisected into 15 + 15; Batch 2 (15 items) succeeds directly
    assert [len(b) for b in seen_batches] == [30, 15, 15, 15]
    assert len(scored) == 45
    assert scored[0]["score"] == 0.0
    assert scored[25]["score"] == 0.5
    assert scored[44]["score"] == 0.88


def test_score_caption_strict_context_cap_for_2500_segments_and_400_fallback(monkeypatch):
    # 2,500 segments (longer than Avatar Extended) must strictly stay within context_chars
    many_captions = [
        {"segment_id": i, "start": float(i * 4), "end": float(i * 4 + 4),
         "caption": f"Distinct event {i}: a character moves across the forest and reacts to a sudden threat."}
        for i in range(2500)
    ]
    ctx = score_caption._format_context(many_captions, max_chars=24_000)
    assert len(ctx) <= 24_000

    # Consecutive duplicate segments collapse cleanly in overview context
    dup_captions = [
        {"segment_id": i, "start": float(i * 2), "end": float(i * 2 + 2),
         "caption": ("The scene shifts to a room." if i % 2 == 0 else "The scene transitions to a room.")}
        for i in range(200)
    ]
    dup_ctx = score_caption._format_context(dup_captions, max_chars=24_000)
    assert "x200" in dup_ctx
    assert len(dup_ctx) < 200

    # Simulate vLLM HTTP 400 maximum context length -> score() halves context and succeeds
    seen_prompt_lens = []

    def fake_post(url, *, json, timeout, trust_env):
        user_msg = json["messages"][1]["content"]
        seen_prompt_lens.append(len(user_msg))
        if len(user_msg) > 5000:
            return _make_response(400, '{"error":{"message":"This model\'s maximum context length is 65536 tokens. (parameter=input_tokens)"}}')
        keys = json["response_format"]["json_schema"]["schema"]["required"]
        return _make_response(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"content": json_lib.dumps({k: 50 for k in keys})},
        }]})

    json_lib = json
    monkeypatch.setattr(score_caption.httpx, "post", fake_post)
    sample = many_captions[:60]
    scored = score_caption.score(sample, batch_size=30, context_chars=8000, max_retries=2, backoff=0.0)
    assert len(scored) == 60
    assert seen_prompt_lens[0] > 5000
    assert seen_prompt_lens[1] < seen_prompt_lens[0]


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


def test_caption_batch_sends_penalties_and_max_items_and_recovers_collapsed_loop(tmp_path, monkeypatch):
    # 1. Verify caption_batch enforces bounded maxItems + anti-repetition penalties
    frame_file = tmp_path / "000000.jpg"
    frame_file.write_bytes(b"jpg")
    seen_payloads = []

    def fake_post(url, *, json, timeout, trust_env):
        seen_payloads.append(json)
        return _make_response(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"content": '{"segments": [{"start": 0.0, "end": 10.0, "caption": "Hero walks in."}]}'},
        }]})

    monkeypatch.setattr(model.httpx, "post", fake_post)
    monkeypatch.setenv("RECAP_REPETITION_PENALTY", "1.15")
    monkeypatch.setenv("RECAP_FREQUENCY_PENALTY", "0.25")
    segs = model.caption_batch([(0, frame_file)], 90.0, 0.0, 90.0, [])
    assert len(segs) == 1
    payload = seen_payloads[0]
    assert payload["repetition_penalty"] == 1.15
    assert payload["frequency_penalty"] == 0.25
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["segments"]["maxItems"] == 15

    # 2. Verify caption_movie detects lag-2 A-B-A-B loop collapse, bisects, and merges adjacent duplicates
    project = tmp_path / "proj_loop"
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
            # Return period-2 A-B-A-B collapsed loop
            return [
                model.Caption(start=float(i * 10), end=float((i + 1) * 10),
                              caption=("The figure on the right raises their hand." if i % 2 == 0
                                       else "The figure on the left looks down."))
                for i in range(6)
            ]
        # Clean bisected sub-batches with an adjacent near-duplicate across boundary (20-30s and 30-40s)
        if start == 0:
            return [
                model.Caption(start=0.0, end=20.0, caption="A soldier enters the misty forest."),
                model.Caption(start=20.0, end=30.0, caption="A blue-skinned warrior climbs a tree."),
            ]
        return [
            model.Caption(start=30.0, end=40.0, caption="The scene shifts to a blue-skinned warrior climbs a tree."),
            model.Caption(start=40.0, end=60.0, caption="A helicopter flies overhead."),
        ]

    monkeypatch.setattr(pipeline, "AnalysisFrames", FakeFrames)
    monkeypatch.setattr(pipeline.model, "caption_batch", fake_caption_batch)

    job = {"duration": 60.0}
    caption_path = pipeline.caption_movie(project, job, tmp_path / "movie.mp4", [])
    saved = store.read_json(caption_path)
    assert seen_windows == [(0, 60), (0, 30), (30, 60)]
    # Adjacent near-duplicate (20-30s + 30-40s) merged into 20-40s -> 3 clean segments total
    assert len(saved["segments"]) == 3
    assert saved["segments"][1]["start"] == 20.0
    assert saved["segments"][1]["end"] == 40.0


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

    def fake_shorten(text, duration, available, *, factor=0.85, max_words=None):
        shorten_factors.append(factor)
        return shorten_outputs.pop(0)

    monkeypatch.setattr(pipeline.model, "shorten", fake_shorten)
    fitted = pipeline.fit_narration(project, segment, cache_path, full_source=True)
    assert fitted["text"] == "round2 ok"
    assert shorten_factors == [0.85, 0.72]
    assert store.read_json(cache_path) == {"text": "round2 ok"}


def test_fit_narration_speed_boost_and_deterministic_trim_fallback(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "analysis").mkdir(parents=True)
    (project / "audio").mkdir(parents=True)

    # 4.00s clip (120 frames = 96,000 samples)
    segment = {
        "id": "segment_002",
        "text": "The hero runs across the dark bridge to escape the guards.",
        "candidates": [{"id": "segment_002-1", "event_id": "event_002", "start": 0.0, "end": 4.0}],
        "candidate_id": "segment_002-1",
        "source": {"start": 0.0, "end": 4.0},
        "event_ids": ["event_002"],
    }

    # Case A: model.shorten fails to shrink below 4.77s, but Kokoro speed boost (~1.24x) fits 4.00s naturally
    seen_speeds = []

    def fake_speak_speed(proj, text, *, speed=1.0):
        seen_speeds.append(speed)
        # 4.77s at speed=1.0 -> 114,480 samples; scales inversely with speed
        samples = round(114_480 / speed)
        return [{"text": text, "audio": "a.wav", "audio_offset": 0, "samples": samples, "gap_after": pipeline.GAP}]

    monkeypatch.setattr(pipeline, "speak", fake_speak_speed)
    monkeypatch.setattr(pipeline.model, "shorten", lambda text, dur, avail, *, factor=0.85, max_words=None: text)

    cache_a = project / "analysis" / "rewrite-a.json"
    fitted = pipeline.fit_narration(project, segment, cache_a, full_source=True)
    assert fitted["text"] == segment["text"]
    assert any(s > 1.0 for s in seen_speeds)
    saved_a = store.read_json(cache_a)
    assert 1.0 < saved_a["speed"] <= pipeline.MAX_TTS_SPEED

    # Case B: even at MAX_TTS_SPEED (1.25x), 11 words take too long (0.6s/word -> 5.28s > 4.0s); deterministic trim succeeds
    def fake_speak_word_based(proj, text, *, speed=1.0):
        samples = round(len(text.split()) * 14_400 / speed)  # 0.6s per word at 1.0x
        return [{"text": text, "audio": "a.wav", "audio_offset": 0, "samples": samples, "gap_after": pipeline.GAP}]

    monkeypatch.setattr(pipeline, "speak", fake_speak_word_based)
    cache_b = project / "analysis" / "rewrite-b.json"
    fitted_b = pipeline.fit_narration(project, segment, cache_b, full_source=True)
    assert len(fitted_b["text"].split()) < len(segment["text"].split())
    assert fitted_b["text"].endswith(".")
    assert store.read_json(cache_b)["speed"] == pipeline.MAX_TTS_SPEED


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


def test_narrate_selected_uses_fixed_key_schema_and_bisects_on_mismatch(monkeypatch):
    seen_calls = []

    def fake_post(url, *, json, timeout, trust_env):
        schema = json["response_format"]["json_schema"]["schema"]
        keys = schema["required"]
        seen_calls.append(keys)
        # Simulate count mismatch / bad output on 4-scene batch, succeeding once bisected to 2 + 2
        if len(keys) == 4:
            return _make_response(200, {"choices": [{"finish_reason": "stop", "message": {"content": '{"1": "only one"}'}}]})
        return _make_response(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"content": json_lib.dumps({k: f"Narration {len(seen_calls)}-{k}." for k in keys})},
        }]})

    json_lib = json
    monkeypatch.setattr(model.httpx, "post", fake_post)

    scenes = [
        {"caption": f"Scene {i}", "dialogue": [], "max_words": 15}
        for i in range(4)
    ]
    texts = model.narrate_selected(scenes, ["Prev."])
    assert seen_calls == [["1", "2", "3", "4"], ["1", "2"], ["1", "2"]]
    assert len(texts) == 4
    assert all(texts)


def test_knapsack_prefers_duration_weighted_value_and_filters_short_fragments():
    # Budget = 8.0s (8% of 100s). Eight 1.0s fragments @ 0.7 would beat one 8.0s scene @ 0.9 under unweighted score!
    data = {
        "video_path": "movie.mp4",
        "duration": 100.0,
        "segments": [
            {"segment_id": i, "start": float(i), "end": float(i + 1), "caption": f"frag {i}", "score": 0.7}
            for i in range(10)
        ] + [
            {"segment_id": 99, "start": 50.0, "end": 58.0, "caption": "Key climax", "score": 0.9},
        ],
    }
    selected = select_segments_knapsack.select_segments(data)
    assert [s["segment_id"] for s in selected] == [99]

    # Even with min_duration=0.0, duration-weighted value (0.9 * 8 = 7.2 > 0.7 * 8 = 5.6) still picks segment 99
    selected_no_floor = select_segments_knapsack.select_segments(data, min_duration=0.0)
    assert [s["segment_id"] for s in selected_no_floor] == [99]

    # Tiny video where all segments are shorter than min_duration falls back cleanly instead of returning empty
    tiny = {
        "video_path": "tiny.mp4",
        "duration": 20.0,
        "segments": [
            {"segment_id": 0, "start": 0.0, "end": 1.0, "caption": "only shot", "score": 0.8},
        ],
    }
    assert [s["segment_id"] for s in select_segments_knapsack.select_segments(tiny)] == [0]


def test_voice_segment_trims_wav_silence_and_fit_narration_handles_one_word_and_silent_fallback(tmp_path, monkeypatch):
    import wave
    from array import array

    project = tmp_path / "proj"
    (project / "analysis").mkdir(parents=True)
    (project / "audio").mkdir(parents=True)

    # 1. Real 24kHz mono WAV with 0.15s leading silence + 0.75s speech + 0.20s trailing silence = 1.10s total
    wav_path = project / "audio" / "padded.wav"
    pcm = array("h", [0] * 3600 + [5000] * 18_000 + [0] * 4800)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setparams((1, 2, media.SAMPLE_RATE, 0, "NONE", "not compressed"))
        wf.writeframes(pcm.tobytes())

    seg_1s = {
        "id": "segment_001",
        "text": "Explosion.",
        "candidates": [{"id": "segment_001-1", "event_id": "event_001", "start": 0.0, "end": 1.0}],
        "candidate_id": "segment_001-1",
        "source": {"start": 0.0, "end": 1.0},
        "event_ids": ["event_001"],
    }
    monkeypatch.setattr(
        pipeline,
        "speak",
        lambda proj, text, **kw: [{
            "text": text, "audio": "padded.wav", "audio_offset": 0,
            "samples": len(pcm), "gap_after": pipeline.GAP,
        }],
    )
    voiced = pipeline.voice_segment(project, seg_1s, full_source=True, allow_trim_gap=True)
    assert voiced["cues"][0]["audio_offset"] > 0
    assert voiced["cues"][0]["samples"] <= media.SAMPLE_RATE

    # 2. One-word cue that still takes 1.10s at MAX_TTS_SPEED (1.25x) on a 1.00s clip -> Tier 3b emergency speed fits
    seen_speeds = []

    def fake_speak_one_word(proj, text, *, speed=1.0):
        seen_speeds.append(speed)
        # 1.10s at 1.25x => 1.375s at 1.0x (33,000 samples)
        samples = round(33_000 / speed)
        return [{"text": text, "audio": "missing.wav", "audio_offset": 0, "samples": samples, "gap_after": pipeline.GAP}]

    monkeypatch.setattr(pipeline, "speak", fake_speak_one_word)
    monkeypatch.setattr(pipeline.model, "shorten", lambda text, dur, avail, *, factor=0.85, max_words=None: text)
    cache_emerg = project / "analysis" / "rewrite-emerg.json"
    fitted_emerg = pipeline.fit_narration(project, seg_1s, cache_emerg, full_source=True)
    assert fitted_emerg["cues"]
    saved_emerg = store.read_json(cache_emerg)
    assert pipeline.MAX_TTS_SPEED < saved_emerg["speed"] <= pipeline.EMERGENCY_TTS_SPEED

    # 3. Physically un-narratable 0.50s clip -> Tier 4 silent visual bridge fallback + timeline works cleanly
    seg_half = {
        "id": "segment_002",
        "text": "Run.",
        "candidates": [{"id": "segment_002-1", "event_id": "event_002", "start": 10.0, "end": 10.5}],
        "candidate_id": "segment_002-1",
        "source": {"start": 10.0, "end": 10.5},
        "event_ids": ["event_002"],
    }
    cache_silent = project / "analysis" / "rewrite-silent.json"
    fitted_silent = pipeline.fit_narration(project, seg_half, cache_silent, full_source=True)
    assert fitted_silent["cues"] == []
    assert sum(c["frames"] for c in fitted_silent["clips"]) == 15
    assert store.read_json(cache_silent)["silent"] is True
    tl = pipeline.make_timeline([fitted_emerg, fitted_silent], version=1)
    assert tl["duration"] == pytest.approx(1.5, abs=1e-6)


def test_write_selected_timeline_invalidates_stale_script_cache_when_selection_changes(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    directory = project / "analysis" / "recap-1"
    directory.mkdir(parents=True)
    (project / "audio").mkdir(parents=True)

    # Pre-populate stale script & rewrite cache without a matching signature
    store.write_json(directory / "selected-script-0.json", ["stale narration 1", "stale narration 2"])
    store.write_json(directory / "selected-rewrite-0.json", {"text": "stale rewrite"})

    narrate_calls = []

    def fake_narrate(scenes, previous):
        narrate_calls.append(len(scenes))
        return [f"Fresh line {i + 1}." for i in range(len(scenes))]

    monkeypatch.setattr(pipeline.model, "narrate_selected", fake_narrate)
    monkeypatch.setattr(
        pipeline,
        "speak",
        lambda proj, text, **kw: [{"text": text, "audio": "x.wav", "audio_offset": 0, "samples": 24_000, "gap_after": 0}],
    )

    selection = {
        "original_duration": 100.0,
        "target_ratio": 0.08,
        "selected_segments": [
            {"segment_id": 5, "start": 10.0, "end": 14.0, "caption": "Scene A", "score": 0.9},
        ],
    }
    job = {"duration": 100.0}
    tl1 = pipeline.write_selected_timeline(project, job, directory, selection, [], version=1)
    assert narrate_calls == [1]
    assert tl1["segments"][0]["text"] == "Fresh line 1."

    # Running again with identical selection reuses the cache (no second narrate_selected call)
    tl2 = pipeline.write_selected_timeline(project, job, directory, selection, [], version=1)
    assert narrate_calls == [1]
    assert tl2["segments"][0]["text"] == "Fresh line 1."
