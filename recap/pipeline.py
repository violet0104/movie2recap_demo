import os
import textwrap
import time
import wave
from functools import cache
from hashlib import sha256
from math import ceil, floor

from utils.score_caption import score_record
from utils.select_segments_knapsack import RATIO, select_record

from . import model
from .media import (
    FPS, SAMPLES_PER_FRAME, SAMPLE_RATE, AnalysisFrames, probe, read_subtitles, render, window_dialogue,
)
from .store import append_run_log, movie_path, publish_timeline, read_json, version_path, write_json

CAPTION_FRAMES = 300
MIN_CAPTION_FRAMES = 15
NARRATION_BATCH = 12
GAP = round(0.16 * SAMPLE_RATE)
SHORTEN_FACTORS = (0.9, 0.8, 0.7)


def progress(project, job, stage, done=None, total=None, **details):
    count = None if total is None else {"done": done, "total": total}
    job.update(stage=stage, stage_progress=count, **details)
    write_json(project / "job.json", job)


def _cleanup_frames(directory):
    if os.environ.get("RECAP_CLEANUP_FRAMES", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    if not directory.is_dir():
        return
    for frame in directory.glob("*.jpg"):
        frame.unlink(missing_ok=True)


def caption_movie(project, job, movie, dialogue):
    path = project / "analysis" / "caption.json"
    duration = job["duration"]
    state = read_json(path) if path.is_file() else {
        "video_path": str(movie), "duration": duration, "target_fps": 1.0,
        "processed_frames": 0, "complete": False, "segments": [],
    }
    if state["complete"]:
        return path
    segments = state["segments"]
    first = state["processed_frames"]
    frames_dir = path.parent / "frames"
    with AnalysisFrames(movie, frames_dir, duration) as frames:
        while first < frames.total:
            stop = frames.through(first + CAPTION_FRAMES)
            if first >= stop:
                break
            pending = [(first, stop)]
            while pending:
                win_start, win_stop = pending.pop()
                progress(project, job, "events", source_time=win_start, event_count=len(segments))
                win_end = min(duration, win_stop)
                try:
                    batch = model.caption_batch(
                        [(index, frames.path(index)) for index in range(win_start, win_stop)],
                        duration, win_start, win_end, window_dialogue(dialogue, win_start, win_end),
                    )
                except model.ModelOutputTooLong:
                    if win_stop - win_start < MIN_CAPTION_FRAMES * 2:
                        raise
                    mid = (win_start + win_stop) // 2
                    append_run_log(project, f"events batch [{win_start}, {win_stop}) too long, bisecting at {mid}")
                    pending.append((mid, win_stop))
                    pending.append((win_start, mid))
                    continue
                cursor = win_start
                for event in sorted(batch, key=lambda item: item.start):
                    start, stop_time = max(cursor, event.start), min(win_end, event.end)
                    if stop_time - start < 1 / FPS or not event.caption.strip():
                        continue
                    segments.append({
                        "segment_id": len(segments), "start": start, "end": stop_time,
                        "caption": event.caption.strip(),
                    })
                    cursor = stop_time
                first = state["processed_frames"] = win_stop
                write_json(path, state)
    state["complete"] = True
    write_json(path, state)
    _cleanup_frames(frames_dir)
    return path


def select_captions(project, job, caption_path):
    captions = read_json(caption_path)
    if not captions["segments"]:
        raise ValueError("模型未识别到可用的剧情片段")
    progress(project, job, "scoring", source_time=job["duration"], event_count=len(captions["segments"]))
    score_path = caption_path.with_name("caption_score.json")
    if score_path.is_file():
        scored = read_json(score_path)
    else:
        started = time.monotonic()
        scored = score_record(captions)
        write_json(score_path, scored)
        append_run_log(project, f"scoring done ({len(scored['segments'])} segments, {time.monotonic() - started:.1f}s)")
    progress(project, job, "selection")
    selected = select_record(scored)
    write_json(caption_path.with_name("selected_segments.json"), selected)
    if not selected["selected_segments"]:
        raise ValueError("原片时长的 8% 预算内没有可选片段")
    append_run_log(
        project,
        f"selection done ({len(selected['selected_segments'])} segments, "
        f"{selected['selected_duration']:.1f}/{selected['target_duration']:.1f}s)",
    )
    return selected


@cache
def kokoro():
    import torch
    from kokoro import KPipeline

    if library := os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(library)
        EspeakWrapper.set_data_path(os.environ.get("PHONEMIZER_ESPEAK_DATA_PATH"))
    torch.set_num_threads(4)
    return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")


def caption_groups(tokens):
    group = []
    length = 0
    for token in tokens:
        if group and group[-1].whitespace and length + len(token.text) > 84:
            yield group
            group, length = [], 0
        group.append(token)
        length += len(token.text) + len(token.whitespace)
        if length >= 36 and token.text.endswith((".", "!", "?")) and token.whitespace:
            yield group
            group, length = [], 0
    if group:
        yield group


def speak(project, text):
    key = sha256(f"af_heart:paragraph-v1\n{text}".encode()).hexdigest()
    path = project / "audio" / f"{key}.wav"
    index = path.with_suffix(".json")
    if path.is_file() and index.is_file():
        return read_json(index)

    starts = []
    samples = 0
    temporary = path.with_suffix(".wav.tmp")
    with wave.open(str(temporary), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        for result in kokoro()(text, voice="af_heart", split_pattern=None):
            audio = result.audio
            if audio is None or not audio.numel():
                continue
            pcm = (audio.clamp(-1, 1) * 32767).round().numpy().astype("<i2")
            output.writeframes(pcm.tobytes())
            for i, group in enumerate(caption_groups(result.tokens)):
                start = 0 if i == 0 else next(
                    (round(token.start_ts * SAMPLE_RATE) for token in group if token.start_ts is not None), 0,
                )
                offset = samples + min(start, len(pcm) - 1)
                caption = "".join(token.text + token.whitespace for token in group).strip()
                if starts and offset <= starts[-1][0]:
                    starts[-1] = (starts[-1][0], f"{starts[-1][1]} {caption}")
                else:
                    starts.append((offset, caption))
            samples += len(pcm)
    if not starts or not samples:
        raise ValueError(f"Kokoro 未生成语音：{text}")
    cues = [{
        "text": textwrap.fill(caption, width=44, break_long_words=False, break_on_hyphens=False),
        "audio": path.name, "audio_offset": start,
        "samples": end - start, "gap_after": 0,
    } for (start, caption), (end, _) in zip(starts, starts[1:] + [(samples, "")])]
    cues[-1]["gap_after"] = GAP
    temporary.replace(path)
    write_json(index, cues)
    return cues


def available_ranges(primary, candidates):
    used = [primary]
    for candidate in candidates:
        parts = [candidate]
        for taken in used:
            remaining = []
            for part in parts:
                if part["start"] < min(part["end"], taken["start"]):
                    remaining.append({"start": part["start"], "end": min(part["end"], taken["start"])})
                if max(part["start"], taken["end"]) < part["end"]:
                    remaining.append({"start": max(part["start"], taken["end"]), "end": part["end"]})
            parts = remaining
        used.extend(parts)
    return used


class ShortFootage(ValueError):
    def __init__(self, needed, available):
        self.needed = needed
        self.available = available
        super().__init__(f"旁白需要 {needed:.2f} 秒，可用画面 {available:.2f} 秒。请缩短旁白或扩大剪辑区间。")


def voice_segment(project, segment, *, full_source=False, allow_trim_gap=False):
    cues = [dict(cue) for cue in speak(project, segment["text"])]
    required = ceil(sum(cue["samples"] + cue["gap_after"] for cue in cues) / SAMPLES_PER_FRAME)
    selected = next(shot for shot in segment["candidates"] if shot["id"] == segment["candidate_id"])
    if segment.get("use_all_candidates"):
        alternatives = segment["candidates"][1:]
    else:
        alternatives = [
            shot for shot in segment["candidates"]
            if shot["id"] != selected["id"] and shot["event_id"] == selected["event_id"]
        ]
    ranges = available_ranges(segment["source"], alternatives)
    capacities = [floor((shot["end"] - shot["start"]) * FPS + 1e-8) for shot in ranges]
    total_capacity = sum(capacities)
    if total_capacity < required and allow_trim_gap and cues and cues[-1].get("gap_after", 0) > 0:
        cues[-1]["gap_after"] = 0
        trimmed = ceil(sum(cue["samples"] + cue["gap_after"] for cue in cues) / SAMPLES_PER_FRAME)
        if trimmed <= total_capacity:
            required = trimmed
    if total_capacity < required:
        raise ShortFootage(required / FPS, total_capacity / FPS)
    if full_source:
        required = total_capacity
    clips = []
    for shot, capacity in zip(ranges, capacities, strict=True):
        length = min(required, capacity)
        if length:
            clips.append({"start": shot["start"], "end": shot["start"] + length / FPS, "frames": length})
            required -= length
        if not required:
            break
    return {**segment, "cues": cues, "clips": clips}


def make_timeline(segments, version):
    cursor = 0
    for segment in segments:
        segment["start"] = cursor / SAMPLE_RATE
        cue_start = cursor
        for cue in segment["cues"]:
            cue["start_sample"] = cue_start
            cue["end_sample"] = cue_start + cue["samples"]
            cue_start = cue["end_sample"] + cue.get("gap_after", GAP)
        cursor += sum(clip["frames"] for clip in segment["clips"]) * SAMPLES_PER_FRAME
        segment["end"] = cursor / SAMPLE_RATE
    return {"version": version, "duration": cursor / SAMPLE_RATE, "samples": cursor, "segments": segments}


def script_segments(script, events, prefix="segment"):
    by_id = {event["id"]: event for event in events}
    segments = []
    for line in script.segments:
        segment_id = f"{prefix}_{len(segments) + 1:03}"
        candidates = []
        for shot in line.candidates:
            event = by_id.get(shot.event_id)
            if event is None:
                continue
            start, end = max(shot.start, event["start"]), min(shot.end, event["end"])
            if end - start >= 1 / FPS:
                candidates.append({
                    "id": f"{segment_id}-{len(candidates) + 1}", "event_id": shot.event_id,
                    "start": start, "end": end,
                })
        if line.text.strip() and candidates:
            segment = {
                "id": segment_id, "text": line.text.strip(), "candidates": candidates,
                "candidate_id": candidates[0]["id"],
                "source": {key: candidates[0][key] for key in ("start", "end")},
                "event_ids": list(dict.fromkeys(shot["event_id"] for shot in candidates)),
            }
            if getattr(line, "use_all_candidates", False) and all(
                a["end"] <= b["start"] for a, b in zip(candidates, candidates[1:])
            ):
                segment["use_all_candidates"] = True
            segments.append(segment)
    return segments


def fit_narration(project, segment, path, *, full_source=False):
    try:
        return voice_segment(project, segment, full_source=full_source, allow_trim_gap=True)
    except ShortFootage as error:
        current_text = segment["text"]
        current_error = error
        if path.is_file():
            cached = read_json(path).get("text", "").strip()
            if cached:
                try:
                    return voice_segment(project, {**segment, "text": cached}, full_source=full_source, allow_trim_gap=True)
                except ShortFootage as cached_error:
                    append_run_log(
                        project,
                        f"rewrite cache stale for {segment['id']}: "
                        f"needed {cached_error.needed:.2f}s > available {cached_error.available:.2f}s",
                    )
                    if len(cached.split()) <= len(current_text.split()):
                        current_text = cached
                        current_error = cached_error
        for round_index, factor in enumerate(SHORTEN_FACTORS, 1):
            shortened = model.shorten(
                current_text, current_error.needed, current_error.available, factor=factor,
            ).strip()
            if not shortened:
                continue
            write_json(path, {"text": shortened})
            try:
                return voice_segment(
                    project, {**segment, "text": shortened}, full_source=full_source, allow_trim_gap=True,
                )
            except ShortFootage as next_error:
                append_run_log(
                    project,
                    f"shorten round {round_index} for {segment['id']} still long: "
                    f"{next_error.needed:.2f}s > {next_error.available:.2f}s",
                )
                current_text = shortened
                current_error = next_error
        raise current_error


def write_selected_timeline(project, job, directory, selection, dialogue, version):
    events = [{**segment, "id": f"event_{segment['segment_id'] + 1:03}"}
              for segment in selection["selected_segments"]]
    narration = []
    started = time.monotonic()
    for offset in range(0, len(events), NARRATION_BATCH):
        progress(project, job, "script", offset, len(events))
        path = directory / f"selected-script-{offset}.json"
        if path.is_file():
            texts = read_json(path)
        else:
            scenes = [{
                "caption": event["caption"],
                "dialogue": [cue["text"] for cue in window_dialogue(dialogue, event["start"], event["end"])],
                "max_words": max(1, floor((event["end"] - event["start"] - 0.25) * 2.3)),
            } for event in events[offset:offset + NARRATION_BATCH]]
            texts = model.narrate_selected(scenes, narration[-2:])
            write_json(path, texts)
        narration.extend(texts)
    append_run_log(project, f"script done ({len(narration)} segments, {time.monotonic() - started:.1f}s)")
    script = model.Script(segments=[
        model.ScriptLine(text=text, candidates=[
            model.Candidate(event_id=event["id"], start=event["start"], end=event["end"])
        ])
        for event, text in zip(events, narration, strict=True)
    ])
    draft = script_segments(script, events)
    if len(draft) != len(events):
        raise ValueError("所选片段缺少有效的旁白或画面")
    segments = []
    audio_started = time.monotonic()
    for index, segment in enumerate(draft):
        progress(project, job, "audio", index, len(draft))
        segments.append(fit_narration(
            project, segment, directory / f"selected-rewrite-{index}.json", full_source=True,
        ))
    append_run_log(project, f"audio done ({len(segments)} segments, {time.monotonic() - audio_started:.1f}s)")
    timeline = make_timeline(segments, version)
    timeline["duration_budget"] = selection["original_duration"] * selection["target_ratio"]
    publish_timeline(project, timeline)
    return timeline


def generate(project, job, movie, version):
    current = version_path(project, version)
    if current.is_file():
        return read_json(current)
    analysis = project / "analysis"
    progress(project, job, "probe")
    media = probe(movie)
    write_json(analysis / "media.json", media)
    subtitles_path = project / "source" / "dialogue.srt"
    dialogue = read_subtitles(subtitles_path.read_bytes()) if subtitles_path.is_file() else []
    job["duration"] = media["duration"]
    directory = analysis / f"recap-{version}"
    directory.mkdir(exist_ok=True)
    events_started = time.monotonic()
    caption_path = caption_movie(project, job, movie, dialogue)
    append_run_log(project, f"events done ({time.monotonic() - events_started:.1f}s)")
    selection = select_captions(project, job, caption_path)
    return write_selected_timeline(project, job, directory, selection, dialogue, version)


def edit(project, job, task):
    timeline = read_json(project / "timeline.json")
    if timeline["version"] > task["version"]:
        return timeline
    progress(project, job, "editing")
    segments = timeline["segments"]
    index = next(index for index, segment in enumerate(segments) if segment["id"] == task["segment_id"])
    segment = {**segments[index], "text": task["text"].strip(),
               "candidate_id": task["candidate_id"],
               "source": {"start": task["start"], "end": task["end"]}}
    if segment["candidate_id"] != segments[index]["candidate_id"] or segment["source"] != segments[index]["source"]:
        segment.pop("use_all_candidates", None)
    segments[index] = voice_segment(project, segment, full_source=True, allow_trim_gap=True)
    updated = make_timeline(segments, timeline["version"] + 1)
    budget = timeline.get("duration_budget") or (job["duration"] * RATIO)
    updated["duration_budget"] = budget
    if updated["duration"] > budget + 1e-6:
        raise ValueError("请调整旁白长度，使成片时长不超过原片的 8%")
    publish_timeline(project, updated)
    return updated


def process(project, job):
    task = job["task"]
    movie = movie_path(project, job["movie"])
    if task["kind"] == "edit":
        job["timeline_version"] = edit(project, job, task)["version"]
        return
    if task["kind"] == "generate":
        timeline = generate(project, job, movie, task.get("version", 1))
        job["timeline_version"] = timeline["version"]
    else:
        timeline = read_json(version_path(project, task["version"]))
    render_started = time.monotonic()
    render(project, movie, timeline, lambda stage, done, total: progress(project, job, stage, done, total))
    append_run_log(project, f"render+encode v{timeline['version']} done ({time.monotonic() - render_started:.1f}s)")
    job["asset_version"] = timeline["version"]
