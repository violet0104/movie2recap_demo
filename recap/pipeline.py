import os
import textwrap
import wave
from functools import cache
from hashlib import sha256
from math import ceil, floor

from utils.score_caption import score_record
from utils.select_segments_knapsack import select_record

from . import model
from .media import (
    FPS, SAMPLES_PER_FRAME, SAMPLE_RATE, AnalysisFrames, probe, read_subtitles, render, window_dialogue,
)
from .store import movie_path, publish_timeline, read_json, version_path, write_json

CAPTION_FRAMES = 300
NARRATION_BATCH = 12
GAP = round(0.16 * SAMPLE_RATE)
MIN_RUNTIME = 600
MAX_RUNTIME = 1200
CHAPTERS = 6


def progress(project, job, stage, done=None, total=None, **details):
    count = None if total is None else {"done": done, "total": total}
    job.update(stage=stage, stage_progress=count, **details)
    write_json(project / "job.json", job)


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
    with AnalysisFrames(movie, path.parent / "frames", duration) as frames:
        while first < frames.total:
            progress(project, job, "events", source_time=first, event_count=len(segments))
            stop = frames.through(first + CAPTION_FRAMES)
            if first >= stop:
                break
            end = min(duration, stop)
            batch = model.caption_batch(
                [(index, frames.path(index)) for index in range(first, stop)],
                duration, first, end, window_dialogue(dialogue, first, end),
            )
            cursor = first
            for event in sorted(batch, key=lambda item: item.start):
                start, stop_time = max(cursor, event.start), min(end, event.end)
                if stop_time - start < 1 / FPS or not event.caption.strip():
                    continue
                segments.append({
                    "segment_id": len(segments), "start": start, "end": stop_time,
                    "caption": event.caption.strip(),
                })
                cursor = stop_time
            first = state["processed_frames"] = stop
            write_json(path, state)
    state["complete"] = True
    write_json(path, state)
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
        scored = score_record(captions)
        write_json(score_path, scored)
    progress(project, job, "selection")
    selected = select_record(scored)
    write_json(caption_path.with_name("selected_segments.json"), selected)
    if not selected["selected_segments"]:
        raise ValueError("原片时长的 8% 预算内没有可选片段")
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


def voice_segment(project, segment, *, full_source=False):
    cues = speak(project, segment["text"])
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
    if sum(capacities) < required:
        raise ShortFootage(required / FPS, sum(capacities) / FPS)
    if full_source:
        required = sum(capacities)
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


def build_story(project, job, directory, events, dialogue, duration):
    chapter_events = {}
    for event in events:
        number = min(CHAPTERS - 1, int(event["start"] / duration * CHAPTERS))
        chapter_events.setdefault(number, []).append(event)
    chapters = []
    for chapter_index, (number, items) in enumerate(sorted(chapter_events.items())):
        progress(project, job, "story", chapter_index, len(chapter_events))
        path = directory / f"chapter-{number}.json"
        if path.is_file():
            chapter = read_json(path)
        else:
            midpoint = ceil(len(items) / 2)
            halves = [part for part in (items[:midpoint], items[midpoint:]) if part]
            chapter = {"synopsis": "", "beats": []}
            for index, part in enumerate(halves):
                cues = window_dialogue(dialogue, part[0]["start"], part[-1]["end"])
                section = model.outline(
                    part, [{"start": cue["start"], "text": cue["text"]} for cue in cues],
                    [earlier["synopsis"] for earlier in chapters] + [chapter["synopsis"]],
                    number == max(chapter_events) and index == len(halves) - 1,
                ).model_dump()
                chapter["synopsis"] = (chapter["synopsis"] + " " + section["synopsis"]).strip()
                chapter["beats"].extend(section["beats"])
            allowed = {event["id"] for event in items}
            for beat in chapter["beats"]:
                beat["event_ids"] = [event_id for event_id in beat["event_ids"] if event_id in allowed]
            if any(not beat["event_ids"] for beat in chapter["beats"]):
                raise ValueError(f"第 {number + 1} 章的主线转折缺少有效事件编号")
            chapter.update(number=number, event_ids=[event["id"] for event in items])
            write_json(path, chapter)
        if chapter["beats"]:
            chapters.append(chapter)
    if not chapters:
        raise ValueError("全片主线没有足够的剧情转折")
    chapters[-1]["final"] = True
    return chapters


def draft_segments(project, job, directory, events, chapters, aim, words, feedback, attempt, dialogue=()):
    by_id = {event["id"]: event for event in events}
    story = " ".join(chapter["synopsis"] for chapter in chapters)
    beat_count = sum(len(chapter["beats"]) for chapter in chapters)
    segments = []
    for index, chapter in enumerate(chapters):
        progress(project, job, "script", index, len(chapters))
        number = chapter["number"]
        path = directory / f"script-{attempt}-{number}.json"
        if path.is_file():
            draft = model.Draft.model_validate(read_json(path))
        else:
            event_ids = dict.fromkeys(event_id for beat in chapter["beats"] for event_id in beat["event_ids"])
            source_ranges = []
            frame_times = set()
            frames_dir = directory.parent / "frames"
            for event_id in event_ids:
                event = by_id[event_id]
                start, end = event["start"], event["end"]
                cues = window_dialogue(dialogue, start, end)
                source_ranges.append({
                    "id": event_id, "start": start, "end": end,
                    "dialogue": [{"start": cue["start"], "text": cue["text"]} for cue in cues],
                })
                first, last = ceil(start), ceil(end) - 1
                if first <= last:
                    frame_times.update((first, (first + last) // 2, last))
            frames = [(second, frames_dir / f"{second:06}.jpg") for second in sorted(frame_times)]
            frames = [(second, frame) for second, frame in frames if frame.is_file()]
            share = len(chapter["beats"]) / beat_count
            draft = model.draft(
                source_ranges, chapter, story,
                aim * share, round(words * share), feedback,
                frames=frames, duration=job["duration"],
            )
            write_json(path, draft.model_dump())
        for beat_number, beat in enumerate(chapter["beats"], 1):
            sources = [by_id[key] for key in beat["event_ids"]]
            prefix = f"chapter_{number + 1:02}_beat_{beat_number:02}"
            lines = [line for line in draft.segments if line.beat == beat_number]
            chapter_segments = script_segments(model.Script(segments=lines), sources, prefix) if lines else []
            if not chapter_segments:
                source = max(sources, key=lambda event: event["end"] - event["start"])
                candidate = model.Candidate(start=source["start"], end=source["end"], event_id=source["id"])
                fallback = model.Script(segments=[model.ScriptLine(text=beat["description"], candidates=[candidate])])
                chapter_segments = script_segments(fallback, sources, prefix)
            if not chapter_segments:
                raise ValueError(f"第 {number + 1} 章第 {beat_number} 个转折缺少可用画面")
            for segment in chapter_segments:
                segment.update(chapter=number, beat=beat_number)
            segments.extend(chapter_segments)
    return segments


def fit_narration(project, segment, path, *, full_source=False):
    try:
        return voice_segment(project, segment, full_source=full_source)
    except ShortFootage as error:
        if path.is_file():
            text = read_json(path)["text"]
        else:
            text = model.shorten(segment["text"], error.needed, error.available).strip()
            write_json(path, {"text": text})
        return voice_segment(project, {**segment, "text": text}, full_source=full_source)


def write_timeline(project, job, directory, events, chapters, version, *, keep_script=False, dialogue=()):
    target = job["settings"]["target_minutes"] * 60
    aim = max(MIN_RUNTIME + 12, min(MAX_RUNTIME - 12, target))
    words = round(aim / 60 * 145)
    tolerance = max(30, target * 0.07)
    feedback = ""
    best = None
    for attempt in range(1, 2 if keep_script else 4):
        draft = draft_segments(project, job, directory, events, chapters, aim, words, feedback, attempt, dialogue)
        segments = []
        for index, segment in enumerate(draft):
            progress(project, job, "audio", index, len(draft))
            path = directory / f"rewrite-{attempt}-{segment['id']}.json"
            try:
                segments.append(fit_narration(project, segment, path))
            except ShortFootage:
                continue

        covered = {(segment["chapter"], segment["beat"]) for segment in segments}
        required = {(segment["chapter"], segment["beat"]) for segment in draft}
        if missing := required - covered:
            number, beat = min(missing)
            raise ValueError(f"第 {number + 1} 章第 {beat} 个转折的画面时长不足以匹配旁白")
        timeline = make_timeline(segments, version)
        duration = timeline["duration"]
        if MIN_RUNTIME <= duration <= MAX_RUNTIME:
            if best is None or abs(duration - target) < abs(best["duration"] - target):
                best = timeline
            if abs(duration - target) <= tolerance:
                break
        actual_words = sum(len(segment["text"].split()) for segment in segments)
        words = round(actual_words * aim / duration) if duration else words
        feedback = (
            f"The usable narration measured {duration:.0f} seconds after synthesis. "
            "When lengthening, explain the key decisions and consequences more clearly and use enough "
            "distinct footage to cover them. When shortening, cut repetition within beats without losing "
            "the explanation that connects one beat to the next."
        )
    if best is None:
        raise ValueError(f"配音后未得到 10–20 分钟的成片时间线，最后一轮为 {duration / 60:.1f} 分钟")
    publish_timeline(project, best)
    return best


def write_selected_timeline(project, job, directory, selection, dialogue, version):
    events = [{**segment, "id": f"event_{segment['segment_id'] + 1:03}"}
              for segment in selection["selected_segments"]]
    narration = []
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
    for index, segment in enumerate(draft):
        progress(project, job, "audio", index, len(draft))
        segments.append(fit_narration(
            project, segment, directory / f"selected-rewrite-{index}.json", full_source=True,
        ))
    timeline = make_timeline(segments, version)
    timeline["duration_budget"] = selection["original_duration"] * selection["target_ratio"]
    publish_timeline(project, timeline)
    return timeline


def generate(project, job, movie, version, *, keep_script=False):
    current = version_path(project, version)
    if current.is_file():
        return read_json(current)
    analysis = project / "analysis"
    progress(project, job, "probe")
    media = probe(movie)
    write_json(analysis / "media.json", media)
    dialogue = read_subtitles((project / "source" / "dialogue.srt").read_bytes(), job["settings"]["subtitle_offset"])
    job["duration"] = media["duration"]
    directory = analysis / f"recap-{version}"
    directory.mkdir(exist_ok=True)
    if keep_script:
        events = read_json(analysis / "events.json")["events"]
        chapters = build_story(project, job, directory, events, dialogue, media["duration"])
        return write_timeline(project, job, directory, events, chapters, version, keep_script=True, dialogue=dialogue)
    caption_path = caption_movie(project, job, movie, dialogue)
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
    segments[index] = voice_segment(project, segment, full_source="duration_budget" in timeline)
    updated = make_timeline(segments, timeline["version"] + 1)
    if "duration_budget" in timeline:
        updated["duration_budget"] = timeline["duration_budget"]
        if updated["duration"] > updated["duration_budget"]:
            raise ValueError("请调整旁白长度，使成片时长不超过原片的 8%")
    elif not MIN_RUNTIME <= updated["duration"] <= MAX_RUNTIME:
        raise ValueError("请调整旁白长度，使成片保持在 10–20 分钟")
    publish_timeline(project, updated)
    return updated


def process(project, job):
    task = job["task"]
    movie = movie_path(project, job["movie"])
    if task["kind"] == "edit":
        job["timeline_version"] = edit(project, job, task)["version"]
        return
    if task["kind"] == "generate":
        timeline = generate(project, job, movie, task.get("version", 1), keep_script=task.get("keep_script", False))
        job["timeline_version"] = timeline["version"]
    else:
        timeline = read_json(version_path(project, task["version"]))
    render(project, movie, timeline, lambda stage, done, total: progress(project, job, stage, done, total))
    job["asset_version"] = timeline["version"]
