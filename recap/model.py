import base64
import json
import os
from math import ceil

import httpx
from pydantic import BaseModel, Field

MODEL = os.environ.get("RECAP_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_URL = os.environ.get("RECAP_MODEL_URL", "http://127.0.0.1:8001/v1").rstrip("/")


class Range(BaseModel):
    start: float = Field(allow_inf_nan=False)
    end: float = Field(allow_inf_nan=False)


class Caption(Range):
    caption: str = Field(min_length=1, max_length=900)


class CaptionBatch(BaseModel):
    segments: list[Caption]


class Beat(BaseModel):
    description: str = Field(max_length=400)
    event_ids: list[str] = Field(min_length=1, max_length=5)


class Chapter(BaseModel):
    synopsis: str = Field(max_length=800)
    beats: list[Beat] = Field(max_length=2)


class Candidate(Range):
    event_id: str


class ScriptLine(BaseModel):
    text: str
    candidates: list[Candidate] = Field(min_length=1)


class Script(BaseModel):
    segments: list[ScriptLine] = Field(min_length=1)


class DraftLine(ScriptLine):
    beat: int = Field(ge=1, le=4)
    text: str = Field(max_length=360)
    candidates: list[Candidate] = Field(min_length=1, max_length=3)
    use_all_candidates: bool = False


class Draft(Script):
    segments: list[DraftLine] = Field(min_length=1, max_length=12)


class Narration(BaseModel):
    text: str = Field(min_length=1)


class NarrationBatch(BaseModel):
    segments: list[Narration]


class ModelOutputTooLong(RuntimeError):
    pass


def ask(schema, prompt, data, *, frames=None, duration=None, max_tokens=4096):
    content = [{"type": "text", "text": f"{prompt}\n\n{json.dumps(data, ensure_ascii=False)}"}]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
    }
    if frames:
        jpegs = ",".join(base64.b64encode(path.read_bytes()).decode("ascii") for _, path in frames)
        content.insert(0, {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{jpegs}"}})
        payload["media_io_kwargs"] = {"video": {
            "fps": 1.0, "num_frames": -1,
            "frames_indices": [index for index, _ in frames], "duration": duration,
            "total_num_frames": ceil(duration), "do_sample_frames": False,
        }}
    response = httpx.post(f"{MODEL_URL}/chat/completions", json=payload, timeout=900, trust_env=False)
    if response.is_error:
        raise RuntimeError(f"模型服务 HTTP {response.status_code}: {response.text[:1000]}")
    choice = response.json()["choices"][0]
    if choice["finish_reason"] == "length":
        raise ModelOutputTooLong(f"模型输出超过长度限制（{schema.__name__}，{max_tokens} tokens）")
    return schema.model_validate_json(choice["message"]["content"])


def caption_batch(frames, duration, start, end, dialogue):
    return ask(CaptionBatch, (
        "Describe all events visible in this batch of video frames in the order they appear. "
        "Group consecutive shots of the same action into one event. Include brief and minor events "
        "so a later step can judge their importance. If an event crosses a batch boundary, describe "
        "the part visible here. Use two short English sentences for a brief event and up to three "
        "for a longer event. Describe the actions and visible consequences. Use descriptive labels "
        "such as 'a soldier' for people; identifying characters by name is unnecessary. "
        "Use the matching dialogue only as context. Give each event absolute start and end seconds "
        "within the batch, with non-overlapping intervals and exclusive end times."
    ), {
        "window_start": start, "window_end": end, "frame_count": len(frames),
        "sample_seconds": [index for index, _ in frames], "dialogue": dialogue,
    }, frames=frames, duration=duration, max_tokens=8192).segments


def outline(events, dialogue, earlier, final):
    return ask(Chapter, (
        "Plan this part of a chapter for a faithful movie recap. Read the timed dialogue "
        "through the end of this range before choosing what matters. Use only events from "
        "this time range as evidence. Summarize what changes and select up to two turning points in "
        "the order the film presents them. Keep flashbacks and revelations where they appear. "
        "For each, state a decision or action and its consequence, citing "
        "the event IDs covering those moments. Use the dialogue to check names, allegiances "
        "and outcomes when the rough visual notes conflict. Leave uncertain identities "
        "unspecified. Earlier sections supply context. Leave out routine activity, repeated "
        "descriptions and credits. Return no beats when this range contains only filler. "
        "In the final section, distinguish the outcome of the conflict from the protagonist's final choice."
    ), {"dialogue": dialogue, "visual_notes": events, "earlier_sections": earlier, "final": final},
        max_tokens=1800)


def narrate_selected(segments, earlier):
    result = ask(NarrationBatch, (
        "Write a continuous English movie recap for these selected scenes in input order. "
        "Return exactly one spoken paragraph for each scene. Describe the action and its consequence, "
        "and use the previous narration for continuity. Use the caption and timed dialogue as evidence. "
        "Use a name only when the evidence supports it. Stay within each scene's word limit "
        "so the narration fits the selected footage."
    ), {"scenes": segments, "previous_narration": earlier}, max_tokens=2400)
    if len(result.segments) != len(segments):
        raise ValueError("英文旁白段数与选中的片段数不一致")
    return [line.text for line in result.segments]


def draft(events, chapter, story, seconds, words, feedback, *, frames, duration):
    return ask(Draft, (
        "Write only this chapter of a continuous English movie recap. Include each of its "
        "numbered beats in ascending order and put that number in beat. Follow the order "
        "the viewer learns what happens, including flashbacks where the film tells them. "
        "Connect actions to their consequences. Use the whole-film synopsis for continuity "
        "without revealing later events early. Treat the chapter's beat plan as the factual source. "
        "Do not invent identities, motives, or outcomes. Use the event times only to select footage. "
        "Tell the story the way you would tell it to a friend. Use plain spoken English, vary "
        "sentence length, and carry the thought across adjacent segments. Once a character is "
        "introduced, use their first name or a pronoun where clear. Give specific actions "
        "and consequences. Leave out camera directions, plot-analysis labels and stock commentary. "
        "Spend more words on decisions, reversals and their consequences. Treat the supplied word count "
        "as a guide. Do not fill it with detours, repeated explanations or descriptions of individual movements. "
        "Write narration only in text. For each segment give ordered, non-overlapping candidate "
        "source intervals in absolute seconds, using only the event IDs listed for that beat. "
        "Use the timed frames and dialogue to locate the named person, action or reaction the audience "
        "needs to see. Keep each interval inside its event. For a single shot, put the strongest "
        "match first and make it long enough to cover the speech. When a paragraph moves from cause "
        "to consequence across shots, set use_all_candidates=true and give the shots in viewing order. "
        "Keep the earlier shots short enough to reach the later shot while its action is being narrated. "
        "Allow about one second of footage for every three words. Keep source times increasing "
        "across segments and avoid reusing footage. "
        "If this is the final chapter, close with the resolution."
    ), {"target_seconds": seconds, "target_words": words, "target_segments": min(12, max(1, round(words / 40))),
        "feedback": feedback,
        "story": story, "chapter": chapter, "events": events, "sample_seconds": [index for index, _ in frames]},
        frames=frames, duration=duration, max_tokens=3200)


def shorten(text, duration, available):
    return ask(Narration, (
        "Shorten this English narration while keeping the main action and result. "
        "Keep the conversational delivery and the connection to the next action. "
        "Return only the revised narration in text. Stay within the target word count."
    ), {"text": text, "target_words": max(1, int(len(text.split()) * available / duration * 0.9))},
        max_tokens=512).text
