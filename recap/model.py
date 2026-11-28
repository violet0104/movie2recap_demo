import base64
import json
import os
import re
import time
from math import ceil

import httpx
from pydantic import BaseModel, Field

MODEL = os.environ.get("RECAP_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_URL = os.environ.get("RECAP_MODEL_URL", "http://127.0.0.1:8001/v1").rstrip("/")
MODEL_MAX_RETRIES = max(1, int(os.environ.get("RECAP_MODEL_MAX_RETRIES", "3")))
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class Range(BaseModel):
    start: float = Field(allow_inf_nan=False)
    end: float = Field(allow_inf_nan=False)


class Caption(Range):
    caption: str = Field(min_length=1, max_length=900)


class CaptionBatch(BaseModel):
    segments: list[Caption]


class Candidate(Range):
    event_id: str


class ScriptLine(BaseModel):
    text: str
    candidates: list[Candidate] = Field(min_length=1)


class Script(BaseModel):
    segments: list[ScriptLine] = Field(min_length=1)


class Narration(BaseModel):
    text: str = Field(min_length=1)


class NarrationBatch(BaseModel):
    segments: list[Narration]


class ModelOutputTooLong(RuntimeError):
    pass


def check_model_server(timeout: float = 5.0) -> dict:
    try:
        response = httpx.get(f"{MODEL_URL}/models", timeout=timeout, trust_env=False)
        if response.is_error:
            return {"ok": False, "url": MODEL_URL, "error": f"HTTP {response.status_code}"}
        models = [item.get("id") for item in response.json().get("data", []) if isinstance(item, dict)]
        return {"ok": True, "url": MODEL_URL, "models": models, "target_present": not models or MODEL in models}
    except (httpx.HTTPError, ValueError) as error:
        return {"ok": False, "url": MODEL_URL, "error": str(error)}


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _request_completion(schema_name, json_schema, prompt, data, *, frames=None, duration=None, max_tokens=4096, max_retries=None, backoff=1.0):
    content = [{"type": "text", "text": f"{prompt}\n\n{json.dumps(data, ensure_ascii=False)}"}]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema},
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
    attempts = MODEL_MAX_RETRIES if max_retries is None else max(1, max_retries)
    last_error = None
    for attempt in range(attempts):
        try:
            response = httpx.post(f"{MODEL_URL}/chat/completions", json=payload, timeout=900, trust_env=False)
        except httpx.HTTPError as error:
            last_error = RuntimeError(f"模型服务连接失败: {error}")
            if attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise last_error from error
        if response.is_error:
            last_error = RuntimeError(f"模型服务 HTTP {response.status_code}: {response.text[:1000]}")
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise last_error
        choice = response.json()["choices"][0]
        if choice["finish_reason"] == "length":
            raise ModelOutputTooLong(f"模型输出超过长度限制（{schema_name}，{max_tokens} tokens）")
        return choice["message"]["content"]
    raise last_error


def ask(schema, prompt, data, *, frames=None, duration=None, max_tokens=4096, max_retries=None, backoff=1.0):
    raw = _request_completion(
        schema.__name__, schema.model_json_schema(), prompt, data,
        frames=frames, duration=duration, max_tokens=max_tokens,
        max_retries=max_retries, backoff=backoff,
    )
    return schema.model_validate_json(raw)


def caption_batch(frames, duration, start, end, dialogue):
    return ask(CaptionBatch, (
        "Describe all events visible in this batch of video frames in the order they appear. "
        "Group consecutive shots of the same action into one event. Include brief and minor events "
        "so a later step can judge their importance. If an event crosses a batch boundary, describe "
        "the part visible here. Use two short English sentences for a brief event and up to three "
        "for a longer event. Describe the actions and visible consequences. Use descriptive labels "
        "such as 'a soldier' for people; identifying characters by name is unnecessary. "
        "Use the matching dialogue only as context when provided. Give each event absolute start and end seconds "
        "within the batch, with non-overlapping intervals and exclusive end times."
    ), {
        "window_start": start, "window_end": end, "frame_count": len(frames),
        "sample_seconds": [index for index, _ in frames], "dialogue": dialogue,
    }, frames=frames, duration=duration, max_tokens=8192).segments


def _narration_batch_schema(keys):
    return {
        "type": "object",
        "properties": {key: {"type": "string", "minLength": 1} for key in keys},
        "required": keys,
        "additionalProperties": False,
    }


def _parse_narration_batch(raw, keys):
    parsed = json.loads(_strip_code_fence(raw))
    if isinstance(parsed, dict) and all(isinstance(parsed.get(key), str) for key in keys):
        texts = [parsed[key].strip() for key in keys]
        if all(texts):
            return texts
    if isinstance(parsed, dict) and isinstance(parsed.get("segments"), list) and len(parsed["segments"]) == len(keys):
        texts = [
            item["text"].strip()
            for item in parsed["segments"]
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if len(texts) == len(keys) and all(texts):
            return texts
    raise ValueError(f"英文旁白段数与选中的片段数不一致（期望 {len(keys)} 段）")


def narrate_selected(segments, earlier):
    if not segments:
        return []
    keys = [str(index) for index in range(1, len(segments) + 1)]
    labeled = [{"scene_id": key, **scene} for key, scene in zip(keys, segments)]
    prompt = (
        "Write a continuous English movie recap for these selected scenes in input order. "
        f"Return a JSON object mapping each scene_id ({keys[0]}..{keys[-1]}) to exactly one concise "
        "spoken English paragraph for that scene. Describe the action and its consequence, "
        "and use previous_narration for continuity. Use the caption and any provided timed dialogue as evidence. "
        "Use a name only when the evidence supports it. Stay within each scene's max_words limit "
        "so the narration fits the selected footage."
    )
    max_tokens = max(2400, len(segments) * 260)
    try:
        raw = _request_completion(
            "NarrationBatchMap",
            _narration_batch_schema(keys),
            prompt,
            {"scenes": labeled, "previous_narration": earlier},
            max_tokens=max_tokens,
        )
        return _parse_narration_batch(raw, keys)
    except (ModelOutputTooLong, ValueError, json.JSONDecodeError):
        if len(segments) > 1:
            mid = len(segments) // 2
            left = narrate_selected(segments[:mid], earlier)
            right = narrate_selected(segments[mid:], (list(earlier) + left)[-2:])
            return left + right
        raw = _request_completion(
            "NarrationBatchMap",
            _narration_batch_schema(keys),
            prompt,
            {"scenes": labeled, "previous_narration": earlier},
            max_tokens=max_tokens,
        )
        return _parse_narration_batch(raw, keys)


def shorten(text, duration, available, *, factor=0.9):
    return ask(Narration, (
        "Shorten this English narration while keeping the main action and result. "
        "Keep the conversational delivery and the connection to the next action. "
        "Return only the revised narration in text. Stay within the target word count."
    ), {"text": text, "target_words": max(1, int(len(text.split()) * available / duration * factor))},
        max_tokens=512).text
