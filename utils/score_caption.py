#!/usr/bin/env python
"""Score caption segments with the local Qwen model server."""

import os
import re
import json
import time
import argparse
import httpx


SYSTEM_PROMPT = """
You are an intelligent video-segment scoring assistant. Your task is to evaluate
how important each target video segment is for producing a concise, faithful,
and informative video summary.

You will be given the full video's caption timeline for global context, along
with the specific target segments to score in this batch. Read the full context
first, then score each target segment while considering the video's overall
story. Do not judge a segment in isolation.

Integer score range (0 to 100):
- 0-20: meaningless, repetitive, static transition, intro/outro, or irrelevant
  to the main content;
- 21-40: minor background information whose removal barely affects the summary;
- 41-60: somewhat useful information that supports understanding of main events;
- 61-80: important event or key action suitable for the summary;
- 81-100: core event, key turning point, or essential information required to
  understand the video.

Scoring requirements:
1. Score every target segment_id; do not skip any target segment_id.
2. Each score must be an integer between 0 and 100.
3. Differentiate scores based on narrative importance rather than giving nearly
   identical scores to adjacent segments.
4. Do not score based on caption length. Focus on information value and event
   importance.
5. Use only the provided captions. Do not invent content that is not described.

Return a compact JSON object mapping each target segment_id (as a string) to its
integer score, without explanations, Markdown, or extra keys:
{"0": 35, "1": 80, "2": 15}
"""

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BATCH_SIZE = 40
MIN_BATCH_SIZE = 10
MAX_CONTEXT_CHARS = 80_000
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ScoreOutputTooLong(RuntimeError):
    pass


# extract captions
def extract_captions(record):
    if "segments" in record:
        return record["segments"]
    fps = float(record.get("target_fps", 1.0))
    source = record.get("output") or {}

    captions = []
    previous_end = 0.0
    for key, value in source.items():
        if value.strip() in {"</Silence>", "</Standby>"}:
            continue
        match = re.search(r"\d+", str(key))
        end = (int(match.group()) + 1) / fps
        caption = value.replace("</Response>", "", 1).strip()
        if caption:
            captions.append({
                "segment_id": len(captions),
                "start": round(previous_end, 3),
                "end": round(end, 3),
                "caption": caption,
            })
            previous_end = end
    return captions


def _strip_code_fence(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _normalize_scores(raw_values):
    values = [float(item) for item in raw_values]
    unit_scale = all(0.0 <= v <= 1.0 for v in values) and any(0.0 < v < 1.0 for v in values)
    scale = 1.0 if unit_scale else 100.0
    return [round(max(0.0, min(1.0, v / scale)), 2) for v in values]


def _parse_batch_scores(content, keys):
    if not content or not content.strip():
        raise ValueError("Model returned an empty response")
    parsed = json.loads(_strip_code_fence(content))
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    if all(key in parsed for key in keys):
        return _normalize_scores([parsed[key] for key in keys])
    if isinstance(parsed.get("scores"), list) and len(parsed["scores"]) == len(keys):
        return _normalize_scores(parsed["scores"])
    missing = [key for key in keys if key not in parsed]
    raise ValueError(f"Missing scores for segment_ids: {missing[:10]}")


def _batch_schema(keys):
    return {
        "type": "object",
        "properties": {
            key: {"type": "integer", "minimum": 0, "maximum": 100}
            for key in keys
        },
        "required": keys,
        "additionalProperties": False,
    }


def _format_context(captions):
    if not captions:
        return ""
    lines = [
        f"[{c['segment_id']} {c['start']:.1f}-{c['end']:.1f}s] {c['caption'].strip()}"
        for c in captions
    ]
    text = "\n".join(lines)
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    budget = max(40, (MAX_CONTEXT_CHARS // len(captions)) - 24)
    return "\n".join(
        f"[{c['segment_id']} {c['start']:.1f}-{c['end']:.1f}s] {c['caption'].strip()[:budget]}"
        for c in captions
    )


def _format_user_message(context_text, batch, keys, *, full_movie_in_batch):
    if full_movie_in_batch:
        return (
            f"Video caption timeline ({len(batch)} segments):\n{context_text}\n\n"
            f"Score all {len(batch)} segment_ids ({keys[0]}..{keys[-1]}) as integers 0..100."
        )
    target_lines = "\n".join(
        f"[{key} {c['start']:.1f}-{c['end']:.1f}s] {c['caption'].strip()}"
        for key, c in zip(keys, batch)
    )
    return (
        f"Full video caption timeline (for global context):\n{context_text}\n\n"
        f"Target segments to score in this batch ({len(batch)} segments; "
        f"return ONLY keys {keys[0]}..{keys[-1]}):\n{target_lines}"
    )


def _score_single_batch(base_url, model, context_text, batch, keys, *, full_movie_in_batch, attempts, backoff):
    max_tokens = max(2048, len(batch) * 24)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_message(
                context_text, batch, keys, full_movie_in_batch=full_movie_in_batch,
            )},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "CaptionBatchScores", "schema": _batch_schema(keys)},
        },
    }

    last_error = None
    for attempt in range(attempts):
        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=900,
                trust_env=False,
            )
        except httpx.HTTPError as error:
            last_error = RuntimeError(f"模型评分服务连接失败: {error}")
            if attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise last_error from error

        if response.is_error:
            last_error = RuntimeError(f"模型评分服务 HTTP {response.status_code}: {response.text[:1000]}")
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise last_error

        try:
            choice = response.json()["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            if choice.get("finish_reason") == "length":
                reasoning = message.get("reasoning_content") or ""
                detail = f"content={content[:160]!r}"
                if reasoning:
                    detail += f", reasoning_chars={len(reasoning)}"
                raise ScoreOutputTooLong(f"评分输出超过长度限制（{max_tokens} tokens，{detail}）")
            return _parse_batch_scores(content, keys)
        except ScoreOutputTooLong:
            raise
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = RuntimeError(f"模型返回评分解析失败: {error}")
            if attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise last_error from error

    raise last_error


# Score captions with the same Qwen model service used for video understanding
def score(captions, *, batch_size=None, max_retries=None, backoff=1.0):
    if not captions:
        return []
    base_url = os.environ.get("RECAP_MODEL_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("RECAP_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    attempts = (
        max(1, int(os.environ.get("RECAP_MODEL_MAX_RETRIES", "3")))
        if max_retries is None
        else max(1, max_retries)
    )
    chunk_size = (
        max(1, int(os.environ.get("RECAP_SCORE_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))))
        if batch_size is None
        else max(1, batch_size)
    )

    raw_keys = [str(caption.get("segment_id", index)) for index, caption in enumerate(captions)]
    if len(set(raw_keys)) != len(raw_keys):
        raw_keys = [str(index) for index in range(len(captions))]

    context_text = _format_context(captions)
    scores = []
    pending = [
        (offset, min(offset + chunk_size, len(captions)))
        for offset in range(0, len(captions), chunk_size)
    ]
    pending.reverse()

    while pending:
        start_idx, end_idx = pending.pop()
        batch = captions[start_idx:end_idx]
        keys = raw_keys[start_idx:end_idx]
        try:
            batch_scores = _score_single_batch(
                base_url, model, context_text, batch, keys,
                full_movie_in_batch=(len(batch) == len(captions)),
                attempts=attempts, backoff=backoff,
            )
        except ScoreOutputTooLong:
            if len(batch) < MIN_BATCH_SIZE * 2:
                raise
            mid = (start_idx + end_idx) // 2
            pending.append((mid, end_idx))
            pending.append((start_idx, mid))
            continue
        scores.extend(batch_scores)

    return [
        {
            "segment_id": caption["segment_id"],
            "start": caption["start"],
            "end": caption["end"],
            "caption": caption["caption"],
            "score": score_value,
        }
        for caption, score_value in zip(captions, scores)
    ]


def score_record(data):
    return {
        "video_path": data.get("video_path", ""),
        "duration": data.get("duration"),
        "fps": data.get("target_fps", 1.0),
        "segments": score(extract_captions(data)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as input_file:
        data = json.load(input_file)
    result = score_record(data)
    print(f"Scored {len(result['segments'])} captions: {result['video_path']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=4)
    print(f"Saved JSON: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
