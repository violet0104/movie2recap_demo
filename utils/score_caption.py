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
how important each video segment is for producing a concise, faithful, and
informative video summary.

The input contains the complete list of caption events for one video. Each event
contains:
- start: segment start time in seconds;
- end: segment end time in seconds;
- caption: a natural-language description of the segment.

Read all captions first, then score every event while considering the video's
overall context. Do not judge a segment in isolation.

Score range:
- 0.00-0.20: meaningless, repetitive, static transition, intro/outro, or
  irrelevant to the main content;
- 0.21-0.40: minor background information whose removal barely affects the
  summary;
- 0.41-0.60: somewhat useful information that only supports understanding of
  the main events;
- 0.61-0.80: important event or key action suitable for the summary;
- 0.81-1.00: core event, key turning point, or essential information required
  to understand the video.

Scoring requirements:
1. Score every caption event; do not skip any event.
2. Each score must be a number between 0.00 and 1.00, rounded to two decimals.
3. Do not give nearly identical scores to all segments merely because they are
   adjacent or similar.
4. Do not score based on caption length. Focus on information value and event
   importance.
5. Use only the provided captions. Do not invent content that is not described.
6. Preserve the exact order of the input caption_events.
7. Do not return start, end, or caption fields. The program will attach the
   original timestamps automatically.

Return JSON only, without explanations, Markdown, or any additional text:
{"scores": [0.35, 0.80, 0.15]}

The number of scores must exactly match the number of input caption_events.
"""

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {"type": "number"},
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}


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


def _parse_scores(content, expected_count):
    if not content or not content.strip():
        raise ValueError("Model returned an empty response")
    parsed = json.loads(_strip_code_fence(content))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("scores"), list):
        raise ValueError("Model response JSON must contain a 'scores' list")
    scores = parsed["scores"]
    if len(scores) != expected_count:
        raise ValueError(f"Expected {expected_count} scores, but got {len(scores)}")
    return [round(max(0.0, min(1.0, float(item))), 2) for item in scores]


# Score captions with the same Qwen model service used for video understanding
def score(captions, *, max_retries=None, backoff=1.0):
    base_url = os.environ.get("RECAP_MODEL_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("RECAP_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    attempts = (
        max(1, int(os.environ.get("RECAP_MODEL_MAX_RETRIES", "3")))
        if max_retries is None
        else max(1, max_retries)
    )
    max_tokens = max(2048, min(16384, len(captions) * 12))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(captions, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "CaptionScores", "schema": SCORES_SCHEMA},
        },
    }

    last_error = None
    scores = None
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
            if choice.get("finish_reason") == "length":
                raise ValueError(f"评分输出超过长度限制（{max_tokens} tokens）")
            content = choice["message"]["content"]
            scores = _parse_scores(content, len(captions))
            break
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = RuntimeError(f"模型返回评分解析失败: {error}")
            if attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise last_error from error

    if scores is None:
        raise last_error

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
