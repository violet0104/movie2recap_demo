#!/usr/bin/env python
"""Score caption segments with DeepSeek."""

import os
import re
import json
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

# extract captions
def extract_captions(record):
    if "segments" in record:
        return record["segments"]
    fps = float(record.get("target_fps", 1.0))
    source = record.get("output")

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

# use Deepseek scoring captions
def score(captions):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请为任务进程设置 DEEPSEEK_API_KEY")
    response = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(captions, ensure_ascii=False)},
            ],
            "stream": False,
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
        },
        timeout=900,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError("Model returned an empty response")
    scores = json.loads(content.strip())["scores"]

    if len(scores) != len(captions):
        raise ValueError(
            f"Expected {len(captions)} scores, but got {len(scores)}"
        )

    return [
        {
            "segment_id": caption["segment_id"],
            "start": caption["start"],
            "end": caption["end"],
            "caption": caption["caption"],
            "score": round(float(score), 2),
        }
        for caption, score in zip(captions, scores)
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
