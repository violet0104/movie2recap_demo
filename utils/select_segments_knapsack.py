#!/usr/bin/env python
"""Select high-scoring segments with 0-1 knapsack under a duration limit."""

import argparse
import json
import math
import os
from pathlib import Path


RATIO = 0.08
RESOLUTION = 0.1
MIN_SEGMENT_DURATION = 2.0


def _min_segment_duration(min_duration=None):
    if min_duration is not None:
        return max(0.0, float(min_duration))
    return max(0.0, float(os.environ.get("RECAP_MIN_SEGMENT_SECONDS", str(MIN_SEGMENT_DURATION))))


def select_segments(captions_score, *, min_duration=None):
    segments = [seg for seg in captions_score["segments"] if seg["end"] - seg["start"] > 0]
    capacity = math.floor(captions_score["duration"] * RATIO / RESOLUTION + 1e-9)

    def _weight(segment):
        return math.ceil((segment["end"] - segment["start"]) / RESOLUTION - 1e-9)

    floor_seconds = _min_segment_duration(min_duration)
    eligible = [
        seg for seg in segments
        if seg["end"] - seg["start"] >= floor_seconds and 0 < _weight(seg) <= capacity
    ]
    if not eligible:
        eligible = [seg for seg in segments if 0 < _weight(seg) <= capacity]

    items = [
        (_weight(segment), segment["score"] * (segment["end"] - segment["start"]), segment)
        for segment in eligible
    ]

    dp = [0.0] * (capacity + 1)
    choices = [bytearray(capacity + 1) for _ in items]
    for index, (weight, item_value, segment) in enumerate(items):
        for used in range(capacity, weight - 1, -1):
            value = dp[used - weight] + item_value
            if value > dp[used]:
                dp[used] = value
                choices[index][used] = 1

    selected = []
    for index in range(len(items) - 1, -1, -1):
        if choices[index][capacity]:
            selected.append(items[index][2])
            capacity -= items[index][0]
    return sorted(selected, key=lambda segment: segment["start"])


def select_record(data):
    target_duration = data["duration"] * RATIO
    selected = select_segments(data)

    return {
        "video_path": data["video_path"],
        "original_duration": data["duration"],
        "target_ratio": RATIO,
        "target_duration": round(target_duration, 3),
        "selected_duration": round(
            sum(segment["end"] - segment["start"] for segment in selected), 3
        ),
        "selected_segments": [
            {
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "caption": segment["caption"],
                "score": segment["score"],
            }
            for segment in selected
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = select_record(json.loads(Path(args.input).read_text(encoding="utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Selected {len(result['selected_segments'])} segments, "
        f"{result['selected_duration']}/{result['target_duration']}s: {output}"
    )


if __name__ == "__main__":
    main()
