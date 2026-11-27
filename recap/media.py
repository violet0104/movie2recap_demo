import io
import json
import os
import subprocess
import wave
from datetime import timedelta
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import srt
from PIL import Image

FFMPEG = os.environ.get("RECAP_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("RECAP_FFPROBE", "ffprobe")
ANALYSIS_SIZE = (224, 128)
FPS = 30
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS


def run(command, *, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name}: {result.stderr.strip()}")
    return result.stdout


def ffmpeg(*args, cwd=None, on_frame=None):
    command = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if on_frame is None:
        run([*command, *args], cwd=cwd)
        return
    with TemporaryFile() as errors, subprocess.Popen(
        [*command, "-stats_period", "2", "-progress", "pipe:1", *args],
        cwd=cwd, stdout=subprocess.PIPE, stderr=errors, text=True,
    ) as process:
        for line in process.stdout:
            if line.startswith("frame="):
                on_frame(int(line.partition("=")[2]))
        if process.wait():
            errors.seek(0)
            raise RuntimeError(f"{Path(FFMPEG).name}: {errors.read().decode(errors='replace').strip()}")


def probe(movie):
    info = json.loads(run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", movie]))
    video = next(
        (track for track in info["streams"]
         if track["codec_type"] == "video" and not track.get("disposition", {}).get("attached_pic")),
        None,
    )
    if video is None or float(info["format"]["duration"]) <= 0:
        raise ValueError("原片缺少有效的视频轨道")
    info["duration"] = float(info["format"]["duration"])
    info["width"], info["height"] = video["width"], video["height"]
    return info


def read_subtitles(content):
    try:
        return [
            {"index": item.index, "start": item.start.total_seconds(),
             "end": item.end.total_seconds(), "text": item.content}
            for item in srt.parse(content.decode("utf-8-sig"))
        ]
    except (srt.SRTParseError, UnicodeDecodeError) as error:
        raise ValueError("字幕须为 UTF-8 编码的有效 SRT") from error


def check_media_tools():
    errors = []
    warnings = []
    try:
        run([FFPROBE, "-version"])
    except (OSError, RuntimeError) as error:
        errors.append(f"ffprobe 不可用: {error}")
    try:
        run([FFMPEG, "-version"])
        filters = run([FFMPEG, "-hide_banner", "-filters"])
        if not any(line.split()[1] == "subtitles" for line in filters.splitlines() if len(line.split()) >= 2):
            errors.append("ffmpeg 缺少 subtitles (libass) 滤镜")
        encoders = run([FFMPEG, "-hide_banner", "-encoders"])
        encoder_names = {line.split()[1] for line in encoders.splitlines() if len(line.split()) >= 2}
        for codec in ("libx264", "aac"):
            if codec not in encoder_names:
                errors.append(f"ffmpeg 缺少 {codec} 编码器")
    except (OSError, RuntimeError) as error:
        errors.append(f"ffmpeg 不可用: {error}")
    try:
        matched = run(["fc-match", "DejaVu Sans"]).strip()
        if "DejaVu" not in matched:
            warnings.append(f"未检测到 DejaVu Sans 字体，当前回退为: {matched}")
    except (OSError, RuntimeError):
        warnings.append("未能通过 fc-match 检查 DejaVu Sans 字体")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def window_dialogue(dialogue, start, end):
    return [
        {**line, "start": max(line["start"], start), "end": min(line["end"], end)}
        for line in dialogue if line["start"] < end and line["end"] > start
    ]


def fit_video(width, height):
    return (
        f"scale=w='trunc(iw*sar/2)*2':h=ih,setsar=1,"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


class AnalysisFrames:
    def __init__(self, movie, directory, duration):
        self.movie = movie
        self.directory = directory
        self.total = ceil(duration)
        self.count = 0
        self.process = None
        self.directory.mkdir(exist_ok=True)
        while self.count < self.total and self.path(self.count).is_file():
            self.count += 1

    def path(self, index):
        return self.directory / f"{index:06}.jpg"

    def __enter__(self):
        self.errors = TemporaryFile()
        return self

    def __exit__(self, *exc):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.stdout.close()
            self.process.wait()
        self.errors.close()

    def through(self, stop):
        stop = min(stop, self.total)
        if self.count >= stop:
            return stop
        width, height = ANALYSIS_SIZE
        if self.process is None:
            self.process = subprocess.Popen(
                [FFMPEG, "-nostdin", "-loglevel", "error", "-ss", str(self.count),
                 "-i", self.movie, "-map", "0:V:0", "-an",
                 "-vf", f"fps=1:start_time=0:round=up:eof_action=pass,{fit_video(width, height)}",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads:v", "1", "pipe:1"],
                stdout=subprocess.PIPE, stderr=self.errors,
            )
        size = width * height * 3
        while self.count < stop:
            data = self.process.stdout.read(size)
            if len(data) < size:
                if self.process.wait():
                    self.errors.seek(0)
                    raise RuntimeError(self.errors.read().decode(errors="replace").strip())
                self.total = self.count
                break
            image = Image.frombytes("RGB", ANALYSIS_SIZE, data)
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=78)
            path = self.path(self.count)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded.getvalue())
            temporary.replace(path)
            self.count += 1
        return self.count


def preview_clip(movie, start, end, destination):
    ffmpeg(
        "-ss", f"{start:.6f}", "-i", movie, "-t", f"{end - start:.6f}", "-map", "0:V:0", "-map", "0:a:0?",
        "-vf", f"fps={FPS},{fit_video(960, 540)}", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", destination,
    )


def render(project, movie, timeline, report):
    destination = project / "exports" / str(timeline["version"])
    if destination.is_dir():
        return
    # The render only refers to relative scratch filenames in FFmpeg filters and the concat list.
    with TemporaryDirectory(dir=destination.parent) as scratch:
        work = Path(scratch)
        output = work / "output"
        output.mkdir()
        subtitles = []
        clips = []
        total_clips = sum(len(segment["clips"]) for segment in timeline["segments"])
        report("render", 0, total_clips)
        narration_file = work / "narration.wav"
        sample_cursor = 0
        with wave.open(str(narration_file), "wb") as narration:
            narration.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            for segment in timeline["segments"]:
                for cue in segment["cues"]:
                    start = cue["start_sample"]
                    end = cue["end_sample"]
                    narration.writeframes(b"\0\0" * (start - sample_cursor))
                    with wave.open(str(project / "audio" / cue["audio"]), "rb") as audio:
                        audio.setpos(cue.get("audio_offset", 0))
                        narration.writeframes(audio.readframes(cue["samples"]))
                    subtitles.append(srt.Subtitle(
                        index=len(subtitles) + 1, start=sample_time(start), end=sample_time(end),
                        content=cue["text"],
                    ))
                    sample_cursor = end
                for clip in segment["clips"]:
                    filename = f"shot-{len(clips):05}.mp4"
                    ffmpeg(
                        "-ss", f"{clip['start']:.6f}", "-i", movie, "-map", "0:V:0", "-an",
                        "-vf", f"fps={FPS},{fit_video(1280, 720)},tpad=stop_mode=clone:stop_duration=1",
                        "-frames:v", str(clip["frames"]), "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "17", "-pix_fmt", "yuv420p", work / filename,
                    )
                    clips.append(f"file {filename}\n")
                    report("render", len(clips), total_clips)
            narration.writeframes(b"\0\0" * (timeline["samples"] - sample_cursor))
        (work / "shots.txt").write_text("".join(clips), encoding="utf-8")
        (output / "recap.srt").write_text(srt.compose(subtitles), encoding="utf-8")
        duration = timeline["duration"]
        report("encode", 0, duration)
        ffmpeg(
            "-f", "concat", "-safe", "1", "-i", "shots.txt", "-i", "narration.wav",
            "-vf", f"setpts=N/({FPS}*TB),"
            "subtitles=output/recap.srt:force_style='FontName=DejaVu Sans,FontSize=18,Outline=2,MarginV=16'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "output/recap.mp4", cwd=work,
            on_frame=lambda frame: report("encode", min(frame / FPS, duration), duration),
        )
        output.replace(destination)


def sample_time(samples):
    return timedelta(seconds=samples / SAMPLE_RATE)
