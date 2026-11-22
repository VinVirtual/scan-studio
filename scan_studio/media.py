"""Reading the capture: a video through ffmpeg, or a folder of photos. Every input is size-checked first."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .budgets import DEFAULT, Budgets, candidate_plan, check_total
from .errors import BudgetError, ScanError, UsageError

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
PHOTO_EXT = {".jpg", ".jpeg", ".png"}


def find_ffmpeg() -> str:
    """$FF, else ffmpeg on PATH, else the binary that the imageio-ffmpeg package ships."""
    env = os.environ.get("FF")
    if env:
        if not (os.path.isfile(env) and os.access(env, os.X_OK)):
            raise ScanError(f"FF={env!r} is not an executable file")
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any failure means "not available"
        raise ScanError("ffmpeg not found: install ffmpeg, or `pip install imageio-ffmpeg` in the venv, "
                        "or set FF=/path/to/ffmpeg") from None


def _regular_file(path: str, what: str) -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise UsageError(f"{what}: not a usable path")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", path) and not os.path.exists(path):
        raise UsageError(f"{what}: only local files are read, not URLs ({path.split(':', 1)[0]}:)")
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise UsageError(f"{what}: {os.path.basename(path)!r} is not a file")
    return real


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    rotation: int

    @property
    def frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))


def parse_probe(stderr: str) -> tuple[int, int, float, float, int]:
    """Read width, height, fps, duration and rotation from `ffmpeg -i` output (ffprobe is not always there)."""
    dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    # the first video stream that is not cover art
    vid = next((m for m in re.finditer(r"Stream #[^\n]*Video:[^\n]*?(\d{2,5})x(\d{2,5})[^\n]*", stderr)
                if "attached pic" not in m.group(0)), None)
    if not dur or not vid:
        raise ScanError("this file has no video stream that ffmpeg can read")
    line = vid.group(0)
    fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line) or re.search(r"(\d+(?:\.\d+)?)\s*tbr", line)
    fps = float(fps_m.group(1)) if fps_m else 30.0
    duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))
    rot_m = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", stderr) or re.search(r"rotate\s*:\s*(-?\d+)", stderr)
    rotation = int(round(float(rot_m.group(1)))) % 360 if rot_m else 0
    w, h = int(vid.group(1)), int(vid.group(2))
    if rotation in (90, 270):
        w, h = h, w   # ffmpeg turns the picture upright when decoding
    if not (0 < fps <= 1000) or duration <= 0:
        raise ScanError("the video reports no frame rate or no length")
    return w, h, fps, duration, rotation


def probe_video(path: str, budgets: Budgets = DEFAULT) -> VideoInfo:
    real = _regular_file(path, "video")
    if os.path.splitext(real)[1].lower() not in VIDEO_EXT:
        raise UsageError(f"video: expected one of {', '.join(sorted(VIDEO_EXT))}")
    check_total("video size in bytes", os.path.getsize(real), budgets.input_bytes)
    ff = find_ffmpeg()
    proc = subprocess.run([ff, "-hide_banner", "-nostdin", "-protocol_whitelist", "file,pipe", "-i", real],
                          capture_output=True, text=True, timeout=60)
    w, h, fps, duration, rot = parse_probe(proc.stderr)
    return VideoInfo(real, w, h, fps, duration, rot)


def ffmpeg_version(ff: str) -> tuple[int, int]:
    """(major, minor) of an ffmpeg binary; (0, 0) when it cannot be read (then the oldest options are used)."""
    try:
        out = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    m = re.search(r"version\s+n?(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def passthrough_args(version: tuple[int, int]) -> list[str]:
    """Keep every selected frame as it is: -fps_mode exists from ffmpeg 5.1, older ones use -vsync."""
    return ["-fps_mode", "passthrough"] if version >= (5, 1) else ["-vsync", "0"]


DECODE_SECONDS = 1800   # wall-clock limit for decoding one video


def decode_candidates(info: VideoInfo, budgets: Budgets = DEFAULT):
    """Yield (time in seconds, RGB frame) for evenly spaced candidates, inside the frame and pixel budgets."""
    n, w, h = candidate_plan(info.frames, info.width, info.height, budgets)
    step = max(1, int(np.ceil(info.frames / n)))
    ff = find_ffmpeg()
    vf = f"select='not(mod(n\\,{step}))',scale={w}:{h}:flags=area"
    cmd = [ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-protocol_whitelist", "file,pipe",
           "-i", info.path, "-vf", vf, *passthrough_args(ffmpeg_version(ff)), "-f", "rawvideo", "-pix_fmt", "rgb24",
           "pipe:1"]
    size = w * h * 3
    # stderr goes to a temporary file, so a chatty ffmpeg can never block on a full pipe
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
    count = 0
    start = time.monotonic()
    try:
        while count < n:
            if time.monotonic() - start > DECODE_SECONDS:
                raise ScanError(f"decoding took longer than {DECODE_SECONDS} s; trim the video")
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield (count * step) / info.fps, frame
            count += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
        errf.seek(0)
        err = errf.read()[-400:].decode("utf-8", "replace")
        errf.close()
    if count == 0:
        raise ScanError(f"ffmpeg decoded no frames: {err.strip() or 'unknown error'}")


def list_photos(folder: str, budgets: Budgets = DEFAULT) -> list[str]:
    real = os.path.realpath(folder)
    if not os.path.isdir(real):
        raise UsageError(f"photos: {os.path.basename(folder)!r} is not a folder")
    names = sorted(n for n in os.listdir(real) if os.path.splitext(n)[1].lower() in PHOTO_EXT and not n.startswith("."))
    # only regular files that really live in the folder (a symlink pointing elsewhere is skipped)
    paths = [os.path.join(real, n) for n in names if os.path.isfile(os.path.join(real, n))
             and os.path.dirname(os.path.realpath(os.path.join(real, n))) == real]
    if not paths:
        raise UsageError("photos: the folder has no .jpg, .jpeg or .png files")
    if len(paths) > budgets.photos:
        raise BudgetError(f"photos: {len(paths)} files is over the limit of {budgets.photos}")
    check_total("photo folder size in bytes", sum(os.path.getsize(p) for p in paths), budgets.input_bytes)
    return paths


def load_photo(path: str, budgets: Budgets = DEFAULT, long_side: int | None = None) -> np.ndarray:
    """Open one photo, check its pixel count before decoding it, turn it upright, scale it to the work size."""
    name = os.path.basename(path)
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w * h > budgets.photo_pixels:
                raise BudgetError(f"{name}: {w}x{h} is over the limit of {budgets.photo_pixels:,} pixels")
            im = ImageOps.exif_transpose(im).convert("RGB")
    except Image.DecompressionBombError as e:
        raise BudgetError(f"{name}: too many pixels") from e
    except (OSError, SyntaxError, ValueError) as e:
        raise ScanError(f"{name}: not a readable image ({e.__class__.__name__})") from None
    s = min(1.0, (long_side or budgets.work_long_side) / max(im.size))
    if s < 1.0:
        im = im.resize((max(2, round(im.size[0] * s)), max(2, round(im.size[1] * s))), Image.Resampling.LANCZOS)
    return np.asarray(im)


def photo_long_side(n_photos: int, budgets: Budgets = DEFAULT) -> int:
    """Long side for each photo so that all of them together stay inside the decoded-pixel budget
    (assuming the squarest common shape, 4:3)."""
    if n_photos <= 0:
        raise UsageError("no photos")
    fit = int((budgets.candidate_pixels / (n_photos * 0.75)) ** 0.5)
    return max(256, min(budgets.work_long_side, fit))


Image.MAX_IMAGE_PIXELS = DEFAULT.photo_pixels  # Pillow refuses anything bigger before decoding it
