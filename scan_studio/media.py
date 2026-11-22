"""Reading the capture: a video through ffmpeg, or a folder of photos. Every input is size-checked first."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
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


# only these demuxers may open a video: no concat lists, playlists or image sequences sniffed from content
FORMAT_WHITELIST = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi"
FF_THREADS = "4"
MAX_SOURCE_SIDE = 4096          # longer side of the source video, in pixels (4K phone video fits)
PROBE_SECONDS = 60
DECODE_SECONDS = 1800           # wall-clock limit for decoding one video
PHOTO_FORMATS = ("JPEG", "PNG")   # Pillow decoders allowed for photos (the JPEG one also reads MPO phone photos)


def _ff_input(ff: str, path: str) -> list[str]:
    return [ff, "-hide_banner", "-nostdin", "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMAT_WHITELIST,
            "-threads", FF_THREADS, "-i", path]


def parse_probe(stderr: str) -> tuple[int, int, float, float | None, int]:
    """Read width, height, fps, duration and rotation from `ffmpeg -i` output (ffprobe is not always there).
    Duration is None when the file does not state it (`Duration: N/A`)."""
    dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    # the first video stream that is not cover art
    vid = next((m for m in re.finditer(r"Stream #[^\n]*Video:[^\n]*?(\d{2,5})x(\d{2,5})[^\n]*", stderr)
                if "attached pic" not in m.group(0)), None)
    if not vid:
        raise ScanError("this file has no video stream that ffmpeg can read")
    line = vid.group(0)
    fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line) or re.search(r"(\d+(?:\.\d+)?)\s*tbr", line)
    fps = float(fps_m.group(1)) if fps_m else 30.0
    duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else None
    rot_m = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", stderr) or re.search(r"rotate\s*:\s*(-?\d+)", stderr)
    rotation = int(round(float(rot_m.group(1)))) % 360 if rot_m else 0
    w, h = int(vid.group(1)), int(vid.group(2))
    if rotation in (90, 270):
        w, h = h, w   # ffmpeg turns the picture upright when decoding
    if not (0 < fps <= 1000) or (duration is not None and duration <= 0):
        raise ScanError("the video reports no frame rate or no length")
    return w, h, fps, duration, rotation


def count_frames(ff: str, path: str) -> int:
    """Frames in the first video stream, by demuxing it (for files whose header states no duration)."""
    try:
        proc = subprocess.run(_ff_input(ff, path) + ["-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                              capture_output=True, text=True, timeout=PROBE_SECONDS * 5)
    except subprocess.TimeoutExpired:
        raise ScanError("counting the video's frames took too long") from None
    found = re.findall(r"frame=\s*(\d+)", proc.stderr)
    if not found or int(found[-1]) <= 0:
        raise ScanError("the video states no length and no frames could be counted")
    return int(found[-1])


def probe_video(path: str, budgets: Budgets = DEFAULT) -> VideoInfo:
    real = _regular_file(path, "video")
    if os.path.splitext(real)[1].lower() not in VIDEO_EXT:
        raise UsageError(f"video: expected one of {', '.join(sorted(VIDEO_EXT))}")
    check_total("video size in bytes", os.path.getsize(real), budgets.input_bytes)
    ff = find_ffmpeg()
    try:
        proc = subprocess.run(_ff_input(ff, real), capture_output=True, text=True, timeout=PROBE_SECONDS)
    except subprocess.TimeoutExpired:
        raise ScanError(f"ffmpeg could not read the video within {PROBE_SECONDS} s") from None
    except OSError as e:
        raise ScanError(f"ffmpeg could not start ({e.__class__.__name__})") from None
    w, h, fps, duration, rot = parse_probe(proc.stderr)
    if max(w, h) > MAX_SOURCE_SIDE:
        raise BudgetError(f"video is {w}x{h}; the longer side may be at most {MAX_SOURCE_SIDE} pixels "
                          "(film in 4K or less, or scale it down first)")
    if duration is None:
        duration = count_frames(ff, real) / fps
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


def decode_candidates(info: VideoInfo, budgets: Budgets = DEFAULT, deadline: float = DECODE_SECONDS):
    """Yield (time in seconds, RGB frame) for evenly spaced candidates, inside the frame and pixel budgets.
    A watchdog kills ffmpeg after `deadline` seconds, even while a read is blocked."""
    n, w, h = candidate_plan(info.frames, info.width, info.height, budgets)
    step = max(1, int(np.ceil(info.frames / n)))
    ff = find_ffmpeg()
    vf = f"select='not(mod(n\\,{step}))',scale={w}:{h}:flags=area"
    cmd = _ff_input(ff, info.path) + ["-loglevel", "error", "-vf", vf, *passthrough_args(ffmpeg_version(ff)),
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    size = w * h * 3
    # stderr goes to a temporary file, so a chatty ffmpeg can never block on a full pipe
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
    timed_out = threading.Event()

    def watchdog():
        timed_out.set()
        proc.kill()
    timer = threading.Timer(deadline, watchdog)
    timer.daemon = True
    timer.start()
    count = 0
    try:
        while count < n:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield (count * step) / info.fps, frame
            count += 1
    finally:
        timer.cancel()
        proc.stdout.close()
        proc.kill()
        proc.wait()
        errf.seek(0)
        err = errf.read()[-400:].decode("utf-8", "replace")
        errf.close()
    if timed_out.is_set():
        raise ScanError(f"decoding took longer than {deadline:.0f} s; trim the video")
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


def _open(path: str):
    """Open a photo with the JPEG or PNG decoder only. Pillow otherwise guesses the format from the
    content, which would let a file named .jpg reach other decoders (EPS goes to Ghostscript, for example)."""
    name = os.path.basename(path)
    try:
        return Image.open(path, formats=PHOTO_FORMATS)
    except Image.DecompressionBombError as e:
        raise BudgetError(f"{name}: too many pixels") from e
    except (OSError, SyntaxError, ValueError) as e:
        raise ScanError(f"{name}: not a readable JPEG or PNG ({e.__class__.__name__})") from None


def photo_sizes(paths: list[str], budgets: Budgets = DEFAULT) -> list[tuple[int, int]]:
    """Width and height of every photo, read from the headers only (nothing is decoded)."""
    sizes = []
    for p in paths:
        with _open(p) as im:
            w, h = im.size
        if w * h > budgets.photo_pixels:
            raise BudgetError(f"{os.path.basename(p)}: {w}x{h} is over the limit of {budgets.photo_pixels:,} pixels")
        sizes.append((w, h))
    return sizes


def load_photo(path: str, budgets: Budgets = DEFAULT, long_side: int | None = None) -> np.ndarray:
    """Open one photo, check its pixel count before decoding it, turn it upright, scale it to the work size."""
    name = os.path.basename(path)
    with _open(path) as im:
        w, h = im.size
        if w * h > budgets.photo_pixels:
            raise BudgetError(f"{name}: {w}x{h} is over the limit of {budgets.photo_pixels:,} pixels")
        try:
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Image.DecompressionBombError as e:
            raise BudgetError(f"{name}: too many pixels") from e
        except (OSError, SyntaxError, ValueError) as e:
            raise ScanError(f"{name}: not a readable image ({e.__class__.__name__})") from None
    s = min(1.0, (long_side or budgets.work_long_side) / max(im.size))
    if s < 1.0:
        im = im.resize((max(2, round(im.size[0] * s)), max(2, round(im.size[1] * s))), Image.Resampling.LANCZOS)
    return np.asarray(im)


def photo_long_side(sizes: list[tuple[int, int]], budgets: Budgets = DEFAULT) -> int:
    """The largest long side (at most the work size) at which all photos, scaled down to it, fit together inside
    the decoded-pixel budget. Uses the real sizes, so any mix of shapes is counted exactly."""
    if not sizes:
        raise UsageError("no photos")

    def total(side: int) -> float:
        return sum(w * h * min(1.0, side / max(w, h)) ** 2 for w, h in sizes)
    lo, hi = 16, budgets.work_long_side
    if total(hi) <= budgets.candidate_pixels:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if total(mid) <= budgets.candidate_pixels else (lo, mid)
    if lo < 256:
        raise BudgetError(f"{len(sizes)} photos cannot fit the pixel budget at a useful size; use fewer photos")
    return lo


Image.MAX_IMAGE_PIXELS = DEFAULT.photo_pixels  # Pillow refuses anything bigger before decoding it
