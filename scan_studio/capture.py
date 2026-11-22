"""Stage 1: from a video or a photo folder to a set of sharp, well spread frames on disk."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import faces
from . import frames as fr
from . import media
from .budgets import Budgets, check_total
from .errors import ScanError
from .safeio import ensure_dir, write_bytes


@dataclass
class Capture:
    kind: str                    # "video" | "photos"
    image_dir: str
    names: list[str]             # chosen frames, in order, as written in image_dir
    times: list[float]           # seconds into the video (video) or photo index (photos)
    sources: list[str]           # photo file names (photos), empty for a video
    scores: list[float]          # blur score of every candidate
    chosen: list[int]            # indices into the candidates
    blurry: list[int]
    duplicates: list[int]
    thresholds: list[float]
    width: int
    height: int
    video: dict | None
    same_size: bool = True
    face_sample: dict = field(default_factory=dict)   # candidate index -> face boxes, for a sample of candidates


def _save(path: str, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ScanError("could not encode a frame")
    write_bytes(path, buf.tobytes())


FACE_SAMPLE_EVERY = 3   # while decoding a video, also check every 3rd candidate for faces (not only the kept ones)


def from_video(path: str, work: str, target: int, budgets: Budgets, log=print) -> Capture:
    info = media.probe_video(path, budgets)
    log(f"  video: {info.width}x{info.height}, {info.fps:.3g} fps, {info.duration:.1f} s")
    cand_dir = ensure_dir(os.path.join(work, "candidates"), work)
    try:
        scores, thumbs, times, sample = [], [], [], {}
        size = (0, 0)
        pixels = 0
        for i, (t, rgb) in enumerate(media.decode_candidates(info, budgets)):
            pixels += rgb.shape[0] * rgb.shape[1]
            check_total("decoded pixels", pixels, budgets.candidate_pixels)
            scores.append(fr.blur_score(rgb))
            thumbs.append(fr.thumbnail(rgb))
            times.append(t)
            size = (rgb.shape[1], rgb.shape[0])
            if i % FACE_SAMPLE_EVERY == 1:
                sample[i] = faces.detect(rgb)
            _save(os.path.join(cand_dir, f"{i:05d}.jpg"), rgb)
        sel = fr.select_frames(scores, target, thumbs=thumbs)
        cap = _finish(work, cand_dir, [f"{i:05d}.jpg" for i in range(len(scores))], sel, scores, times, size, "video")
    finally:
        shutil.rmtree(cand_dir, ignore_errors=True)
    cap.face_sample = sample
    cap.video = {"width": info.width, "height": info.height, "fps": info.fps, "duration": info.duration,
                 "rotation": info.rotation}
    return cap


def from_photos(folder: str, work: str, target: int, budgets: Budgets, log=print) -> Capture:
    paths = media.list_photos(folder, budgets)
    long_side = media.photo_long_side(media.photo_sizes(paths, budgets), budgets)
    cand_dir = ensure_dir(os.path.join(work, "candidates"), work)
    try:
        scores, sizes = [], set()
        pixels = 0
        for i, p in enumerate(paths):
            rgb = media.load_photo(p, budgets, long_side)
            pixels += rgb.shape[0] * rgb.shape[1]
            check_total("decoded pixels", pixels, budgets.candidate_pixels)
            scores.append(fr.blur_score(rgb))
            sizes.add((rgb.shape[1], rgb.shape[0]))
            _save(os.path.join(cand_dir, f"{i:05d}.jpg"), rgb)
        log(f"  photos: {len(paths)}")
        # photos are not a timeline: drop the blurry ones, keep order, thin evenly if there are too many
        sel = fr.select_frames(scores, min(target, len(scores)), radius=len(scores))
        size = max(sizes) if sizes else (0, 0)
        cap = _finish(work, cand_dir, [f"{i:05d}.jpg" for i in range(len(scores))], sel, scores,
                      [float(i) for i in range(len(scores))], size, "photos")
    finally:
        shutil.rmtree(cand_dir, ignore_errors=True)
    cap.sources = [os.path.basename(paths[i]) for i in sel.chosen]
    cap.same_size = len(sizes) == 1
    return cap


def _finish(work, cand_dir, cand_names, sel, scores, times, size, kind) -> Capture:
    image_dir = ensure_dir(os.path.join(work, "images"), work)
    for n in os.listdir(image_dir):
        if n.endswith(".jpg"):
            os.unlink(os.path.join(image_dir, n))
    names = []
    for k, i in enumerate(sel.chosen):
        name = f"{k:04d}.jpg"
        os.replace(os.path.join(cand_dir, cand_names[i]), os.path.join(image_dir, name))
        names.append(name)
    return Capture(kind, image_dir, names, [float(times[i]) for i in sel.chosen], [], [float(s) for s in scores],
                   list(sel.chosen), list(sel.blurry), list(sel.duplicates), list(sel.threshold), size[0], size[1], None)


def load_chosen(cap: Capture):
    for n in cap.names:
        bgr = cv2.imread(os.path.join(cap.image_dir, n), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ScanError(f"could not read {n}")
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def forget_frames(cap: Capture) -> None:
    """Delete the extracted frames (used when a capture is refused because people are in it)."""
    shutil.rmtree(cap.image_dir, ignore_errors=True)
