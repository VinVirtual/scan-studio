"""Blur scoring and frame selection. Pure functions on arrays, so they are easy to test."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SCORE_WIDTH = 640     # frames are scored at this width, so scores compare across resolutions
THUMB = 48            # side of the tiny grey thumbnail used to spot a camera that is standing still


def blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image at a fixed width. Higher is sharper."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = grey.shape[:2]
    if w != SCORE_WIDTH:
        grey = cv2.resize(grey, (SCORE_WIDTH, max(1, round(h * SCORE_WIDTH / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey.astype(np.float32), cv2.CV_32F, ksize=3).var())


def thumbnail(image: np.ndarray) -> np.ndarray:
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def local_median(scores: np.ndarray, radius: int) -> np.ndarray:
    n = len(scores)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(scores[max(0, i - radius): i + radius + 1])
    return out


@dataclass(frozen=True)
class Selection:
    chosen: tuple[int, ...]          # indices into the candidates, in time order
    blurry: tuple[int, ...]          # rejected as blurry
    duplicates: tuple[int, ...]      # rejected because the camera had not moved
    threshold: tuple[float, ...]     # the blur threshold each candidate was held to


def select_frames(scores, target: int, *, min_relative: float = 0.45, radius: int = 12,
                  thumbs: list[np.ndarray] | None = None, still_threshold: float = 0.003) -> Selection:
    """Pick up to `target` frames that are sharp and spread evenly over the capture.

    A candidate is blurry when its score is below `min_relative` times the median score of its neighbours
    (so a dim or plain scene is not rejected as a whole, only the frames that are worse than their neighbours).
    The timeline is cut into `target` equal parts and the sharpest frame that is not blurry is taken from each.
    When thumbnails are given, a frame almost identical to the last one taken is skipped (a camera standing still
    adds nothing)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("scores must be a non-empty list of numbers")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if target < 1:
        raise ValueError("target must be at least 1")
    if thumbs is not None and len(thumbs) != len(s):
        raise ValueError("one thumbnail per score is needed")
    n = len(s)
    thr = min_relative * local_median(s, radius)
    blurry = s < thr
    edges = np.linspace(0, n, min(target, n) + 1).round().astype(int)
    chosen: list[int] = []
    dups: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        idx = [i for i in range(a, b) if not blurry[i]]
        if not idx:
            continue
        for i in sorted(idx, key=lambda k: -s[k]):
            if thumbs is not None and chosen and float(np.mean(np.abs(thumbs[i] - thumbs[chosen[-1]]))) < still_threshold:
                dups.append(i)
                continue
            chosen.append(i)
            break
    return Selection(tuple(chosen), tuple(int(i) for i in np.flatnonzero(blurry)), tuple(sorted(set(dups))),
                     tuple(float(t) for t in thr))


def spread(indices, n: int) -> float:
    """How evenly frames cover the timeline: 1.0 is perfectly even, near 0 is clumped. Used in the report."""
    idx = sorted(indices)
    if n <= 1 or len(idx) < 2:
        return 0.0
    gaps = np.diff([0, *idx, n - 1])
    ideal = (n - 1) / (len(idx) + 1)
    return float(max(0.0, 1.0 - np.std(gaps) / max(ideal, 1e-9)))
