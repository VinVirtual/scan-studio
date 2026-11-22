"""Hard limits. Every one caps a total, not only a single item, so a large input cannot exhaust the machine."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import BudgetError, UsageError


@dataclass(frozen=True)
class Budgets:
    input_bytes: int = 4 * 1024 ** 3          # one video, or the sum of all photos
    photos: int = 400                          # photos in a photo set
    photo_pixels: int = 60_000_000             # one photo (a 50 MP phone photo fits)
    candidate_frames: int = 600                # frames decoded from a video for scoring
    candidate_pixels: int = 600 * 1280 * 720   # all decoded candidates together
    frames_min: int = 12                       # fewer than this cannot make a model
    frames_max: int = 200                      # most frames that go into reconstruction
    work_long_side: int = 1600                 # frames are scaled down to this before anything else
    face_boxes: int = 200                      # face boxes kept for the report
    triangles_max: int = 500_000               # final model
    texture_max: int = 4096                    # final texture side
    viewer_points: int = 60_000                # points shown in the report's "before" view
    report_bytes: int = 48 * 1024 ** 2         # the single-file report


DEFAULT = Budgets()


def bounded_int(name: str, value: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise UsageError(f"{name} must be a whole number, got {value!r}") from None
    if not lo <= v <= hi:
        raise UsageError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def frame_target(requested: int | None, budgets: Budgets = DEFAULT) -> int:
    """How many frames to keep: 90 unless asked, always inside the budget."""
    if requested is None:
        return min(90, budgets.frames_max)
    return bounded_int("--frames", requested, budgets.frames_min, budgets.frames_max)


def check_total(name: str, used: int, limit: int) -> None:
    if used > limit:
        raise BudgetError(f"{name}: {used:,} is over the limit of {limit:,}")


def candidate_plan(n_frames: int, width: int, height: int, budgets: Budgets = DEFAULT) -> tuple[int, int, int]:
    """Decode at most `candidate_frames` frames, scaled so their total pixels stay inside `candidate_pixels`.
    Returns (frames to decode, width, height) with even sizes."""
    if n_frames <= 0 or width <= 0 or height <= 0:
        raise UsageError("the video has no frames or no picture size")
    n = min(n_frames, budgets.candidate_frames)
    scale = min(1.0, budgets.work_long_side / max(width, height))
    per = budgets.candidate_pixels / n
    if width * height * scale * scale > per:
        scale = (per / (width * height)) ** 0.5
    w = max(2, int(width * scale) // 2 * 2)
    h = max(2, int(height * scale) // 2 * 2)
    return n, w, h


def with_overrides(**kw) -> Budgets:
    return replace(DEFAULT, **kw)
