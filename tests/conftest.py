import os
import subprocess

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def textured(h=360, w=640, seed=0):
    """A busy, sharp synthetic picture (random blocks and lines), no faces."""
    rng = np.random.default_rng(seed)
    img = (rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    for _ in range(40):
        p1 = tuple(int(v) for v in rng.integers(0, [w, h]))
        p2 = tuple(int(v) for v in rng.integers(0, [w, h]))
        cv2.line(img, p1, p2, tuple(int(c) for c in rng.integers(0, 255, 3)), 2)
    return img


def blurred(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def ffmpeg():
    from scan_studio.media import find_ffmpeg
    return find_ffmpeg()


def write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), check=True)
    return path


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


# No silent skips: a test that skips (a missing package, a missing tool) fails the whole run.
_skipped = []


def pytest_runtest_logreport(report):
    if report.skipped:
        _skipped.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if _skipped and exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    if _skipped:
        terminalreporter.write_line(f"FAILED: {len(_skipped)} test(s) skipped; every test must run: " + ", ".join(_skipped))
