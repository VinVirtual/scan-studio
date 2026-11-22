"""Which Python packages and tools scan-studio can use on this machine."""
from __future__ import annotations

import importlib
import shutil

REQUIRED = ["numpy", "scipy", "cv2", "PIL", "pycolmap", "open3d", "trimesh", "xatlas"]


def report() -> tuple[bool, list[str]]:
    ok = True
    lines = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            lines.append(f"ok   {mod} {getattr(m, '__version__', '')}".rstrip())
        except Exception as e:  # noqa: BLE001 - report any import failure
            ok = False
            lines.append(f"MISS {mod} ({e.__class__.__name__})")
    try:
        import cv2
        import os
        if not os.path.isfile(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")):
            ok = False
            lines.append("MISS OpenCV face models (opencv-python-headless 4.x ships them; 5.x does not)")
        else:
            lines.append("ok   OpenCV face models")
    except Exception:  # noqa: BLE001
        pass
    try:
        from .media import find_ffmpeg
        lines.append(f"ok   ffmpeg ({'system' if shutil.which('ffmpeg') else 'imageio-ffmpeg'})" if find_ffmpeg() else "")
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append(f"MISS ffmpeg: {e}")
    try:
        import pycolmap
        cuda = bool(getattr(pycolmap, "has_cuda", False))
        lines.append(f"note pycolmap CUDA: {'yes' if cuda else 'no'} (scan-studio runs on the CPU either way)")
    except Exception:  # noqa: BLE001
        pass
    from . import ar
    lines.append(f"note ar-models (only for --usdz): {'found' if ar.find() else 'not found'}")
    return ok, [ln for ln in lines if ln]
