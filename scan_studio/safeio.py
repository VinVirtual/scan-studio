"""Writing outputs without following planted symlinks.

Every file is written to a fresh temporary file in the same folder (O_EXCL, never followed) and then moved over
the destination with os.replace, which replaces a symlink itself instead of writing through it. Output folders
must be real folders (not symlinks) that sit where they are expected to."""
from __future__ import annotations

import os
import tempfile


class UnsafeOutput(PermissionError):
    pass


def ensure_dir(path: str, parent: str | None = None) -> str:
    """Create `path` if needed. Refuse a symlink, or a folder that resolves somewhere other than parent/name."""
    path = os.path.abspath(path)
    if os.path.islink(path):
        raise UnsafeOutput(f"output folder {os.path.basename(path)!r} is a symbolic link; refusing to write through it")
    if os.path.exists(path) and not os.path.isdir(path):
        raise UnsafeOutput(f"output folder {os.path.basename(path)!r} exists and is not a folder")
    os.makedirs(path, exist_ok=True)
    expected_parent = os.path.realpath(parent if parent is not None else os.path.dirname(path))
    if os.path.realpath(path) != os.path.join(expected_parent, os.path.basename(path)):
        raise UnsafeOutput(f"output folder {os.path.basename(path)!r} resolves outside its parent folder")
    return path


def write_bytes(path: str, data: bytes) -> str:
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    if os.path.islink(folder):
        raise UnsafeOutput(f"the folder of {os.path.basename(path)!r} is a symbolic link")
    if os.path.lexists(path) and (os.path.islink(path) or not os.path.isfile(path)):
        raise UnsafeOutput(f"{os.path.basename(path)!r} exists and is not a regular file (a symbolic link?); refusing to replace it")
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".scan-studio-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_text(path: str, text: str) -> str:
    return write_bytes(path, text.encode("utf-8"))


def write_npz(path: str, **arrays) -> str:
    """Compressed .npz through write_bytes (no pickles)."""
    import io

    import numpy as np
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return write_bytes(path, buf.getvalue())
