"""Safe output names and folders."""
from __future__ import annotations

import os
import re

from .errors import UsageError

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def from_input(path: str) -> str:
    stem = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")[:64]
    return clean or "scan"


def check(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise UsageError("--name may use letters, digits, '.', '_' and '-' (up to 64, not starting with a dot)")
    return name
