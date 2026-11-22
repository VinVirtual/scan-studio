"""Optional: hand the GLB to the separately installed ar-models skill for a USDZ (iPhone AR Quick Look).

scan-studio does not include or copy ar-models; it only runs its command line when it is installed.
Where it looks: $SCAN_STUDIO_AR_MODELS, else ~/.claude/skills/ar-models (the usual skill install place)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from .errors import ScanError
from .safeio import ensure_dir


def find() -> str | None:
    env = os.environ.get("SCAN_STUDIO_AR_MODELS")
    for base in ([env] if env else []) + [os.path.join(os.path.expanduser("~"), ".claude", "skills", "ar-models")]:
        exe = os.path.join(base, "ar")
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return os.path.realpath(exe)
    return None


def command(exe: str, glb_path: str, out_dir: str, name: str, height: float) -> list[str]:
    if not (0.005 <= float(height) <= 50):
        raise ScanError(f"height {height} m is not a real object size; set the size first")
    return [exe, "convert", os.path.abspath(glb_path), "--out", os.path.abspath(out_dir), "--name", name,
            "--height", f"{float(height):.5f}", "--json"]


def convert(glb_path: str, out: str, name: str, *, height: float, log=print, timeout: int = 900) -> dict:
    exe = find()
    if exe is None:
        raise ScanError("--usdz needs the ar-models skill: install it in ~/.claude/skills/ar-models "
                        "or set SCAN_STUDIO_AR_MODELS to its folder")
    if shutil.which("node") is None:
        raise ScanError("--usdz needs Node.js for the ar-models skill")
    ar_dir = ensure_dir(os.path.join(out, "ar"), out)
    cmd = command(exe, glb_path, ar_dir, name, height)
    log(f"  ar-models: {os.path.basename(exe)} convert -> ar/{name}.glb, ar/{name}.usdz")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd="/")
    except subprocess.TimeoutExpired:
        raise ScanError(f"ar-models took longer than {timeout} s") from None
    except OSError as e:
        raise ScanError(f"ar-models could not start ({e.__class__.__name__})") from None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise ScanError("ar-models could not convert the model: " + " | ".join(tail))
    try:
        summary = json.loads(proc.stdout[proc.stdout.find("{"):]) if "{" in proc.stdout else {}
    except json.JSONDecodeError:
        summary = {}
    usdz = os.path.join(ar_dir, f"{name}.usdz")
    if not os.path.isfile(usdz):
        raise ScanError("ar-models finished but wrote no .usdz")
    return {"usdz": f"ar/{name}.usdz", "glb": f"ar/{name}.glb", "bytes": os.path.getsize(usdz),
            "summary_keys": sorted(summary)[:20] if isinstance(summary, dict) else []}
