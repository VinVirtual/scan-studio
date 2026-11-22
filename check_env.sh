#!/usr/bin/env bash
# Checks what scan-studio needs and prints how to fix anything missing. Tested on Linux (Ubuntu 22.04 / 24.04).
#   SCAN_STUDIO_PYTHON     Python to use (default: ./.venv/bin/python, else python3)
#   FF                     ffmpeg to use (default: ffmpeg on PATH, else the one imageio-ffmpeg ships)
#   SCAN_STUDIO_AR_MODELS  folder of the ar-models skill, only for --usdz (default: ~/.claude/skills/ar-models)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ok=1
good() { echo "ok   $1"; }
miss() { ok=0; echo "MISS $1"; shift; for line in "$@"; do echo "       $line"; done; }
note() { echo "note $1"; shift; for line in "$@"; do echo "       $line"; done; }

if [ -n "${SCAN_STUDIO_PYTHON:-}" ]; then PY="$SCAN_STUDIO_PYTHON"
elif [ -x "$R/.venv/bin/python" ]; then PY="$R/.venv/bin/python"
else PY="$(command -v python3 || true)"; fi

# every probe runs isolated (-I) from a neutral folder, so nothing in the current folder can be imported
if [ -z "$PY" ] || ! (cd / && "$PY" -I -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)') 2>/dev/null; then
  miss "Python 3.10 or newer (3.12 tested)" "install it (e.g. uv python install 3.12), then create the venv below"
else
  good "python $(cd / && "$PY" -I -c 'import sys; print(sys.version.split()[0])') ($PY)"
  if out="$(cd / && "$PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); from scan_studio.envcheck import report; ok, lines = report(); print("\n".join(lines)); sys.exit(0 if ok else 1)' "$R" 2>&1)"; then echo "$out"
  else echo "$out"; miss "Python packages" \
    "python3.12 -m venv \"$R/.venv\"   (or: uv venv --python 3.12 \"$R/.venv\")" \
    "\"$R/.venv/bin/pip\" install -r \"$R/requirements.txt\"   (requirements-dev.txt adds pytest)"; fi
fi

if command -v nproc >/dev/null 2>&1; then good "$(nproc) CPU threads (no GPU is needed or used)"; fi
if command -v node >/dev/null 2>&1; then good "node $(node -v) (for render and --usdz)"
else note "node (only for render and --usdz)" "install Node.js 18+ from https://nodejs.org"; fi
if [ -x /snap/bin/blender ] || command -v blender >/dev/null 2>&1; then good "blender (only to remake the example capture)"
else note "blender (only to remake the example capture)" "https://www.blender.org/download/"; fi

if [ $ok = 1 ]; then echo "all good"; else echo "something is missing: fix the MISS lines above and run this again"; exit 1; fi
