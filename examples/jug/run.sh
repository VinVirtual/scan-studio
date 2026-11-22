#!/usr/bin/env bash
# Remake the demo from scratch: render the capture in Blender, encode it, reconstruct it, measure it.
#   ./run.sh            reconstruct and measure the committed capture.mp4 (no Blender needed)
#   ./run.sh --render   render the capture again first (about 30 to 60 minutes of CPU in Blender)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(cd "$HERE/../.." && pwd)/scan-studio"
cd "$HERE"
if [ "${1:-}" = "--render" ]; then
  BLENDER="${BLENDER:-$(command -v blender || echo /snap/bin/blender)}"
  TMP="$(mktemp -d)"
  "$BLENDER" -b --factory-startup --python-exit-code 1 -P make_capture.py -- "$TMP"
  FF="${FF:-$(command -v ffmpeg || "$HERE/../../.venv/bin/python" -I -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}"
  "$FF" -y -loglevel error -framerate 24 -i "$TMP/frames/%04d.png" -c:v libx264 -preset slow -crf 23 \
    -pix_fmt yuv420p -movflags +faststart capture.mp4
  cp "$TMP/ground_truth.ply" "$TMP/cameras_gt.json" .
  rm -rf "$TMP"
fi
# the jug is 0.1995 m tall: the one real measurement a person would take with a ruler
"$S" run capture.mp4 --out jug-out --name jug --height 0.1995 --texture 1024 --triangles 40000 --usdz
"$S" evaluate jug-out --truth ground_truth.ply --cameras cameras_gt.json --threshold-mm 2
