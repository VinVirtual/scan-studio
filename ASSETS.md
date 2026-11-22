# Third-party assets and software

## In this repository

| File | What | Licence | Source |
|---|---|---|---|
| `scan_studio/viewer/viewer.bundle.js` | three.js 0.170.0 bundled with the report viewer (`viewer-src/`) | MIT (full text in `scan_studio/viewer/THREE-LICENSE.txt` and inside the bundle) | npm `three`, github.com/mrdoob/three.js |
| `scan_studio/viewer/InstrumentSerif-Regular.ttf` | display serif of the report | SIL Open Font License 1.1 (full text in `scan_studio/viewer/OFL.txt`) | github.com/google/fonts, `ofl/instrumentserif` |
| `scan_studio/viewer/JetBrainsMono.ttf` | monospace of the report | SIL Open Font License 1.1 (full text in `scan_studio/viewer/OFL.txt`) | github.com/google/fonts, `ofl/jetbrainsmono` |

The report embeds the bundle and both fonts, so every report carries the full three.js licence (in the script) and the full SIL Open Font License with the fonts' copyright lines (in the style sheet).

## Made for this repository

- **The demo jug** (`examples/jug/`): geometry, procedural materials, lights and camera path are written in `examples/jug/make_capture.py` and rendered with Blender. No scanned model, texture or photograph from anyone else is used. `capture.mp4`, `ground_truth.ply`, `cameras_gt.json` and everything in `jug-out/` were made by running that script and this pipeline on one machine.
- **Test frames** (`tests/data/jug-frames/`): 19 frames of the same render, scaled to 640x360.
- **Test faces**: drawn at test time from ellipses with OpenCV (`tests/drawn_face.py`). No photograph of any person is used anywhere.

## Installed by pip, not included

pycolmap (BSD-3-Clause, COLMAP), Open3D (MIT), OpenCV `opencv-python-headless` (Apache-2.0; its Haar face models are under the Intel open-source licence shipped with it), NumPy and SciPy (BSD-3-Clause), Pillow (MIT-CMU), trimesh (MIT), xatlas-python (MIT), imageio-ffmpeg (BSD-2-Clause; its wheel ships an ffmpeg binary under ffmpeg's own LGPL/GPL licence, which scan-studio runs as a separate program).
