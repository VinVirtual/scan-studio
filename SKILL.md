---
name: scan-studio
description: Turn a phone video (or a set of photos) of an OBJECT into a 3D model for the web and AR, on a CPU with no GPU. Picks sharp, well spread frames (blur scoring), refuses captures with people in them (face check), runs structure from motion with pycolmap (COLMAP), builds a dense surface from two-view stereo fused in a TSDF volume, removes the table and stray bits, stands the model upright on the floor, sets the real size from a known height or two measured points, simplifies and textures it, and writes a GLB plus a one-file HTML report with a before/after 3D view, a measuring tool, the blur chart and stage timings. Optionally makes the iPhone USDZ through the ar-models skill. Use when the user wants to scan, digitise or photogrammetry an object, make a 3D model from a video or photos, turn a phone video into a GLB, 3D-scan a product, sculpture, shoe, toy or part for a website or AR, or measure or fix the size of such a scan. Objects only, never people.
---

# scan-studio

Pipeline: **frames** (ffmpeg, blur score, even spread) -> **people check** (OpenCV face detectors; refuse) -> **structure from motion** (pycolmap, CPU) -> **dense** (OpenCV semi-global stereo per frame pair, left-right and multi-view checks, Open3D TSDF fusion) -> **clean** (ground plane, far-from-features fins, largest piece, optional box) -> **orient** (up from camera axes and the ground) -> **scale** -> **simplify + texture** (quadric decimation, xatlas, colour from the frames) -> **GLB** -> **report.html** (+ optional USDZ via ar-models).

Skill folder (written `$S` below): this folder. Every command is `$S/scan-studio <command>`. `$S/scan-studio run --help` lists all options.

## Before you start

- **Objects only.** Never scan a person, a face or anyone's body, with or without consent. If the user asks for a person, say no and explain that this tool is for objects. The run stops by itself (exit code 4) when faces are found in several frames, and deletes the extracted frames. `--confirm-object` exists only for faces that belong to an object (a statue, a doll, a printed picture) with no person in the frames. Ask the user before using it and never use it for a real person.
- **Rights.** The model is a copy of the object's shape and surface. Ask whose object it is if that matters (artwork, products of other brands).
- **Size.** Structure from motion has no units. For AR, ask the user for one real measurement: the height, the longest side, or the distance between two points they can measure with a ruler. Never guess it.
- **No GPU here.** Everything runs on the CPU. The demo (89 frames of 720p video) took 96 seconds on 8 threads; larger frames, more frames and photo sets take longer. Say so before starting a long run.

## How to film (tell the user)

Walk slowly all the way round the object, phone upright, at two heights if you can (one ring at table level, one looking down). Keep the whole object in frame and filling about half of it. Diffuse light, no flash. Put it on a textured surface (a newspaper, a wooden table), not on a plain white sheet. Shiny, glass, black or plain white objects reconstruct badly. 30 to 90 seconds of video is plenty.

## Workflow

1. `$S/scan-studio check` (runs `check_env.sh`). Fix any MISS line (Python 3.12 venv with `requirements.txt`).
2. Run:
   ```
   $S/scan-studio run path/to/video.mp4 --height 0.19 [--out DIR] [--name NAME] [--usdz]
   ```
   A folder of photos works the same (`run path/to/photos/`). Output goes to `<name>-out/` next to the input: `<name>.glb`, `report.html`, `report.json`, `work/` (frames, COLMAP database and sparse model in text form). With `--usdz`: `ar/<name>.glb` and `ar/<name>.usdz`.
3. Look at the result: render the report and **Read the PNGs**:
   ```
   $S/scan-studio render <out>/report.html --png /tmp/after.png
   $S/scan-studio render <out>/report.html --png /tmp/before.png --mode before
   ```
   Check: is the object complete, upright, standing on the floor, free of table pieces and fins, and is the texture right?
4. Fix what you see:
   - table left under the object -> it was not found as a plane; use `--box xmin,ymin,zmin,xmax,ymax,zmax` (metres, y up, numbers from the report's Measure tool) or film with more of the table visible.
   - parts of the object missing -> `--keep-ground` if a flat base was cut off; otherwise film more slowly with more overlap and more light.
   - lying on its side -> the camera was held sideways for most of the video; film again upright.
   - few cameras placed -> plain or shiny surface; add texture around the object (newspaper), or use photos.
5. Real size after the fact (no new reconstruction): open the report, click **Measure**, click two points the user can also measure on the real object, then
   ```
   $S/scan-studio scale <out> --points "x,y,z" "x,y,z" --length 0.123 [--usdz]
   ```
   or `scale <out> --height 0.19`.
6. Deliver: the GLB path and size in MB, the real size in centimetres, the triangle count, the run time, and the report path. Say plainly that AR was not tested on a phone here.

## Options worth knowing

`--frames N` (default 90, 12 to 200), `--triangles N` (default 80000), `--texture 2048` (power of two, `0` for vertex colours), `--dense stereo|sparse|auto`, `--threads N`, `--keep-ground`, `--box`. `scan-studio frames <video>` and `scan-studio faces <video|folder|image>` run one stage alone. `scan-studio evaluate <out> --truth mesh.ply --cameras cams.json` measures a test capture against its ground truth. `scan-studio report <out>` rewrites the page.

Exit codes: 0 done, 1 the capture could not become a model, 2 usage mistake, 3 over a size limit, 4 people found, 5 internal error.

## Safety

- Inputs: only local files (no URLs; ffmpeg runs with `-protocol_whitelist file,pipe`). Budgets cap the total work: input bytes, photos, pixels per photo, decoded candidate frames and their total pixels, kept frames, triangles, texture size, report size.
- Outputs are written through temporary files and renamed; symlinked output files or folders are refused.
- The report makes no network requests: its content security policy allows only its own inline script and style by hash. Every name from the input is escaped.
- The launcher runs Python isolated (`-I`): nothing is imported from the current folder or `PYTHONPATH`.

## Limits (say these)

- CPU only by design: COLMAP's PatchMatch stereo needs CUDA, so dense depth comes from two-view semi-global matching instead. Detail is lower than a GPU photogrammetry tool; fine engraving and thin parts (wires, handles thinner than about 1 % of the object) may be lost or thickened.
- No Gaussian splats: no splat trainer runs in reasonable time on a CPU. `work/sparse/` holds the cameras and points in COLMAP text format, which GPU splat tools can read.
- The face check uses Haar detectors: they miss faces turned away, small or in the dark, and can fire on face-like patterns. It is a guard against mistakes, not a guarantee.
- The underside that sat on the table is not seen and stays open or flat.

## Troubleshooting

- `structure from motion placed only N cameras`: too little texture or too much blur; film slower, add texture, more light.
- `only N usable stereo pairs`: it falls back to a coarser Poisson surface on the sparse points (`dense.method` in the report says which).
- `--usdz needs the ar-models skill`: install it in `~/.claude/skills/ar-models` or set `SCAN_STUDIO_AR_MODELS`.
- `render` says Playwright not found: `npm i -g playwright-core`; no browser: install Chrome or `npx playwright-core install chromium`.
