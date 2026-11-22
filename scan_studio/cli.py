"""Command line: scan-studio run | scale | evaluate | faces | frames | env."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .errors import ScanError, UsageError

EPILOG = """exit codes: 0 done, 1 the capture could not be turned into a model, 2 a usage mistake,
3 over a size limit, 4 refused because people are in the frames, 5 internal error"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scan-studio", description="A phone video or photo set of an object -> a 3D model "
                                "(GLB, optional USDZ) and a one-file HTML report. CPU only.", epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"scan-studio {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="the whole pipeline", epilog=EPILOG)
    r.add_argument("input", help="a video (.mp4 .mov .m4v .mkv .webm .avi) or a folder of photos (.jpg .png)")
    r.add_argument("--out", help="output folder (default: <input name>-out next to the input)")
    r.add_argument("--name", help="file name for the model, without extension (default: from the input)")
    r.add_argument("--frames", type=int, help="frames to keep for reconstruction (default 90, 12 to 200)")
    size = r.add_argument_group("real size (pick one; without it the model has no units and no USDZ is made)")
    size.add_argument("--height", type=float, help="real height in metres, base to top")
    size.add_argument("--longest", type=float, help="real length of the longest side in metres")
    size.add_argument("--width", type=float, help="real width in metres (the wider of the two sideways sides)")
    size.add_argument("--points", nargs=2, metavar=("X,Y,Z", "X,Y,Z"),
                      help="two points picked in the report's Measure tool (from an earlier run)")
    size.add_argument("--length", type=float, help="the real distance between --points, in metres")
    r.add_argument("--box", help="keep only what is inside xmin,ymin,zmin,xmax,ymax,zmax (output metres, y up)")
    r.add_argument("--keep-ground", action="store_true", help="do not remove the table or floor plane")
    r.add_argument("--dense", default="auto", choices=["auto", "stereo", "sparse"],
                   help="stereo: depth maps fused in a volume; sparse: Poisson on sparse points; auto: stereo, else sparse")
    r.add_argument("--triangles", type=int, default=80_000, help="triangle budget of the final model (default 80000)")
    r.add_argument("--texture", type=int, default=2048, help="texture size, power of two; 0 = vertex colours (default 2048)")
    r.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="CPU threads (default up to 8)")
    r.add_argument("--confirm-object", action="store_true",
                   help="the faces found belong to an object (statue, doll, print) and no person is in the frames")
    r.add_argument("--usdz", action="store_true", help="also make a USDZ for iPhone AR with the ar-models skill")

    s = sub.add_parser("scale", help="set the real size of a finished run without running it again")
    s.add_argument("out", help="the output folder of an earlier run")
    s.add_argument("--height", type=float); s.add_argument("--longest", type=float); s.add_argument("--width", type=float)
    s.add_argument("--points", nargs=2, metavar=("X,Y,Z", "X,Y,Z")); s.add_argument("--length", type=float)
    s.add_argument("--usdz", action="store_true", help="also make the USDZ with the ar-models skill")

    e = sub.add_parser("evaluate", help="measure a run against a ground-truth mesh (for test captures)")
    e.add_argument("out", help="the output folder of an earlier run")
    e.add_argument("--truth", required=True, help="ground-truth mesh (.ply .obj .stl .glb), metres, Z up")
    e.add_argument("--cameras", required=True, help="true camera centres (JSON, as examples/jug/make_capture.py writes)")
    e.add_argument("--threshold-mm", type=float, default=2.0, help="distance for the F-score (default 2 mm)")

    f = sub.add_parser("faces", help="only check images or a video for faces (objects only, never people)")
    f.add_argument("input", help="a video, a folder of photos or one image")
    f.add_argument("--json", action="store_true")

    fr = sub.add_parser("frames", help="only choose frames from a video and print the blur scores")
    fr.add_argument("input"); fr.add_argument("--frames", type=int)
    fr.add_argument("--json", action="store_true")

    rr = sub.add_parser("report", help="write report.html again from an output folder (after updating scan-studio)")
    rr.add_argument("out")

    sub.add_parser("env", help="which packages and tools are present")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "run":
            return _run(args)
        if args.cmd == "scale":
            from .rescale import rescale
            rescale(args.out, height=args.height, longest=args.longest, width=args.width,
                    points=args.points, length=args.length, usdz=args.usdz)
            return 0
        if args.cmd == "evaluate":
            from .evaluate import evaluate_run
            acc = evaluate_run(args.out, args.truth, args.cameras, threshold_mm=args.threshold_mm)
            print(json.dumps(acc, indent=1))
            return 0
        if args.cmd == "report":
            from .rescale import rewrite_report
            print(rewrite_report(args.out))
            return 0
        if args.cmd == "faces":
            return _faces(args)
        if args.cmd == "frames":
            return _frames(args)
        if args.cmd == "env":
            from .envcheck import report
            ok, lines = report()
            print("\n".join(lines))
            return 0 if ok else 1
    except ScanError as e:
        print(f"scan-studio: {e}", file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        return 130
    except PermissionError as e:
        print(f"scan-studio: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - last resort: one line, no traceback dump of user paths
        print(f"scan-studio: internal error: {e.__class__.__name__}: {e}", file=sys.stderr)
        if os.environ.get("SCAN_STUDIO_DEBUG"):
            raise
        return 5
    return 2


def _run(args) -> int:
    from . import names
    from .pipeline import Options, run
    if not os.path.exists(args.input):
        raise UsageError(f"input not found: {os.path.basename(args.input)!r}")
    name = names.check(args.name) if args.name else names.from_input(args.input)
    src = os.path.abspath(args.input)
    out = args.out or os.path.join(os.path.dirname(src.rstrip(os.sep)), f"{name}-out")
    o = Options(input=src, out=os.path.abspath(out), name=name, frames=args.frames, height=args.height,
                longest=args.longest, width=args.width, points=tuple(args.points) if args.points else None,
                length=args.length, box=args.box, keep_ground=args.keep_ground, dense=args.dense,
                triangles=args.triangles, texture=args.texture, threads=args.threads,
                confirm_object=args.confirm_object, usdz=args.usdz)
    run(o)
    return 0


def _faces(args) -> int:
    from . import faces, media
    from .budgets import DEFAULT
    src = args.input
    if os.path.isdir(src):
        imgs = (media.load_photo(p) for p in media.list_photos(src))
    elif os.path.splitext(src)[1].lower() in media.PHOTO_EXT:
        media._regular_file(src, "image")
        imgs = iter([media.load_photo(src)])
    else:
        info = media.probe_video(src)
        imgs = (f for _, f in media.decode_candidates(info, DEFAULT))
    rep = faces.check_frames(imgs, confirmed_object=False)
    print(json.dumps(rep.to_json(), indent=1) if args.json else rep.message)
    return 4 if rep.verdict == "refused" else 0


def _frames(args) -> int:
    from . import frames, media
    from .budgets import DEFAULT, frame_target
    info = media.probe_video(args.input)
    scores, thumbs, times = [], [], []
    for t, f in media.decode_candidates(info, DEFAULT):
        scores.append(frames.blur_score(f)); thumbs.append(frames.thumbnail(f)); times.append(t)
    sel = frames.select_frames(scores, frame_target(args.frames), thumbs=thumbs)
    res = {"candidates": len(scores), "chosen": list(sel.chosen), "blurry": list(sel.blurry),
           "duplicates": list(sel.duplicates), "times": [round(times[i], 3) for i in sel.chosen],
           "spread": round(frames.spread(sel.chosen, len(scores)), 3), "scores": [round(s, 2) for s in scores]}
    if args.json:
        print(json.dumps(res))
    else:
        print(f"{len(sel.chosen)} of {len(scores)} frames chosen, {len(sel.blurry)} blurry, "
              f"{len(sel.duplicates)} still; spread {res['spread']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
