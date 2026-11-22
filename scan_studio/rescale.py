"""Set the real size of a finished run (scan-studio scale OUT ...) without reconstructing again.

The model keeps its base on y = 0 and its centre on the y axis, so scaling about the origin keeps it standing
where it is. The before cloud, the camera path and the report are scaled alike."""
from __future__ import annotations

import json
import os

import numpy as np

from . import geometry, glb, names, report
from .errors import ScanError, UsageError
from .safeio import write_bytes, write_npz


def _load_run(out: str):
    out = os.path.realpath(out)
    jpath = os.path.join(out, "report.json")
    if not os.path.isfile(jpath):
        raise UsageError("no report.json in that folder; give the output folder of an earlier `scan-studio run`")
    try:
        with open(jpath, encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise UsageError(f"report.json could not be read ({e.__class__.__name__})") from None
    if not isinstance(rep, dict) or rep.get("tool") != "scan-studio" or not isinstance(rep.get("model"), dict):
        raise UsageError("that report.json was not written by a finished scan-studio run")
    name = os.path.basename(str(rep["model"].get("file", "")))
    if name != rep["model"].get("file") or not name.endswith(".glb"):
        raise ScanError("report.json names a model outside its folder; refusing")
    names.check(name[:-4])
    for key in ("size", "triangles"):
        if key not in rep["model"]:
            raise UsageError(f"report.json has no model {key}; run the pipeline again")
    if not os.path.isfile(os.path.join(out, name)):
        raise UsageError(f"{name} is missing from that folder")
    gpath = os.path.join(out, name)
    with open(gpath, "rb") as fh:
        data = fh.read()
    before_path = os.path.join(out, "work", "before.npz")
    if os.path.isfile(before_path):
        with np.load(before_path, allow_pickle=False) as z:
            before = {"points": z["points"], "colors": z["colors"]}
    else:
        before = {"points": np.zeros((0, 3), np.float32), "colors": np.zeros((0, 3), np.uint8)}
    return out, rep, gpath, data, before


def scale_glb(data: bytes, s: float) -> bytes:
    """Multiply every vertex of a scan-studio GLB by s, keeping texture and colours."""
    import trimesh
    import io
    glb.validate(data)
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    for geom in scene.geometry.values():
        geom.apply_scale(s)
    out = scene.export(file_type="glb")
    glb.validate(out)
    return out


def rescale(out: str, *, height=None, longest=None, width=None, points=None, length=None, usdz=False, log=print) -> dict:
    out, rep, gpath, data, before = _load_run(out)
    size = np.array(rep["model"]["size"], float)
    previous = float(rep.get("scale", {}).get("factor", 1.0) or 1.0)
    if points is not None:
        if length is None:
            raise UsageError("--points needs --length")
        p1, p2 = geometry.parse_point(points[0]), geometry.parse_point(points[1])
        s = geometry.scale_from_points(p1, p2, length)
        info = {"method": "points", "points": [p1.tolist(), p2.tolist()], "length_m": float(length)}
    else:
        if length is not None:
            raise UsageError("--length goes with --points")
        s = geometry.scale_from_size(size, height=height, longest=longest, width=width)
        which, val = next((k, v) for k, v in (("height", height), ("longest", longest), ("width", width)) if v is not None)
        info = {"method": which, "value_m": float(val)}
    geometry.check_real_size(size * s)
    if usdz:
        from . import ar
        if ar.find() is None:
            raise UsageError("--usdz needs the ar-models skill: install it in ~/.claude/skills/ar-models "
                             "or set SCAN_STUDIO_AR_MODELS to its folder")
    # build everything first; write the files only when nothing else can fail
    new = scale_glb(data, s)
    stats = glb.validate(new)
    rep["scale"] = info | {"factor": previous * s, "units": "metres"}
    rep["model"].update(stats)
    rep["model"]["size"] = (np.array(stats["bounds_max"]) - np.array(stats["bounds_min"])).round(5).tolist()
    for c in rep.get("sfm", {}).get("cameras", []):
        c["centre"] = (np.array(c["centre"]) * s).tolist()
    t = rep.get("transform")
    if t:
        rep["transform"] = {"R": t["R"], "t": (np.array(t["t"]) * s).tolist(), "scale": t["scale"] * s}
    rep.pop("accuracy", None)   # measured at the old scale; run evaluate again
    before = {"points": (np.asarray(before["points"]) * s).astype(np.float32), "colors": before["colors"]}
    stem = os.path.splitext(os.path.basename(gpath))[0]
    if usdz:
        pending = write_bytes(os.path.join(out, f".pending-{stem}.glb"), new)
        try:
            rep["usdz"] = ar.convert(pending, out, stem, height=float(rep["model"]["size"][1]), log=log)
        finally:
            os.unlink(pending)
    write_bytes(gpath, new)
    if os.path.isdir(os.path.join(out, "work")):
        write_npz(os.path.join(out, "work", "before.npz"), **before)
    report.write(out, rep, new, before)
    log(f"scaled by {s:.6g}: the model is now {' x '.join(f'{v:.3f}' for v in rep['model']['size'])} m")
    return rep


def rewrite_report(out: str) -> str:
    """Write report.html again from report.json, the GLB and the before cloud (after a viewer update)."""
    out, rep, _gpath, data, before = _load_run(out)
    return report.write(out, rep, data, before)[1]
