"""The whole run: capture -> faces -> structure from motion -> dense -> clean -> orient -> scale -> texture -> GLB
-> report. Every stage is timed; the numbers land in report.json and the HTML report."""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import __version__, capture, dense, faces, geometry, glb, report, sfm, texture
from .budgets import DEFAULT, Budgets, frame_target
from .errors import FacesFound, ScanError, UsageError
from .safeio import ensure_dir, write_bytes, write_npz, write_text


@dataclass
class Options:
    input: str
    out: str
    name: str
    frames: int | None = None
    height: float | None = None
    longest: float | None = None
    width: float | None = None
    points: tuple[str, str] | None = None
    length: float | None = None
    box: str | None = None
    keep_ground: bool = False
    dense: str = "auto"                 # auto | stereo | sparse
    triangles: int = 80_000
    texture: int = 2048                 # 0 = vertex colours
    threads: int = 8
    confirm_object: bool = False
    usdz: bool = False
    fill_holes: bool = False
    budgets: Budgets = field(default_factory=lambda: DEFAULT)


class Timer:
    def __init__(self, log):
        self.stages: list[dict] = []
        self.log = log
        self._t = None
        self._name = None

    def start(self, name: str):
        self.stop()
        self._name, self._t = name, time.perf_counter()
        self.log(f"[{len(self.stages) + 1}] {name}")

    def stop(self):
        if self._name:
            self.stages.append({"stage": self._name, "seconds": round(time.perf_counter() - self._t, 2)})
            self._name = None

    @property
    def total(self) -> float:
        return round(sum(s["seconds"] for s in self.stages), 2)


def _scale_requested(o: Options) -> bool:
    return any(v is not None for v in (o.height, o.longest, o.width, o.points))


def validate(o: Options) -> None:
    given = [k for k, v in (("--height", o.height), ("--longest", o.longest), ("--width", o.width),
                            ("--points", o.points)) if v is not None]
    if len(given) > 1:
        raise UsageError(f"give one way to set the size, not {' and '.join(given)}")
    if o.points is not None and o.length is None:
        raise UsageError("--points needs --length (the real distance between the two points, in metres)")
    if o.length is not None and o.points is None:
        raise UsageError("--length goes with --points")
    # check every number now, not after an hour of reconstruction
    for flag, value in (("--height", o.height), ("--longest", o.longest), ("--width", o.width), ("--length", o.length)):
        if value is not None:
            geometry.positive_length(value, flag)
    if o.points is not None:
        geometry.parse_point(o.points[0]); geometry.parse_point(o.points[1])
    if o.box is not None:
        geometry.parse_box(o.box)
    if o.usdz:
        from . import ar
        if ar.find() is None:
            raise UsageError("--usdz needs the ar-models skill: install it in ~/.claude/skills/ar-models "
                             "or set SCAN_STUDIO_AR_MODELS to its folder")
    if o.dense not in ("auto", "stereo", "sparse"):
        raise UsageError("--dense is auto, stereo or sparse")
    if not 1000 <= o.triangles <= o.budgets.triangles_max:
        raise UsageError(f"--triangles must be between 1000 and {o.budgets.triangles_max}")
    if o.texture and (o.texture < 256 or o.texture > o.budgets.texture_max or o.texture & (o.texture - 1)):
        raise UsageError(f"--texture is a power of two from 256 to {o.budgets.texture_max}, or 0 for vertex colours")
    if not 1 <= o.threads <= 64:
        raise UsageError("--threads must be between 1 and 64")
    if o.usdz and not _scale_requested(o):
        raise UsageError("--usdz needs the real size (--height, --longest, --width or --points with --length): "
                         "AR shows objects at true scale")
    frame_target(o.frames, o.budgets)


def run(o: Options, log=print) -> dict:
    validate(o)
    cv2.setNumThreads(o.threads)
    out = ensure_dir(o.out)
    work = ensure_dir(os.path.join(out, "work"), out)
    timer = Timer(log)
    target = frame_target(o.frames, o.budgets)

    timer.start("frames: decode, score blur, choose")
    if os.path.isdir(o.input):
        cap = capture.from_photos(o.input, work, target, o.budgets, log)
    else:
        cap = capture.from_video(o.input, work, target, o.budgets, log)
    log(f"  chose {len(cap.names)} of {len(cap.scores)} candidates ({len(cap.blurry)} blurry, "
        f"{len(cap.duplicates)} still)")
    if len(cap.names) < o.budgets.frames_min:
        raise ScanError(f"only {len(cap.names)} usable frames; at least {o.budgets.frames_min} are needed. "
                        "Film slower, with more light, all the way round")

    timer.start("faces: objects only, never people")
    # every kept frame, plus the sample of other candidates checked while decoding
    per_frame = [faces.detect(f) for f in capture.load_chosen(cap)]
    per_frame += [boxes for i, boxes in sorted(cap.face_sample.items()) if i not in set(cap.chosen)]
    face_rep = faces.judge(per_frame, confirmed_object=o.confirm_object, max_boxes=o.budgets.face_boxes)
    log(f"  {face_rep.message}")
    if face_rep.verdict in ("warning", "confirmed-object"):
        log("  " + "!" * 72 + f"\n  WARNING: {face_rep.message}\n  " + "!" * 72)
    if face_rep.verdict == "refused":
        capture.forget_frames(cap)
        _clear_previous(out, o.name)
        write_text(os.path.join(out, "report.json"), json.dumps(
            {"tool": "scan-studio", "version": __version__, "refused": "faces", "faces": face_rep.to_json()}, indent=1))
        raise FacesFound(face_rep.message)

    timer.start("structure from motion (pycolmap, CPU)")
    model = sfm.run(cap.image_dir, cap.names, work, ordered=cap.kind == "video", threads=o.threads,
                    one_camera=cap.same_size, log=log)
    log(f"  placed {model.registered} of {model.total} cameras, {len(model.points):,} points, "
        f"reprojection error {model.reprojection_error:.2f} px")
    if model.registered < 0.3 * model.total:
        raise ScanError(f"structure from motion placed only {model.registered} of {model.total} cameras (under 30 %); "
                        "the model would cover only part of the object. Film slower, with more overlap and texture")

    timer.start("dense surface")
    views = model.views
    up = geometry.up_from_cameras(np.array([v.right for v in views]), np.array([v.up for v in views]))
    mesh, dense_info = _dense(model, cap.image_dir, o, up, log)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    raw_tris = len(F)

    timer.start("clean, orient, scale")
    radius = float(dense_info["region_radius"])
    plane = None
    keep = np.ones(len(F), bool)
    if not o.keep_ground:
        thr = max(dense_info.get("voxel", 0.0) * 1.5, 0.006 * radius)
        sample = V[np.random.default_rng(1).choice(len(V), min(len(V), 60_000), replace=False)]
        plane = geometry.ground_plane(sample, up, threshold=thr)
        if plane is not None:
            cams_above = np.mean([np.dot(plane[0], v.centre) + plane[1] > 0 for v in views])
            if cams_above < 0.5:
                plane = None   # a "ground" the cameras are under is a wall or a ceiling, not the table
        if plane is not None:
            up = plane[0]
            keep &= geometry.remove_below_plane(V, F, plane, margin=3 * thr)
    support = None
    if not o.keep_ground:
        centre = np.array(dense_info["region_centre"])
        sp = model.points[np.linalg.norm(model.points - centre, axis=1) < radius]
        if plane is not None:
            sp = sp[sp @ plane[0] + plane[1] > 0]
        if len(sp) >= 200:
            support = 0.08 * radius
            keep &= geometry.near_points(V, F, sp, support)
    comp = geometry.largest_component(F[keep], len(V), geometry.triangle_areas(V, F[keep]))
    idx = np.flatnonzero(keep)[comp]
    mask = np.zeros(len(F), bool)
    mask[idx] = True
    V2, F2, C2 = geometry.compact(V, F, mask, C)
    if len(F2) >= 100:
        n_before = len(F2)
        F2 = _fill_holes(V2, F2, 0.08 * radius) if o.fill_holes else F2
        log(f"  filled small holes: {len(F2) - n_before:,} triangles added")
    if len(F2) < 100:
        raise ScanError("after cleaning almost nothing is left; try --keep-ground, or film the object larger in frame")
    place, scale_info, V2, F2, C2 = place_scale_crop(o, V2, F2, C2, up, views[0].centre)
    log(f"  kept {len(F2):,} of {raw_tris:,} triangles; ground plane {'removed' if plane is not None else 'not found'}")

    timer.start("decimate and texture")
    Vd, Fd, Cd = texture.decimate(V2, F2, C2, o.triangles)
    tex_share = None
    if o.texture:
        Vt, Ft, UV, TEX, tex_share = texture.bake(Vd, Fd, views, cap.image_dir, size=o.texture,
                                                  dense_vertices=V2, dense_colors=C2, log=log)
        place = _refloor(place, Vt)
        data = glb.build(place.apply(Vt), Ft, uv=UV, texture=TEX, name=o.name)
    else:
        place = _refloor(place, Vd)
        data = glb.build(place.apply(Vd), Fd, vertex_colors=Cd, name=o.name)

    timer.start("write GLB and report")
    glb_path = write_bytes(os.path.join(out, f"{o.name}.glb"), data)
    stats = glb.validate(data)
    before = _before_cloud(V, C, place, o.budgets.viewer_points)
    write_npz(os.path.join(work, "before.npz"), **before)
    cams = place.apply(np.array([v.centre for v in views]))
    rep = {
        "tool": "scan-studio", "version": __version__, "name": o.name, "out_name": os.path.basename(out),
        "input": {"kind": cap.kind, "file": os.path.basename(os.path.normpath(o.input)), "video": cap.video,
                  "candidates": len(cap.scores), "work_size": [cap.width, cap.height]},
        "frames": {"scores": [round(s, 2) for s in cap.scores], "chosen": cap.chosen, "blurry": cap.blurry,
                   "duplicates": cap.duplicates, "thresholds": [round(t, 2) for t in cap.thresholds], "times": [round(t, 3) for t in cap.times],
                   "names": cap.names, "sources": cap.sources},
        "faces": face_rep.to_json(),
        "sfm": {"registered": model.registered, "total": model.total, "points": int(len(model.points)),
                "reprojection_error_px": round(model.reprojection_error, 3),
                "cameras": [{"name": v.name, "centre": c.tolist()} for v, c in zip(views, cams)]},
        "dense": {k: v for k, v in dense_info.items() if k not in ("region_centre",)}
                 | {"triangles": raw_tris, "voxel_m": dense_info["voxel"] * place.scale if dense_info.get("voxel")
                    and scale_info["method"] != "none" else None},
        "clean": {"ground_plane": plane is not None, "support_distance": support, "kept_triangles": int(len(F2)), "box": o.box},
        "scale": scale_info,
        "transform": place.to_json(),
        "model": {"file": os.path.basename(glb_path), **stats,
                  "size": (np.array(stats["bounds_max"]) - np.array(stats["bounds_min"])).round(5).tolist(),
                  "texture": o.texture, "texture_coverage": tex_share},
        "machine": {"cpu_threads": o.threads, "gpu": False, "python": platform.python_version()},
    }
    usdz_error = None
    if o.usdz:
        timer.start("USDZ via ar-models")
        from . import ar
        try:
            rep["usdz"] = ar.convert(glb_path, out, o.name, height=float(rep["model"]["size"][1]), log=log)
        except ScanError as e:   # keep the model and the report; say so loudly and exit non-zero at the end
            usdz_error = str(e)
            rep["usdz"] = {"error": usdz_error}
    timer.stop()
    rep["timings"] = {"stages": timer.stages, "total_seconds": timer.total}
    report.write(out, rep, data, before)
    log(f"done in {timer.total:.0f} s: {glb_path}")
    if usdz_error:
        raise ScanError(f"the GLB and the report were written, but the USDZ failed: {usdz_error}")
    return rep


def place_scale_crop(o: Options, V, F, C, up, first_cam):
    """Orient, crop to --box and scale. The box is read in the frame the model would have without it (same size
    option); then the kept part is set on the floor and centred again, and --height/--longest/--width are
    applied to the kept part, so the size given is the size of what is delivered."""
    place = geometry.place(V, up, first_cam)
    scale_info = {"method": "none", "units": "unknown (structure from motion has no scale)"}
    s = 1.0
    if _scale_requested(o):
        s, scale_info = _scale(o, place.apply(V))
    if o.box:
        lo, hi = geometry.parse_box(o.box)
        m = geometry.inside_box(place.then_scale(s).apply(V), F, lo, hi)
        if m.sum() < 100:
            raise UsageError("--box leaves almost nothing of the model; check the numbers against the report")
        V, F, C = geometry.compact(V, F, m, C)
        place = geometry.place(V, up, first_cam)
        if _scale_requested(o) and o.points is None:
            s, scale_info = _scale(o, place.apply(V))
    if _scale_requested(o):
        place = place.then_scale(s)
        out = place.apply(V)
        geometry.check_real_size(out.max(0) - out.min(0))
    return place, scale_info, V, F, C


def _clear_previous(out: str, name: str) -> None:
    """A refused run must not leave an earlier model or report in a reused output folder."""
    from . import ar
    for p in (os.path.join(out, "report.html"), os.path.join(out, f"{name}.glb"),
              os.path.join(out, "work", "before.npz"), os.path.join(out, "work", "colmap.db"),
              os.path.join(out, "work", "pairs.txt")):
        if os.path.lexists(p) and (os.path.islink(p) or os.path.isfile(p)):
            os.unlink(p)
    sparse = os.path.join(out, "work", "sparse")
    if os.path.isdir(sparse) and not os.path.islink(sparse):
        shutil.rmtree(sparse, ignore_errors=True)
    ar.remove_pair(out, name)


def _dense(model, image_dir, o: Options, up, log):
    if o.dense in ("auto", "stereo"):
        try:
            try:
                return dense.run(model, image_dir, up=up, log=log)
            except (cv2.error, np.linalg.LinAlgError, ValueError) as e:
                raise ScanError(f"stereo step failed: {e.__class__.__name__}") from e
        except ScanError as e:
            if o.dense == "stereo":
                raise
            log(f"  stereo failed ({e}); falling back to a Poisson surface on the sparse points")
    return dense.sparse_poisson(model, up=up, log=log)


def _scale(o: Options, Vout: np.ndarray):
    extent = Vout.max(0) - Vout.min(0)
    if o.points is not None:
        p1, p2 = geometry.parse_point(o.points[0]), geometry.parse_point(o.points[1])
        s = geometry.scale_from_points(p1, p2, o.length)
        return s, {"method": "points", "points": [p1.tolist(), p2.tolist()], "length_m": float(o.length),
                   "factor": s, "units": "metres"}
    s = geometry.scale_from_size(extent, height=o.height, longest=o.longest, width=o.width)
    which, val = next((k, v) for k, v in (("height", o.height), ("longest", o.longest), ("width", o.width)) if v is not None)
    return s, {"method": which, "value_m": float(val), "factor": s, "units": "metres"}


def _fill_holes(V, F, size: float) -> np.ndarray:
    """Close holes up to `size` across (stereo gaps, removed fins); bigger openings, such as the mouth of a jug,
    stay open. Only triangles are added, never vertices."""
    import open3d as o3d
    t = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(V, np.float64)), o3d.core.Tensor(np.asarray(F, np.int64)))
    filled = t.fill_holes(hole_size=float(size))
    if filled.vertex.positions.shape[0] != len(V):
        return np.asarray(F)    # never expected; keep the unfilled mesh rather than mismatch the colours
    return filled.triangle.indices.numpy().astype(np.int64)


def _refloor(place: geometry.Similarity, vertices) -> geometry.Similarity:
    """Simplifying moves vertices a little: put the final base exactly on y = 0 again."""
    floor = float(place.apply(vertices)[:, 1].min())
    return geometry.Similarity(place.R, place.t - np.array([0.0, floor, 0.0]), place.scale)


def _before_cloud(V, C, place, limit: int) -> dict:
    """A sample of the raw dense surface (before cleaning), in the output frame, for the report's before view."""
    rng = np.random.default_rng(2)
    idx = rng.choice(len(V), min(len(V), limit), replace=False)
    pts = place.apply(V[idx]).astype(np.float32)
    cols = (np.clip(C[idx], 0, 1) * 255).astype(np.uint8) if C is not None else np.full((len(idx), 3), 180, np.uint8)
    return {"points": pts, "colors": cols}
