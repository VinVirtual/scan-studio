"""Measure a finished run against a known ground truth (for test captures rendered from a model you own).

Alignment uses only the cameras: the reconstructed camera centres are matched to the true ones (rotation and
translation, no rescaling, so a wrong real size shows up as error), then refined with rigid ICP on the surface.
Distances are measured both ways between points sampled on the two surfaces. The underside of the truth (faces
that point down within 2 mm of the floor) is left out: it sat on the table and no camera could see it."""
from __future__ import annotations

import io
import json
import os

import numpy as np

from . import geometry, report
from .errors import ScanError, UsageError
from .rescale import _load_run


def kabsch(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = False):
    """Best rotation R, translation t (and scale s) with dst ~ s R src + t (Umeyama 1991)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("need two matching lists of at least three 3D points")
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()) if with_scale else 1.0
    t = md - s * R @ ms
    return R, t, s


def surface_distances(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]


def metrics(model_pts, truth_pts, threshold: float) -> dict:
    d_mt, d_tm = surface_distances(model_pts, truth_pts)
    precision = float(np.mean(d_mt < threshold))
    recall = float(np.mean(d_tm < threshold))
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = np.concatenate([d_mt, d_tm])
    return {"accuracy_mean": float(d_mt.mean()), "completeness_mean": float(d_tm.mean()),
            "chamfer_mean": float((d_mt.mean() + d_tm.mean()) / 2), "median": float(np.median(both)),
            "p95": float(np.percentile(both, 95)), "precision": precision, "recall": recall, "fscore": f}


def _truth(path: str, samples: int, seed: int):
    import trimesh
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise UsageError("--truth: not a file")
    if os.path.getsize(real) > 512 * 1024 ** 2:
        raise UsageError("--truth: file larger than 512 MB")
    mesh = trimesh.load(real, force="mesh", process=False)
    if len(mesh.faces) == 0:
        raise ScanError("--truth has no triangles")
    v = mesh.vertices @ geometry.Z_UP_TO_Y_UP.T
    f = mesh.faces
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    floor = v[:, 1].min()
    hidden = (n[:, 1] < -0.9) & (tri[:, :, 1].max(1) < floor + 0.002)
    vis = trimesh.Trimesh(v, f[~hidden], process=False)
    pts, _ = trimesh.sample.sample_surface(vis, samples, seed=seed)
    return np.asarray(pts), v


def evaluate_run(out: str, truth: str, cameras: str, *, threshold_mm: float = 2.0, samples: int = 120_000) -> dict:
    import trimesh
    if not 0.01 <= threshold_mm <= 1000:
        raise UsageError("--threshold-mm must be between 0.01 and 1000")
    out, rep, _gpath, data, before = _load_run(out)
    if rep.get("scale", {}).get("units") != "metres":
        raise UsageError("this run has no real size; set it with `scan-studio scale` first")
    try:
        with open(os.path.realpath(cameras), encoding="utf-8") as fh:
            cams = json.load(fh)
        fps = float(cams["fps"])
        true_c = {int(c["frame"]): np.array(c["centre"], float).reshape(3) @ geometry.Z_UP_TO_Y_UP.T
                  for c in cams["cameras"]}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise UsageError(f"--cameras: not a camera list like examples/jug/cameras_gt.json ({e.__class__.__name__})") from None
    if not 0 < fps <= 1000:
        raise UsageError("--cameras: fps must be between 0 and 1000")
    names = rep["frames"]["names"]
    times = rep["frames"]["times"]
    src, dst = [], []
    for c in rep["sfm"]["cameras"]:
        if c["name"] in names:
            frame = int(round(times[names.index(c["name"])] * fps)) + 1
            if frame in true_c:
                src.append(c["centre"]); dst.append(true_c[frame])
    if len(src) < 3:
        raise ScanError("fewer than three cameras could be matched to the true camera list")
    src, dst = np.array(src), np.array(dst)
    _, _, s_cams = kabsch(src, dst, with_scale=True)
    R, t, _ = kabsch(src, dst)
    cam_err = np.linalg.norm(src @ R.T + t - dst, axis=1)
    mesh = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)
    model_pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=5)
    model_pts = np.asarray(model_pts) @ R.T + t
    truth_pts, tv = _truth(truth, samples, 6)
    thr = threshold_mm / 1000
    m_cam = metrics(model_pts, truth_pts, thr)
    # rigid ICP: nearest neighbours both ways are recomputed each round; no scaling
    from scipy.spatial import cKDTree
    tree = cKDTree(truth_pts)
    P = model_pts.copy()
    for _ in range(30):
        d, j = tree.query(P)
        keep = d < np.percentile(d, 90)
        R2, t2, _ = kabsch(P[keep], truth_pts[j[keep]])
        P = P @ R2.T + t2
    m_icp = metrics(P, truth_pts, thr)
    size = np.array(rep["model"]["size"])
    true_h = float(tv[:, 1].max() - tv[:, 1].min())
    acc = {
        "chamfer_mean_mm": m_icp["chamfer_mean"] * 1000,
        "chamfer_mean_pct": 100 * m_icp["chamfer_mean"] / float(np.linalg.norm(tv.max(0) - tv.min(0))),
        "median_mm": m_icp["median"] * 1000, "p95_mm": m_icp["p95"] * 1000,
        "accuracy_mean_mm": m_icp["accuracy_mean"] * 1000, "completeness_mean_mm": m_icp["completeness_mean"] * 1000,
        "precision": m_icp["precision"], "recall": m_icp["recall"], "fscore": m_icp["fscore"],
        "fscore_threshold_mm": threshold_mm,
        "cameras_only": {k: (v * 1000 if k in ("accuracy_mean", "completeness_mean", "chamfer_mean", "median", "p95") else v)
                         for k, v in m_cam.items()},
        "camera_position_error_mm": {"mean": float(cam_err.mean() * 1000), "max": float(cam_err.max() * 1000)},
        "cameras_matched": int(len(src)),
        "height_model_mm": float(size[1] * 1000), "height_truth_mm": true_h * 1000,
        "scale_ratio": 1.0 / s_cams,
        "method": ("Aligned by matching the reconstructed camera centres to the true ones (rotation and translation, "
                   "no rescaling), refined with rigid ICP; distances between 120,000 points sampled on each surface, "
                   "both ways. The base that stood on the table is left out of the truth."),
    }
    rep["accuracy"] = acc
    report.write(out, rep, data, before)
    return acc
