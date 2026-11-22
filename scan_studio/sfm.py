"""Structure from motion with pycolmap (COLMAP's Python package), on the CPU.

SIFT features, matching on a chosen list of image pairs (neighbours in time, plus a coarse all-against-all
grid so the end of an orbit is tied to its start), then COLMAP's incremental mapper. The result is a camera
pose for every frame it could place, the shared intrinsics, and a sparse cloud of points."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

from .errors import ScanError
from .safeio import write_text


def pair_list(n: int, *, overlap: int = 12, stride: int = 4, ordered: bool = True) -> list[tuple[int, int]]:
    """Image pairs to match. For a video (ordered): each frame against the next `overlap` frames, plus every
    `stride`-th frame against every other `stride`-th frame (loop closure). For an unordered photo set: all pairs."""
    if n < 2:
        return []
    pairs: set[tuple[int, int]] = set()
    if not ordered:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i in range(n):
        for j in range(i + 1, min(n, i + overlap + 1)):
            pairs.add((i, j))
    grid = list(range(0, n, max(1, stride)))
    for a in range(len(grid)):
        for b in range(a + 1, len(grid)):
            pairs.add((grid[a], grid[b]))
    return sorted(pairs)


@dataclass
class View:
    name: str
    K: np.ndarray            # 3x3 intrinsics (pinhole part)
    dist: np.ndarray         # OpenCV distortion coefficients (k1, k2, p1, p2)
    R: np.ndarray            # world -> camera rotation
    t: np.ndarray            # world -> camera translation
    width: int
    height: int

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    @property
    def up(self) -> np.ndarray:
        return -self.R[1]

    @property
    def right(self) -> np.ndarray:
        return self.R[0]


@dataclass
class SparseModel:
    views: list[View]
    points: np.ndarray       # (N, 3)
    colors: np.ndarray       # (N, 3) uint8
    point_views: list[list[int]]   # which views see each point (indices into views)
    reprojection_error: float
    registered: int
    total: int


def camera_to_opencv(model: str, params, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP camera parameters to an OpenCV K and distortion vector."""
    p = [float(v) for v in params]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p; fx = fy = f; d = [0, 0, 0, 0]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p; d = [0, 0, 0, 0]
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p; fx = fy = f; d = [k1, 0, 0, 0]
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p; fx = fy = f; d = [k1, k2, 0, 0]
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p; d = [k1, k2, p1, p2]
    else:
        raise ScanError(f"camera model {model} is not supported")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    return K, np.array(d, float)


def run(image_dir: str, names: list[str], work: str, *, ordered: bool, threads: int, one_camera: bool = True,
        log=print) -> SparseModel:
    try:
        import pycolmap
    except ImportError:
        raise ScanError("pycolmap is not installed: run check_env.sh for the install line") from None
    try:
        pycolmap.logging.minloglevel = 2   # errors only; the stages print their own summary
    except AttributeError:
        pass
    db = os.path.join(work, "colmap.db")
    if os.path.lexists(db):
        if os.path.isdir(db) and not os.path.islink(db):
            raise ScanError("work/colmap.db is a folder; remove it")
        os.unlink(db)
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = threads
    ext.use_gpu = False
    ext.max_image_size = 1600
    ext.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    # one shared camera for a video, or for photos that all have the same size (one phone, one lens)
    mode = pycolmap.CameraMode.SINGLE if (ordered or one_camera) else pycolmap.CameraMode.AUTO
    log(f"  features: SIFT on {len(names)} frames")
    pycolmap.extract_features(db, image_dir, image_names=names, camera_mode=mode, reader_options=reader,
                              extraction_options=ext, device=pycolmap.Device.cpu)
    pairs = pair_list(len(names), ordered=ordered)
    pairs_path = os.path.join(work, "pairs.txt")
    write_text(pairs_path, "".join(f"{names[a]} {names[b]}\n" for a, b in pairs))
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = threads
    match.use_gpu = False
    imported = pycolmap.ImportedPairingOptions()
    imported.match_list_path = pairs_path
    log(f"  matching: {len(pairs)} pairs")
    pycolmap.match_image_pairs(db, matching_options=match, pairing_options=imported, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = threads
    opts.ba_use_gpu = False
    opts.multiple_models = False
    opts.random_seed = 7
    # pycolmap writes into a fresh private folder; the finished model is then moved to work/sparse
    tmp = tempfile.mkdtemp(dir=work, prefix=".sfm-")
    log("  mapping: incremental structure from motion")
    recs = pycolmap.incremental_mapping(db, image_dir, tmp, options=opts)
    if not recs:
        raise ScanError("structure from motion placed no cameras: the frames share too few features "
                        "(plain or shiny object, too much blur, or the camera moved too fast)")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    final = os.path.join(tmp, "model")
    os.mkdir(final)
    rec.write_text(final)  # cameras.txt / images.txt / points3D.txt, readable by other tools (e.g. splat trainers)
    out = os.path.join(work, "sparse")
    if os.path.islink(out) or (os.path.lexists(out) and not os.path.isdir(out)):
        raise ScanError("work/sparse is a symbolic link or a file; refusing to replace it")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.replace(final, out)
    shutil.rmtree(tmp, ignore_errors=True)
    return to_model(rec, len(names))


def to_model(rec, total: int) -> SparseModel:
    views: list[View] = []
    id_to_view: dict[int, int] = {}
    for image_id in sorted(rec.reg_image_ids(), key=lambda i: rec.image(i).name):
        img = rec.image(image_id)
        cam = rec.camera(img.camera_id)
        K, dist = camera_to_opencv(cam.model_name if hasattr(cam, "model_name") else str(cam.model.name),
                                   cam.params, cam.width, cam.height)
        pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = np.asarray(pose.matrix())
        id_to_view[image_id] = len(views)
        views.append(View(img.name, K, dist, M[:, :3].copy(), M[:, 3].copy(), cam.width, cam.height))
    pts, cols, vis = [], [], []
    for _pid, p in rec.points3D.items():
        pts.append(np.asarray(p.xyz))
        cols.append(np.asarray(p.color))
        vis.append(sorted({id_to_view[e.image_id] for e in p.track.elements if e.image_id in id_to_view}))
    if len(views) < 3 or not pts:
        raise ScanError(f"structure from motion placed only {len(views)} cameras; the capture cannot be reconstructed")
    return SparseModel(views, np.array(pts, float), np.array(cols, np.uint8), vis,
                       float(rec.compute_mean_reprojection_error()), len(views), total)
