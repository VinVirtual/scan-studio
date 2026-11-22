"""Cleaning, orienting and scaling the reconstructed mesh. Plain numpy where possible, so it is testable.

Frames used here:
  model frame   COLMAP's arbitrary frame and units, the frame of the cameras.
  output frame  glTF convention: metres (once scaled), +Y up, the base on y = 0, centred on x = z = 0, and the
                side seen in the first frame facing +Z."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import UsageError

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # (x, y, z) -> (x, z, -y)


# ---------------------------------------------------------------- orientation
def up_from_cameras(rights: np.ndarray, ups: np.ndarray) -> np.ndarray:
    """World up from camera axes. Held phones keep their x axis (right) roughly level, so up is the direction
    most perpendicular to every camera's right vector: the eigenvector of sum(r r^T) with the smallest eigenvalue.
    Its sign is taken from the average camera up vector (people rarely film upside down)."""
    R = np.asarray(rights, float)
    if R.ndim != 2 or R.shape[1] != 3 or len(R) < 2:
        raise ValueError("need at least two camera right vectors")
    w, V = np.linalg.eigh(R.T @ R)
    up = V[:, 0]
    if np.dot(up, np.mean(ups, axis=0)) < 0:
        up = -up
    return up / np.linalg.norm(up)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def yaw_to_face(R_up: np.ndarray, first_cam_centre: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """After R_up (model -> Z up), turn about Z so the first camera sits on the -Y side (glTF +Z, the front)."""
    d = R_up @ (first_cam_centre - centre)
    ang = math.atan2(d[1], d[0])
    target = -math.pi / 2
    a = target - ang
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass(frozen=True)
class Similarity:
    """x_out = scale * R @ x + t."""
    R: np.ndarray
    t: np.ndarray
    scale: float = 1.0

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(pts, float) @ self.R.T) + self.t

    def apply_dir(self, d: np.ndarray) -> np.ndarray:
        return np.asarray(d, float) @ self.R.T

    def then_scale(self, s: float) -> "Similarity":
        return Similarity(self.R, self.t * s, self.scale * s)

    def to_json(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist(), "scale": self.scale}


def place(vertices: np.ndarray, up: np.ndarray, first_cam_centre: np.ndarray) -> Similarity:
    """The model -> output transform: up to +Y, first view at the front, base on y = 0, centred (no scale yet)."""
    centre = vertices.mean(axis=0)
    R1 = rotation_between(up, np.array([0, 0, 1.0]))
    R2 = yaw_to_face(R1, first_cam_centre, centre)
    R = Z_UP_TO_Y_UP @ R2 @ R1
    v = vertices @ R.T
    lo, hi = v.min(0), v.max(0)
    t = -np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return Similarity(R, t, 1.0)


# ---------------------------------------------------------------- scale
def parse_point(text: str) -> np.ndarray:
    try:
        vals = [float(v) for v in str(text).replace(";", ",").split(",")]
    except ValueError:
        raise UsageError(f"a point is three numbers 'x,y,z', got {text!r}") from None
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise UsageError(f"a point is three finite numbers 'x,y,z', got {text!r}")
    return np.array(vals)


def positive_length(value, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise UsageError(f"{name} must be a number in metres, got {value!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise UsageError(f"{name} must be a positive length in metres, got {value!r}")
    if v > 100:
        raise UsageError(f"{name} {v} m is larger than any object this tool is meant for (check the units: metres)")
    return v


def scale_from_points(p1, p2, length) -> float:
    """Factor that makes the distance between two picked points equal a measured length."""
    a, b = np.asarray(p1, float), np.asarray(p2, float)
    d = float(np.linalg.norm(a - b))
    L = positive_length(length, "--length")
    if not math.isfinite(d) or d < 1e-9:
        raise UsageError("the two points are the same point; pick two points apart from each other")
    return L / d


def scale_from_size(extent, *, height=None, longest=None, width=None) -> float:
    """Factor from a known real size. extent is the (x, y, z) size of the model in the output frame (y up)."""
    ex = np.asarray(extent, float)
    given = [k for k, v in (("height", height), ("longest", longest), ("width", width)) if v is not None]
    if len(given) != 1:
        raise UsageError("give exactly one of --height, --longest, --width (or --points with --length)")
    if height is not None:
        L, d = positive_length(height, "--height"), ex[1]
    elif longest is not None:
        L, d = positive_length(longest, "--longest"), float(ex.max())
    else:
        L, d = positive_length(width, "--width"), float(max(ex[0], ex[2]))
    if d <= 1e-12:
        raise UsageError("the model has no size along that direction")
    return L / d


def check_real_size(extent_m) -> None:
    m = float(np.max(extent_m))
    if not 0.005 <= m <= 50:
        raise UsageError(f"after scaling the model would be {m:.4g} m long; check the length you gave (metres)")


# ---------------------------------------------------------------- cleaning
def ground_plane(points: np.ndarray, up: np.ndarray, *, threshold: float, max_angle: float = 30.0,
                 min_share: float = 0.08, iterations: int = 400, seed: int = 3):
    """RANSAC for a plane whose normal is within `max_angle` of up. Returns (normal, d) with normal . x + d = 0
    and the normal pointing up, or None."""
    pts = np.asarray(points, float)
    n = len(pts)
    if n < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    cos_lim = math.cos(math.radians(max_angle))
    for _ in range(iterations):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm /= ln
        if np.dot(nrm, up) < 0:
            nrm = -nrm
        if np.dot(nrm, up) < cos_lim:
            continue
        d = -np.dot(nrm, a)
        count = int(np.count_nonzero(np.abs(pts @ nrm + d) < threshold))
        if count > best_count:
            best, best_count = (nrm, d), count
    if best is None or best_count < min_share * n:
        return None
    nrm, d = best
    inl = pts[np.abs(pts @ nrm + d) < threshold]
    # refit by least squares on the inliers
    c = inl.mean(0)
    _, _, Vt = np.linalg.svd(inl - c, full_matrices=False)
    nrm2 = Vt[2] if np.dot(Vt[2], up) > 0 else -Vt[2]
    return nrm2, float(-np.dot(nrm2, c))


def remove_below_plane(vertices, faces, plane, margin: float):
    """Drop faces with any vertex below the plane + margin. Returns the face mask kept."""
    nrm, d = plane
    above = vertices @ nrm + d > margin
    return above[faces].all(axis=1)


def largest_component(faces: np.ndarray, n_vertices: int, areas: np.ndarray | None = None) -> np.ndarray:
    """Mask of the faces in the connected part with the largest area (or most faces), joined by shared vertices."""
    faces = np.asarray(faces)
    if len(faces) == 0:
        return np.zeros(0, bool)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rows = np.concatenate([faces[:, 0], faces[:, 1]])
    cols = np.concatenate([faces[:, 1], faces[:, 2]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices))
    _, labels = connected_components(graph, directed=False)
    roots = labels[faces[:, 0]]
    w = np.ones(len(faces)) if areas is None else np.asarray(areas, float)
    uniq, inv = np.unique(roots, return_inverse=True)
    totals = np.bincount(inv, weights=w)
    return inv == int(np.argmax(totals))


def triangle_areas(vertices, faces) -> np.ndarray:
    v = np.asarray(vertices, float)
    f = np.asarray(faces)
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def parse_box(text: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        vals = [float(v) for v in str(text).split(",")]
    except ValueError:
        raise UsageError(f"--box is six numbers 'xmin,ymin,zmin,xmax,ymax,zmax' in metres, got {text!r}") from None
    if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
        raise UsageError("--box is six finite numbers 'xmin,ymin,zmin,xmax,ymax,zmax'")
    lo, hi = np.array(vals[:3]), np.array(vals[3:])
    if np.any(hi <= lo):
        raise UsageError("--box: every max must be greater than its min")
    return lo, hi


def inside_box(vertices, faces, lo, hi) -> np.ndarray:
    ok = np.all((vertices >= lo) & (vertices <= hi), axis=1)
    return ok[faces].all(axis=1)


def compact(vertices, faces, face_mask, *extra):
    """Keep the masked faces and only the vertices they use. Extra per-vertex arrays are filtered alike."""
    f = np.asarray(faces)[np.asarray(face_mask, bool)]
    used = np.unique(f)
    remap = -np.ones(len(vertices), int)
    remap[used] = np.arange(len(used))
    out = [np.asarray(vertices)[used], remap[f]]
    out += [None if e is None else np.asarray(e)[used] for e in extra]
    return out


def near_points(vertices, faces, points, distance: float) -> np.ndarray:
    """Faces whose vertices all lie within `distance` of some structure-from-motion point. Stereo mistakes
    (thin fins along silhouettes) reach out where no feature was ever matched; the object's surface does not."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(points, float)).query(np.asarray(vertices, float), distance_upper_bound=distance)
    ok = np.isfinite(d)
    return ok[np.asarray(faces)].all(axis=1)
