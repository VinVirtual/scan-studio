"""Dense reconstruction on the CPU: two-view stereo (OpenCV semi-global matching) fused into a TSDF volume (Open3D).

COLMAP's own multi-view stereo (PatchMatch) needs CUDA, which this pipeline does not assume. Instead each
frame is paired with a neighbour that sees the object from a few degrees away, the pair is rectified with the
poses from structure from motion, a depth map is computed with semi-global matching (checked left-right), and
every depth map is fused into one truncated signed distance volume. Marching cubes on that volume gives a
coloured mesh. If too few pairs work, `sparse_poisson` builds a coarser surface from the sparse points alone."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .errors import ScanError
from .sfm import SparseModel, View

STEREO_WIDTH = 960


def ray_convergence(views: list[View]) -> np.ndarray:
    """The point closest (least squares) to every camera's viewing ray: what the camera kept looking at."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for v in views:
        d = v.forward / np.linalg.norm(v.forward)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ v.centre
    if np.linalg.cond(A) > 1e8:
        raise ScanError("the cameras all look in parallel; no common object in view")
    return np.linalg.solve(A, b)


@dataclass(frozen=True)
class Region:
    centre: np.ndarray
    radius: float
    camera_distance: float
    fitted: str = "cameras"      # "ground" when fitted to the points standing on the ground plane


def region_of_interest(model: SparseModel, up: np.ndarray | None = None) -> Region:
    """A sphere round the object. Starts from the point the cameras looked at; when a ground plane is found
    among the sparse points (with `up` known), the sphere is fitted to the points standing on it, so the table
    and the room are mostly left out of the dense step."""
    from .geometry import ground_plane
    c = ray_convergence(model.views)
    dists = np.array([np.linalg.norm(v.centre - c) for v in model.views])
    D = float(np.median(dists))
    pts = model.points[np.linalg.norm(model.points - c, axis=1) < 0.7 * D]
    if up is not None and len(pts) > 100:
        plane = ground_plane(pts, up, threshold=0.01 * D, min_share=0.05, iterations=1500)
        if plane is None:
            plane = height_plane(pts, up, c, D)
        if plane is not None:
            h = pts @ plane[0] + plane[1]
            obj = pts[h > 0.02 * D]
            if len(obj) > 50:
                centre = np.median(obj, axis=0)
                r = float(np.percentile(np.linalg.norm(obj - centre, axis=1), 97) * 1.3)
                return Region(centre, float(np.clip(r, 0.08 * D, 0.7 * D)), D, "ground")
    pd = np.linalg.norm(model.points - c, axis=1)
    near = pd[pd < 0.7 * D]
    r = float(np.percentile(near, 90) * 1.15) if len(near) > 20 else 0.4 * D
    r = float(np.clip(r, 0.12 * D, 0.7 * D))
    return Region(c, r, D)


def height_plane(pts: np.ndarray, up: np.ndarray, centre: np.ndarray, D: float):
    """Deterministic fallback for the ground: the most crowded height (along up) below the point the cameras
    looked at. A table or floor under the object is a dense band of points at one height."""
    up = up / np.linalg.norm(up)
    h = pts @ up
    below = h[h < centre @ up]
    if len(below) < 50:
        return None
    step = 0.01 * D
    bins = np.arange(below.min(), below.max() + 2 * step, step)
    counts, edges = np.histogram(below, bins)
    k = int(np.argmax(counts))
    if counts[k] < 0.05 * len(below):
        return None
    level = float(np.median(below[(below >= edges[k]) & (below < edges[k + 1])]))
    return up, -level


def view_angle(a: View, b: View) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a.forward, b.forward), -1, 1))))


def choose_pairs(views: list[View], *, max_views: int = 70, lo: float = 3.0, hi: float = 16.0,
                 best: float = 8.0) -> list[tuple[int, int]]:
    """One partner per reference view, a few degrees away. At most `max_views` reference views, evenly spaced."""
    n = len(views)
    refs = np.unique(np.linspace(0, n - 1, min(n, max_views)).round().astype(int))
    pairs = []
    for i in refs:
        cands = [(abs(view_angle(views[i], views[j]) - best), j) for j in range(n) if j != i
                 and lo <= view_angle(views[i], views[j]) <= hi]
        if cands:
            pairs.append((int(i), min(cands)[1]))
    return pairs


def _scaled(K: np.ndarray, s: float) -> np.ndarray:
    K2 = K.copy()
    K2[:2] *= s
    return K2


def _sgbm(num_disp: int, min_disp: int, block: int = 5) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
                                 P1=8 * 3 * block * block, P2=32 * 3 * block * block, disp12MaxDiff=-1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


@dataclass
class DepthView:
    color: np.ndarray        # rectified colour image (H, W, 3) uint8
    depth: np.ndarray        # depth in model units, 0 where unknown
    K: np.ndarray            # intrinsics of the rectified image
    world_to_cam: np.ndarray # 4x4


def rectify_pair(a: View, b: View, img_a: np.ndarray, img_b: np.ndarray):
    s = min(1.0, STEREO_WIDTH / a.width)
    size = (int(round(a.width * s)), int(round(a.height * s)))
    Ka, Kb = _scaled(a.K, s), _scaled(b.K, s)
    R = b.R @ a.R.T
    t = b.t - R @ a.t
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Ka, a.dist, Kb, b.dist, size, R, t, alpha=0,
                                                flags=cv2.CALIB_ZERO_DISPARITY)
    m1 = cv2.initUndistortRectifyMap(Ka, a.dist, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Kb, b.dist, R2, P2, size, cv2.CV_32FC1)
    ra = cv2.remap(cv2.resize(img_a, size, interpolation=cv2.INTER_AREA), *m1, cv2.INTER_LINEAR)
    rb = cv2.remap(cv2.resize(img_b, size, interpolation=cv2.INTER_AREA), *m2, cv2.INTER_LINEAR)
    return ra, rb, R1, P1, P2


def depth_for_pair(a: View, b: View, img_a: np.ndarray, img_b: np.ndarray, pts_a: np.ndarray,
                   region: Region) -> DepthView | None:
    """Depth map for view `a` from the pair (a, b), or None when the pair is unusable."""
    ra, rb, R1, P1, P2 = rectify_pair(a, b, img_a, img_b)
    tx, ty = P2[0, 3], P2[1, 3]
    vertical = abs(ty) > abs(tx)
    base = ty if vertical else tx
    if base > 0:
        return None  # partner on the wrong side; the caller tries the pair the other way round
    f = P1[0, 0]
    fb = -base          # focal length times baseline
    if vertical:
        ra, rb = np.ascontiguousarray(ra.transpose(1, 0, 2)), np.ascontiguousarray(rb.transpose(1, 0, 2))
    # disparity search range from the sparse points this view sees
    X = (R1 @ (a.R @ pts_a.T + a.t[:, None])).T
    z = X[:, 2][X[:, 2] > 0]
    if len(z) < 15:
        return None
    zmin, zmax = np.percentile(z, 2) * 0.8, np.percentile(z, 98) * 1.25
    dmin, dmax = fb / zmax, fb / max(zmin, 1e-9)
    min_disp = int(math.floor(dmin))
    num = int(math.ceil((dmax - min_disp) / 16.0)) * 16
    if num < 16 or num > 320:
        return None
    ga, gb = cv2.cvtColor(ra, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rb, cv2.COLOR_RGB2GRAY)
    dl = _sgbm(num, min_disp).compute(ga, gb).astype(np.float32) / 16.0
    # right-to-left disparity by mirroring both images, for a left-right consistency check
    # (mirroring keeps disparities positive and in the same range; the result is indexed by right-image column)
    dr = _sgbm(num, min_disp).compute(np.ascontiguousarray(gb[:, ::-1]), np.ascontiguousarray(ga[:, ::-1]))
    dr = (dr.astype(np.float32) / 16.0)[:, ::-1]
    h, w = dl.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    valid = dl > max(min_disp, 0.5)
    xr = np.clip(np.round(xs - dl).astype(int), 0, w - 1)
    back = dr[np.arange(h)[:, None], xr]
    valid &= np.abs(back - dl) <= 1.0
    depth = np.where(valid, fb / np.maximum(dl, 1e-6), 0).astype(np.float32)
    if vertical:
        depth = np.ascontiguousarray(depth.T)
        ra = np.ascontiguousarray(ra.transpose(1, 0, 2))
    K = P1[:3, :3].copy()
    w2c = np.eye(4)
    w2c[:3, :3] = R1 @ a.R
    w2c[:3, 3] = R1 @ a.t
    if np.count_nonzero(depth) < 0.01 * depth.size:
        return None
    return DepthView(ra, depth, K, w2c)


def drop_depth_edges(depth: np.ndarray, rel: float = 0.03, size: int = 7) -> np.ndarray:
    """Zero the depth next to jumps (object silhouettes), where block matching smears foreground into background."""
    d = depth.copy()
    valid = d > 0
    big = np.where(valid, d, 0).astype(np.float32)
    small = np.where(valid, d, np.float32(1e9)).astype(np.float32)
    k = np.ones((size, size), np.uint8)
    hi = cv2.dilate(big, k)
    lo = cv2.erode(small, k)
    jump = (hi - lo) > rel * np.where(valid, d, 1)
    holes = cv2.dilate((~valid).astype(np.uint8), k) > 0     # also the rim of every hole
    d[jump | (holes & valid)] = 0
    return d


def mask_region(dv: "DepthView", region: Region) -> "DepthView":
    """Zero depth whose point falls outside the region sphere (after the edge check, so background edges count)."""
    pts, idx = backproject(dv)
    depth = dv.depth.copy().reshape(-1)
    depth[idx[np.linalg.norm(pts - region.centre, axis=1) > region.radius]] = 0
    return DepthView(dv.color, depth.reshape(dv.depth.shape), dv.K, dv.world_to_cam)


def backproject(dv: DepthView) -> tuple[np.ndarray, np.ndarray]:
    """World points of every valid pixel of a depth view, and their flat pixel indices."""
    H, W = dv.depth.shape
    idx = np.flatnonzero(dv.depth.reshape(-1) > 0)
    v, u = np.divmod(idx, W)
    z = dv.depth.reshape(-1)[idx]
    cam = np.stack([(u - dv.K[0, 2]) * z / dv.K[0, 0], (v - dv.K[1, 2]) * z / dv.K[1, 1], z], -1)
    c2w = np.linalg.inv(dv.world_to_cam)
    return cam @ c2w[:3, :3].T + c2w[:3, 3], idx


def agrees(points: np.ndarray, other: DepthView, rel: float = 0.01) -> np.ndarray:
    """Which world points land on the surface another depth view saw (same depth within rel)."""
    cam = points @ other.world_to_cam[:3, :3].T + other.world_to_cam[:3, 3]
    z = cam[:, 2]
    H, W = other.depth.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(other.K[0, 0] * cam[:, 0] / z + other.K[0, 2]).astype(np.int64)
        v = np.round(other.K[1, 1] * cam[:, 1] / z + other.K[1, 2]).astype(np.int64)
    ok = (z > 0) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
    out = np.zeros(len(points), bool)
    d = other.depth[v[ok], u[ok]]
    out[ok] = (d > 0) & (np.abs(d - z[ok]) < rel * z[ok])
    return out


def consistency_filter(depths: list[DepthView], *, neighbours: int = 4, need: int = 2, rel: float = 0.01) -> list[DepthView]:
    """Keep a depth pixel only when at least `need` of the nearby depth maps saw the same surface there.
    Removes most stereo mistakes, which rarely agree across views."""
    out = []
    for k, dv in enumerate(depths):
        pts, idx = backproject(dv)
        votes = np.zeros(len(pts), np.int32)
        for m in range(max(0, k - neighbours), min(len(depths), k + neighbours + 1)):
            if m != k:
                votes += agrees(pts, depths[m], rel)
        depth = dv.depth.copy().reshape(-1)
        depth[idx[votes < need]] = 0
        out.append(DepthView(dv.color, depth.reshape(dv.depth.shape), dv.K, dv.world_to_cam))
    return out


def fuse(depths: list[DepthView], region: Region, resolution: int = 384):
    import open3d as o3d
    voxel = 2 * region.radius / resolution
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=5 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for d in depths:
        H, W = d.depth.shape
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, d.K[0, 0], d.K[1, 1], d.K[0, 2], d.K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(d.color)), o3d.geometry.Image(d.depth),
            depth_scale=1.0, depth_trunc=float(region.camera_distance + 2 * region.radius),
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, d.world_to_cam)
    mesh = vol.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        raise ScanError("dense fusion produced no surface")
    return mesh, voxel


def run(model: SparseModel, image_dir: str, *, up: np.ndarray | None = None, max_views: int = 70,
        resolution: int = 384, log=print):
    """Depth maps for up to `max_views` frames, fused into one coloured mesh. Returns (mesh, info dict)."""
    region = region_of_interest(model, up)
    pairs = choose_pairs(model.views, max_views=max_views)
    cache: dict[int, np.ndarray] = {}

    def img(i: int) -> np.ndarray:
        if i not in cache:
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            bgr = cv2.imread(os.path.join(image_dir, model.views[i].name), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ScanError(f"could not read frame {model.views[i].name}")
            cache[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cache[i]

    seen: dict[int, list[int]] = {}
    for p, vis in enumerate(model.point_views):
        for v in vis:
            seen.setdefault(v, []).append(p)
    depths = []
    for i, j in pairs:
        pts = model.points[seen.get(i, [])]
        d = depth_for_pair(model.views[i], model.views[j], img(i), img(j), pts, region)
        if d is None:   # try the pair the other way round
            pts_j = model.points[seen.get(j, [])]
            d = depth_for_pair(model.views[j], model.views[i], img(j), img(i), pts_j, region)
        if d is not None:
            depths.append(d)
    log(f"  stereo: {len(depths)} depth maps from {len(pairs)} pairs (region fitted to the {region.fitted})")
    if len(depths) < 6:
        raise ScanError(f"only {len(depths)} usable stereo pairs")
    before = sum(int(np.count_nonzero(d.depth)) for d in depths)
    depths = [mask_region(DepthView(d.color, drop_depth_edges(d.depth), d.K, d.world_to_cam), region) for d in depths]
    depths = consistency_filter(depths)
    after = sum(int(np.count_nonzero(d.depth)) for d in depths)
    log(f"  kept {after / max(before, 1):.0%} of depth pixels after edge and multi-view checks")
    mesh, voxel = fuse(depths, region, resolution)
    return mesh, {"method": "sgbm-tsdf", "pairs": len(pairs), "depth_maps": len(depths), "voxel": voxel,
                  "region_radius": region.radius, "region_centre": region.centre.tolist(),
                  "camera_distance": region.camera_distance, "region_fitted": region.fitted}


def sparse_poisson(model: SparseModel, *, up: np.ndarray | None = None, depth: int = 8, log=print):
    """The fallback: a Poisson surface through the sparse points, normals turned towards the cameras."""
    import open3d as o3d
    region = region_of_interest(model, up)
    keep = np.linalg.norm(model.points - region.centre, axis=1) < region.radius
    pts, cols = model.points[keep], model.colors[keep]
    if len(pts) < 200:
        raise ScanError(f"only {len(pts)} sparse points near the object; not enough for a surface")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.colors = o3d.utility.Vector3dVector(cols / 255.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    centres = np.array([v.centre for v in model.views])
    nrm = np.asarray(pc.normals)
    from scipy.spatial import cKDTree
    near_cam = centres[cKDTree(centres).query(pts)[1]]
    flip = ((near_cam - pts) * nrm).sum(1) < 0
    nrm[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(nrm)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.percentile(dens, 8))
    log(f"  sparse Poisson: {len(pts)} points -> {len(mesh.triangles)} triangles")
    return mesh, {"method": "sparse-poisson", "points": int(len(pts)), "region_radius": region.radius,
                  "region_centre": region.centre.tolist(), "camera_distance": region.camera_distance}
