"""Decimate the mesh, unwrap it (xatlas) and bake a colour texture from the original frames.

Each texel is coloured from the frames that see its triangle most squarely (up to three), after a ray test
that the triangle is not hidden from that camera. Texels no frame sees take the colour of the nearest point of
the dense mesh. Small gaps between charts are padded so mipmaps do not bleed black seams."""
from __future__ import annotations

import os

import cv2
import numpy as np

from .sfm import View


def decimate(vertices, faces, colors, target: int):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    if colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    if len(faces) > target:
        m = m.simplify_quadric_decimation(int(target))
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    cols = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None
    return np.asarray(m.vertices), np.asarray(m.triangles), cols


def unwrap(vertices, faces, size: int):
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = int(size)
    pack.padding = 2
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmap, idx, uv = atlas[0]
    return vmap, idx, uv


def rasterize(uv: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Face index per texel (-1 for empty). Texel row 0 is v = 1 (images run top-down, UVs bottom-up)."""
    fid = np.full((size, size), -1, np.int32)
    px = np.stack([uv[:, 0] * size, (1 - uv[:, 1]) * size], -1)
    tri = np.round(px[faces] * 4).astype(np.int32)   # 2 bits of sub-pixel precision
    for i, t in enumerate(tri):
        cv2.fillConvexPoly(fid, t, int(i), lineType=cv2.LINE_8, shift=2)
    return fid


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1).sum(-1); d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1); d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.stack([1 - v - w, v, w], -1)
    bc = np.clip(bc, 0, 1)
    return bc / bc.sum(-1, keepdims=True)


def sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear colour at many (x, y) positions. cv2.remap takes at most 32767 per side, so lay them out in rows."""
    n = len(xy)
    if n == 0:
        return np.zeros((0, 3), np.float32)
    cols = 1024
    rows = -(-n // cols)
    grid = np.full((rows * cols, 2), -1, np.float32)
    grid[:n] = xy
    grid = grid.reshape(rows, cols, 2)
    out = cv2.remap(img, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, img.shape[2] if img.ndim == 3 else 1)[:n].astype(np.float32)


def pad(color: np.ndarray, filled: np.ndarray, steps: int = 6) -> np.ndarray:
    col = color.astype(np.float32)
    m = filled.astype(np.float32)
    for _ in range(steps):
        s = cv2.blur(col * m[..., None], (3, 3))
        w = cv2.blur(m, (3, 3))
        grow = (m == 0) & (w > 0)
        col[grow] = s[grow] / w[grow][:, None]
        m = np.maximum(m, grow.astype(np.float32))
    return col


def bake(vertices, faces, views: list[View], image_dir: str, *, size: int, dense_vertices, dense_colors,
         per_face: int = 3, log=print):
    """Returns (new vertices, new faces, uv, texture RGB uint8, share of texels seen by a camera)."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    vmap, nf, uv = unwrap(vertices, faces, size)
    nv = np.asarray(vertices)[vmap]
    fid = rasterize(uv, nf, size)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    puv = np.stack([(xs + 0.5) / size, 1 - (ys + 0.5) / size], -1)
    bc = barycentric(puv, uv[nf[f, 0]], uv[nf[f, 1]], uv[nf[f, 2]])
    P = (nv[nf[f]] * bc[..., None]).sum(1)
    tri = nv[nf]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    fc = tri.mean(1)
    # score every (face, view): how squarely the view sees the face, zero if behind or outside the frame
    C = np.array([v.centre for v in views])
    score = np.zeros((len(nf), len(views)), np.float32)
    for k, v in enumerate(views):
        d = C[k] - fc
        dist = np.linalg.norm(d, axis=1)
        cos = (fn * d).sum(1) / np.maximum(dist, 1e-20)
        cam = fc @ v.R.T + v.t
        z = cam[:, 2]
        u = v.K[0, 0] * cam[:, 0] / np.maximum(z, 1e-9) + v.K[0, 2]
        w = v.K[1, 1] * cam[:, 1] / np.maximum(z, 1e-9) + v.K[1, 2]
        ok = (z > 0) & (u > 2) & (w > 2) & (u < v.width - 3) & (w < v.height - 3) & (cos > 0.15)
        score[:, k] = np.where(ok, cos ** 2, 0)
    k = min(per_face, score.shape[1])
    top = np.argpartition(-score, k - 1, axis=1)[:, :k] if score.shape[1] > k else np.tile(np.arange(k), (len(nf), 1))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(nv.astype(np.float32)), o3d.core.Tensor(nf.astype(np.uint32)))
    acc = np.zeros((len(P), 3))
    wsum = np.zeros(len(P))
    extent = float(np.linalg.norm(nv.max(0) - nv.min(0)))
    for k, v in enumerate(views):
        sel = np.nonzero((top[f] == k).any(axis=1) & (score[f, k] > 0))[0]
        if len(sel) == 0:
            continue
        p = P[sel]
        d = p - C[k]
        dist = np.linalg.norm(d, axis=1)
        rays = np.concatenate([np.repeat(C[k][None], len(p), 0), d / dist[:, None]], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis = hit >= dist - 0.004 * extent
        if not vis.any():
            continue
        sel, p = sel[vis], p[vis]
        img = cv2.imread(os.path.join(image_dir, v.name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        proj, _ = cv2.projectPoints(p, cv2.Rodrigues(v.R)[0], v.t, v.K, v.dist)
        col = sample(img, proj.reshape(-1, 2))
        wgt = score[f[sel], k]
        acc[sel] += col * wgt[:, None]
        wsum[sel] += wgt
    seen = wsum > 0
    colors = np.zeros((len(P), 3))
    colors[seen] = acc[seen] / wsum[seen][:, None]
    if (~seen).any() and dense_vertices is not None and dense_colors is not None and len(dense_vertices):
        _, nn = cKDTree(dense_vertices).query(P[~seen])
        colors[~seen] = np.asarray(dense_colors)[nn] * 255.0
    tex = np.zeros((size, size, 3), np.float32)
    tex[ys, xs] = colors
    filled = np.zeros((size, size), bool)
    filled[ys, xs] = True
    tex = pad(tex, filled)
    share = float(seen.mean()) if len(seen) else 0.0
    log(f"  texture: {size}x{size}, {len(P):,} texels, {share:.0%} coloured from frames")
    return nv, nf, uv, np.clip(tex, 0, 255).astype(np.uint8), share
