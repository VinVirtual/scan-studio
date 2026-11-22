import math
import os

import cv2
import numpy as np
import pytest

from scan_studio import dense, evaluate, media, sfm, texture
from scan_studio.errors import ScanError
from conftest import blurred, textured, write_video

PROBE = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'x.mov':
  Duration: 00:00:12.50, start: 0.000000, bitrate: 9000 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv), 1920x1080, 8000 kb/s, 29.97 fps, 29.97 tbr, 600 tbn (default)
      Side data:
        displaymatrix: rotation of -90.00 degrees
"""


def test_parse_probe_reads_size_fps_duration_and_turns_portrait_upright():
    w, h, fps, dur, rot = media.parse_probe(PROBE)
    assert (w, h) == (1080, 1920) and fps == pytest.approx(29.97) and dur == pytest.approx(12.5) and rot == 270


def test_parse_probe_without_video_stream():
    with pytest.raises(ScanError):
        media.parse_probe("Duration: 00:00:01.00\n  Stream #0:0: Audio: aac, 48000 Hz")


def test_decode_candidates_from_a_real_video(tmp):
    frames = [textured(240, 320, seed=i) for i in range(20)]
    frames[7] = blurred(frames[7], 5)
    path = write_video(os.path.join(tmp, "clip.mp4"), frames, fps=10)
    info = media.probe_video(path)
    assert (info.width, info.height) == (320, 240) and info.duration == pytest.approx(2.0, abs=0.15)
    out = list(media.decode_candidates(info))
    assert len(out) == 20 and out[3][0] == pytest.approx(0.3)
    from scan_studio import frames as fr
    scores = [fr.blur_score(f) for _, f in out]
    assert int(np.argmin(scores)) == 7


def test_decode_respects_candidate_budget(tmp):
    from scan_studio import budgets
    path = write_video(os.path.join(tmp, "clip.mp4"), [textured(240, 320, seed=i) for i in range(30)], fps=10)
    info = media.probe_video(path)
    out = list(media.decode_candidates(info, budgets.with_overrides(candidate_frames=8)))
    assert len(out) <= 8 and out[1][0] > out[0][0]


def test_a_text_file_named_mp4_is_refused(tmp):
    p = os.path.join(tmp, "fake.mp4"); open(p, "w").write("not a video")
    with pytest.raises(ScanError):
        media.probe_video(p)


# ------------------------------------------------------------------ structure from motion helpers
def test_pair_list_for_a_video():
    pairs = sfm.pair_list(30, overlap=3, stride=10)
    assert (0, 1) in pairs and (0, 3) in pairs and (0, 4) not in pairs
    assert (0, 20) in pairs and (10, 20) in pairs          # loop closure grid
    assert all(a < b for a, b in pairs) and len(set(pairs)) == len(pairs)


def test_pair_list_unordered_is_all_pairs():
    assert len(sfm.pair_list(10, ordered=False)) == 45
    assert sfm.pair_list(1) == []


@pytest.mark.parametrize("model,params,expect", [
    ("SIMPLE_PINHOLE", [500, 320, 240], [0, 0, 0, 0]),
    ("SIMPLE_RADIAL", [500, 320, 240, 0.01], [0.01, 0, 0, 0]),
    ("OPENCV", [500, 510, 320, 240, 0.1, 0.2, 0.001, 0.002], [0.1, 0.2, 0.001, 0.002]),
])
def test_camera_to_opencv(model, params, expect):
    K, d = sfm.camera_to_opencv(model, params, 640, 480)
    assert K[0, 2] == 320 and K[1, 2] == 240 and K[2, 2] == 1
    assert np.allclose(d, expect)


def test_unknown_camera_model():
    with pytest.raises(ScanError):
        sfm.camera_to_opencv("FISHEYE_X", [1, 2, 3], 10, 10)


# ------------------------------------------------------------------ dense helpers
def look_at(centre, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - centre; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ centre


def orbit_views(n=24, radius=1.0, height=0.4):
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    views = []
    for k in range(n):
        a = 2 * math.pi * k / n
        c = np.array([radius * math.cos(a), radius * math.sin(a), height])
        R, t = look_at(c, [0, 0, 0.1])
        views.append(sfm.View(f"{k:04d}.jpg", K, np.zeros(4), R, t, 640, 480))
    return views


def test_view_axes():
    v = orbit_views()[0]
    assert v.forward @ (np.array([0, 0, 0.1]) - v.centre) > 0
    assert v.up[2] > 0 and abs(v.right[2]) < 1e-9


def test_ray_convergence_finds_the_target():
    assert np.allclose(dense.ray_convergence(orbit_views()), [0, 0, 0.1], atol=1e-6)


def test_ray_convergence_refuses_parallel_cameras():
    vs = orbit_views(3)
    for v in vs:
        v.R = np.eye(3)
    with pytest.raises(ScanError):
        dense.ray_convergence(vs)


def test_choose_pairs_picks_neighbours_a_few_degrees_apart():
    vs = orbit_views(72)            # 5 degrees apart
    pairs = dense.choose_pairs(vs, max_views=20)
    assert 15 <= len(pairs) <= 20
    for i, j in pairs:
        assert 3.0 <= dense.view_angle(vs[i], vs[j]) <= 16.0


def test_drop_depth_edges_removes_silhouette_band():
    d = np.full((100, 100), 2.0, np.float32)
    d[:, 50:] = 5.0
    out = dense.drop_depth_edges(d)
    assert np.all(out[:, :45] == 2.0) and np.all(out[:, 56:] == 5.0)
    assert np.all(out[:, 48:52] == 0)


def depth_view(world_to_cam, depth):
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.0]])
    return dense.DepthView(np.zeros((*depth.shape, 3), np.uint8), depth, K, world_to_cam)


def test_consistency_filter_keeps_agreeing_surfaces_and_drops_outliers():
    E = np.eye(4)
    good = np.full((100, 100), 3.0, np.float32)
    bad = good.copy(); bad[10:20, 10:20] = 1.5     # a mistake only one map makes
    maps = [depth_view(E, good), depth_view(E, bad), depth_view(E, good), depth_view(E, good)]
    out = dense.consistency_filter(maps, neighbours=3, need=2)
    assert np.all(out[1].depth[10:20, 10:20] == 0)
    assert np.count_nonzero(out[1].depth) == 100 * 100 - 100
    assert np.all(out[0].depth > 0)


def test_backproject_and_agrees_round_trip():
    E = np.eye(4); E[:3, 3] = [0.1, -0.2, 0.0]
    dv = depth_view(E, np.full((100, 100), 2.0, np.float32))
    pts, idx = dense.backproject(dv)
    assert len(pts) == 10000 and np.allclose(pts[:, 2], 2.0)
    assert dense.agrees(pts, dv).all()


def test_mask_region():
    dv = depth_view(np.eye(4), np.full((100, 100), 2.0, np.float32))
    out = dense.mask_region(dv, dense.Region(np.array([0, 0, 2.0]), 0.3, 2.0))
    assert 0 < np.count_nonzero(out.depth) < 10000
    assert out.depth[50, 50] == 2.0 and out.depth[0, 0] == 0


# ------------------------------------------------------------------ texture helpers
def test_rasterize_and_barycentric():
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    fid = texture.rasterize(uv, faces, 64)
    assert (fid >= 0).mean() > 0.95
    assert fid[63, 0] == 0 and fid[0, 63] == 1      # bottom-left texel is v = 0
    bc = texture.barycentric(np.array([[0.25, 0.25]]), uv[[0]], uv[[1]], uv[[2]])
    assert np.allclose(bc, [[0.5, 0.25, 0.25]])


def test_sample_handles_more_points_than_remap_allows():
    img = np.zeros((10, 10, 3), np.uint8); img[..., 1] = 200
    xy = np.random.default_rng(0).random((70_000, 2)) * 9
    out = texture.sample(img, xy)
    assert out.shape == (70_000, 3) and np.allclose(out[:, 1], 200)


def test_pad_grows_colour_into_gutters():
    col = np.zeros((20, 20, 3), np.float32); col[5:10, 5:10] = 100
    filled = np.zeros((20, 20), bool); filled[5:10, 5:10] = True
    out = texture.pad(col, filled, steps=2)
    assert out[5, 11, 0] == pytest.approx(100) and out[0, 0, 0] == 0


# ------------------------------------------------------------------ evaluation maths
def test_kabsch_recovers_rotation_translation_and_scale():
    rng = np.random.default_rng(3)
    src = rng.random((40, 3))
    ang = 0.7
    R = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    dst = 2.5 * src @ R.T + [1, 2, 3]
    R2, t2, s2 = evaluate.kabsch(src, dst, with_scale=True)
    assert np.allclose(R2, R) and s2 == pytest.approx(2.5) and np.allclose(t2, [1, 2, 3])
    R3, _, s3 = evaluate.kabsch(src, src @ R.T)
    assert s3 == 1.0 and np.allclose(R3, R)


def test_metrics_identical_and_shifted():
    rng = np.random.default_rng(4)
    pts = rng.random((2000, 3))
    m = evaluate.metrics(pts, pts, 0.01)
    assert m["chamfer_mean"] == 0 and m["fscore"] == 1.0
    m2 = evaluate.metrics(pts + [0.05, 0, 0], pts, 0.01)
    assert m2["precision"] < 0.5 and m2["chamfer_mean"] > 0.005


def test_passthrough_option_follows_ffmpeg_version():
    assert media.passthrough_args((7, 1)) == ["-fps_mode", "passthrough"]
    assert media.passthrough_args((5, 1)) == ["-fps_mode", "passthrough"]
    assert media.passthrough_args((4, 4)) == ["-vsync", "0"]
    assert media.passthrough_args((0, 0)) == ["-vsync", "0"]


def test_probe_skips_cover_art():
    text = ("Duration: 00:00:05.00\n  Stream #0:0: Video: mjpeg, yuvj420p, 300x300, 90k tbr (attached pic)\n"
            "  Stream #0:1: Video: h264, yuv420p, 1280x720, 30 fps, 30 tbr\n")
    assert media.parse_probe(text)[:2] == (1280, 720)


def test_photo_long_side_keeps_all_photos_inside_the_pixel_budget():
    from scan_studio import budgets
    b = budgets.DEFAULT
    for n in (1, 50, 300, b.photos):
        side = media.photo_long_side(n)
        assert side <= b.work_long_side
        assert n * side * side * 0.75 <= b.candidate_pixels * 1.001


def test_unreadable_photo_is_a_clear_error(tmp):
    p = os.path.join(tmp, "broken.jpg"); open(p, "wb").write(b"\xff\xd8not really a jpeg")
    with pytest.raises(ScanError, match="not a readable image"):
        media.load_photo(p)
