import math
import os

import cv2
import numpy as np
import pytest

from scan_studio import dense, evaluate, media, sfm, texture
from scan_studio.errors import BudgetError, ScanError
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


def test_photo_long_side_uses_the_real_sizes():
    from scan_studio import budgets
    b = budgets.DEFAULT
    for sizes in ([(4000, 3000)] * 300, [(4000, 1000)] * 400, [(1000, 1000)] * 10, [(4032, 3024), (1080, 1920)] * 150):
        side = media.photo_long_side(sizes)
        total = sum(w * h * min(1, side / max(w, h)) ** 2 for w, h in sizes)
        assert side <= b.work_long_side and total <= b.candidate_pixels
    assert media.photo_long_side([(800, 600)] * 5) == b.work_long_side      # small sets keep the full work size
    with pytest.raises(BudgetError):
        media.photo_long_side([(8000, 8000)] * 400, budgets.with_overrides(candidate_pixels=10_000_000))


def test_unreadable_photo_is_a_clear_error(tmp):
    p = os.path.join(tmp, "broken.jpg"); open(p, "wb").write(b"\xff\xd8not really a jpeg")
    with pytest.raises(ScanError, match="not a readable"):
        media.load_photo(p)


# ------------------------------------------------------------------ hostile inputs
@pytest.mark.parametrize("name,content", [
    ("eps.jpg", b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"),
    ("gif.jpg", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
    ("tiff.png", b"II*\x00\x08\x00\x00\x00\x00\x00"),
])
def test_photos_only_reach_the_jpeg_and_png_decoders(tmp, name, content, monkeypatch):
    from PIL import EpsImagePlugin
    called = []
    monkeypatch.setattr(EpsImagePlugin, "has_ghostscript", lambda: called.append(1) or True, raising=False)
    p = os.path.join(tmp, name); open(p, "wb").write(content)
    with pytest.raises(ScanError, match="JPEG or PNG"):
        media.load_photo(p)
    with pytest.raises(ScanError):
        media.photo_sizes([p])
    assert not called


def test_real_jpeg_and_png_still_open(tmp):
    from PIL import Image
    for n in ("a.jpg", "b.png"):
        Image.new("RGB", (40, 30), (10, 20, 30)).save(os.path.join(tmp, n))
        assert media.load_photo(os.path.join(tmp, n)).shape == (30, 40, 3)


def test_concat_list_named_mp4_is_not_followed(tmp):
    write_video(os.path.join(tmp, "real.mp4"), [textured(240, 320, seed=i) for i in range(5)])
    lst = os.path.join(tmp, "list.mp4")
    open(lst, "w").write("ffconcat version 1.0\nfile 'real.mp4'\n")
    with pytest.raises(ScanError):
        media.probe_video(lst)


def test_duration_not_stated_is_read_as_none_and_frames_are_counted(tmp):
    text = "Duration: N/A, bitrate: N/A\n  Stream #0:0: Video: h264, yuv420p, 320x240, 10 fps, 10 tbr\n"
    assert media.parse_probe(text)[3] is None
    path = write_video(os.path.join(tmp, "clip.mp4"), [textured(240, 320, seed=i) for i in range(17)], fps=10)
    assert media.count_frames(media.find_ffmpeg(), path) == 17


def test_source_resolution_is_capped(tmp):
    from scan_studio.errors import BudgetError as BE
    path = write_video(os.path.join(tmp, "wide.mp4"), [np.zeros((64, 4200, 3), np.uint8)] * 2, fps=10)
    with pytest.raises(BE, match="4096"):
        media.probe_video(path)


def test_decode_deadline_kills_ffmpeg(tmp):
    path = write_video(os.path.join(tmp, "clip.mp4"), [textured(240, 320, seed=i) for i in range(30)], fps=10)
    info = media.probe_video(path)
    with pytest.raises(ScanError, match="longer than"):
        for _ in media.decode_candidates(info, deadline=0.0):
            import time
            time.sleep(0.05)


def test_face_sample_covers_candidates_that_are_not_kept(tmp):
    from scan_studio import budgets, capture
    from drawn_face import drawn_face
    frames = []
    for i in range(30):
        f = textured(720, 1280, seed=i)
        if i % 3 == 1:
            f[150:500, 400:750] = drawn_face(350)
        frames.append(f)
    path = write_video(os.path.join(tmp, "clip.mp4"), frames, fps=10)
    work = os.path.join(tmp, "work"); os.mkdir(work)
    cap = capture.from_video(path, work, 12, budgets.DEFAULT, log=lambda *_: None)
    assert sum(1 for b in cap.face_sample.values() if b) >= 5
    assert not os.path.exists(os.path.join(work, "candidates"))


def test_height_plane_finds_the_table_band():
    rng = np.random.default_rng(0)
    table = np.c_[rng.uniform(-1, 1, (2000, 2)), rng.normal(0, 0.003, 2000)]
    obj = rng.uniform([-0.2, -0.2, 0], [0.2, 0.2, 0.6], (800, 3))
    n, d = dense.height_plane(np.r_[table, obj], np.array([0, 0, 1.0]), np.array([0, 0, 0.3]), 2.0)
    assert np.allclose(n, [0, 0, 1]) and abs(d) < 0.01
    assert dense.height_plane(obj, np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), 2.0) is None


class FakeRec:
    def __init__(self, n):
        self.n = n

    def num_reg_images(self):
        return self.n


def test_mapping_retries_after_a_bad_start():
    tries = []
    results = [FakeRec(2), None, FakeRec(80)]

    def once(attempt, seed):
        tries.append(seed)
        return results[attempt]
    best = sfm.map_with_retries(once, 89, log=lambda *_: None)
    assert best.num_reg_images() == 80 and len(tries) == 3 and len(set(tries)) == 3


def test_mapping_stops_when_good_and_keeps_the_best():
    calls = []
    best = sfm.map_with_retries(lambda a, s: calls.append(a) or FakeRec(70), 89, log=lambda *_: None)
    assert best.n == 70 and calls == [0]
    best = sfm.map_with_retries(lambda a, s: FakeRec([5, 30, 12][a]), 89, log=lambda *_: None)
    assert best.n == 30
    assert sfm.map_with_retries(lambda a, s: None, 89, log=lambda *_: None) is None
