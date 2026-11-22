import math

import numpy as np
import pytest

from scan_studio import geometry as g
from scan_studio.errors import UsageError


# ------------------------------------------------------------------ scaling maths
def test_scale_from_points():
    assert g.scale_from_points([0, 0, 0], [0, 2, 0], 0.5) == pytest.approx(0.25)
    assert g.scale_from_points([1, 1, 1], [4, 5, 1], 0.1) == pytest.approx(0.02)


def test_scale_from_points_refuses_same_point_and_bad_length():
    with pytest.raises(UsageError):
        g.scale_from_points([1, 2, 3], [1, 2, 3], 0.2)
    for bad in (0, -1, float("nan"), float("inf"), "abc", 1000):
        with pytest.raises(UsageError):
            g.scale_from_points([0, 0, 0], [1, 0, 0], bad)


def test_scale_from_size_height_longest_width():
    ext = np.array([2.0, 4.0, 3.0])
    assert g.scale_from_size(ext, height=0.2) == pytest.approx(0.05)
    assert g.scale_from_size(ext, longest=0.8) == pytest.approx(0.2)
    assert g.scale_from_size(ext, width=0.3) == pytest.approx(0.1)   # wider of x and z


def test_scale_from_size_needs_exactly_one():
    with pytest.raises(UsageError):
        g.scale_from_size([1, 1, 1])
    with pytest.raises(UsageError):
        g.scale_from_size([1, 1, 1], height=0.1, longest=0.2)
    with pytest.raises(UsageError):
        g.scale_from_size([1, 0, 1], height=0.1)


def test_scaling_then_measuring_round_trips():
    rng = np.random.default_rng(0)
    pts = rng.random((50, 3)) * 7.3
    s = g.scale_from_size(pts.max(0) - pts.min(0), height=0.25)
    scaled = pts * s
    assert scaled[:, 1].max() - scaled[:, 1].min() == pytest.approx(0.25)
    s2 = g.scale_from_points(scaled[0], scaled[1], np.linalg.norm(scaled[0] - scaled[1]))
    assert s2 == pytest.approx(1.0)


def test_check_real_size():
    g.check_real_size([0.1, 0.2, 0.1])
    for bad in ([0.001, 0.001, 0.001], [60, 1, 1]):
        with pytest.raises(UsageError):
            g.check_real_size(bad)


@pytest.mark.parametrize("text,ok", [("1,2,3", True), ("1;2;3", True), ("-0.5, 2e-3 ,7", True),
                                     ("1,2", False), ("a,b,c", False), ("1,2,nan", False), ("1,2,3,4", False)])
def test_parse_point(text, ok):
    if ok:
        assert g.parse_point(text).shape == (3,)
    else:
        with pytest.raises(UsageError):
            g.parse_point(text)


def test_parse_box():
    lo, hi = g.parse_box("-0.1,0,-0.1,0.1,0.3,0.1")
    assert lo.tolist() == [-0.1, 0, -0.1] and hi.tolist() == [0.1, 0.3, 0.1]
    for bad in ("1,2,3", "0,0,0,1,1,x", "0,0,0,0,1,1", "1,1,1,0,0,0"):
        with pytest.raises(UsageError):
            g.parse_box(bad)


# ------------------------------------------------------------------ orientation
def orbit_rights_ups(true_up, n=36, tilt_deg=30, roll_deg=3, seed=0):
    rng = np.random.default_rng(seed)
    up = np.asarray(true_up, float) / np.linalg.norm(true_up)
    a = np.cross(up, [1, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    rights, ups = [], []
    for k in range(n):
        th = 2 * math.pi * k / n
        pos = math.cos(th) * a + math.sin(th) * b
        fwd = -pos * math.cos(math.radians(tilt_deg)) - up * math.sin(math.radians(tilt_deg))
        r = np.cross(fwd, up); r /= np.linalg.norm(r)
        u = np.cross(r, fwd)
        roll = math.radians(rng.normal(0, roll_deg))
        rights.append(math.cos(roll) * r + math.sin(roll) * u)
        ups.append(-math.sin(roll) * r + math.cos(roll) * u)
    return np.array(rights), np.array(ups)


@pytest.mark.parametrize("true_up", [[0, 0, 1], [0, -1, 0], [0.3, 0.2, 0.93], [1, 0, 0]])
def test_up_from_cameras_recovers_up(true_up):
    rights, ups = orbit_rights_ups(true_up)
    up = g.up_from_cameras(rights, ups)
    t = np.asarray(true_up, float) / np.linalg.norm(true_up)
    assert math.degrees(math.acos(np.clip(np.dot(up, t), -1, 1))) < 3


def test_up_from_cameras_needs_two():
    with pytest.raises(ValueError):
        g.up_from_cameras(np.zeros((1, 3)), np.zeros((1, 3)))


@pytest.mark.parametrize("a,b", [([1, 0, 0], [0, 0, 1]), ([0, 0, 1], [0, 0, 1]), ([0, 0, 1], [0, 0, -1]),
                                 ([0.2, 0.5, -0.8], [0.9, -0.1, 0.3])])
def test_rotation_between(a, b):
    R = g.rotation_between(np.array(a, float), np.array(b, float))
    an, bn = np.array(a) / np.linalg.norm(a), np.array(b) / np.linalg.norm(b)
    assert np.allclose(R @ an, bn, atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)


def test_place_stands_model_on_floor_centred_and_facing_first_camera():
    rng = np.random.default_rng(1)
    up = np.array([0.0, -1.0, 0.0])                 # model frame with -Y up (as COLMAP often is)
    pts = rng.random((500, 3)) * [0.2, 0.5, 0.2] + [3, -2, 1]
    cam = np.array([3.1, -2.6, 4.0])                # first camera, off to +Z
    sim = g.place(pts, up, cam)
    out = sim.apply(pts)
    assert out[:, 1].min() == pytest.approx(0.0, abs=1e-9)
    assert out[:, 1].max() - out[:, 1].min() == pytest.approx(0.5, rel=0.05)
    assert (out[:, 0].min() + out[:, 0].max()) / 2 == pytest.approx(0.0, abs=1e-9)
    c = sim.apply(cam[None])[0]
    assert c[2] > 0 and abs(c[0]) < 0.01 * abs(c[2])


def test_similarity_then_scale():
    sim = g.Similarity(np.eye(3), np.array([1.0, 0, 0]))
    s2 = sim.then_scale(2.0)
    assert np.allclose(s2.apply([[1, 1, 1]]), [[4, 2, 2]])
    assert s2.to_json()["scale"] == 2.0


# ------------------------------------------------------------------ cleaning
def table_and_box(seed=0):
    rng = np.random.default_rng(seed)
    table = np.c_[rng.uniform(-1, 1, (3000, 2)), rng.normal(0, 0.002, 3000)]
    box = rng.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 0.5], (1500, 3))
    return table, box


def test_ground_plane_found_and_points_up():
    table, box = table_and_box()
    plane = g.ground_plane(np.r_[table, box], np.array([0.1, 0, 1.0]), threshold=0.01)
    assert plane is not None
    n, d = plane
    assert n[2] > 0.999 and abs(d) < 0.005


def test_ground_plane_rejects_walls_and_too_few_points():
    rng = np.random.default_rng(2)
    wall = np.c_[rng.normal(0, 0.002, 2000), rng.uniform(-1, 1, (2000, 2))]
    assert g.ground_plane(wall, np.array([0, 0, 1.0]), threshold=0.01) is None
    assert g.ground_plane(wall[:10], np.array([1.0, 0, 0]), threshold=0.01) is None


def test_remove_below_plane():
    v = np.array([[0, 0, -0.1], [1, 0, -0.1], [0, 1, -0.1], [0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]], float)
    f = np.array([[0, 1, 2], [3, 4, 5], [0, 4, 5]])
    keep = g.remove_below_plane(v, f, (np.array([0, 0, 1.0]), 0.0), margin=0.01)
    assert keep.tolist() == [False, True, False]


def two_blobs():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],         # a 2-triangle square
                  [5, 5, 5], [5.1, 5, 5], [5, 5.1, 5]], float)          # a small far triangle
    f = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    return v, f


def test_largest_component_by_area():
    v, f = two_blobs()
    assert g.largest_component(f, len(v), g.triangle_areas(v, f)).tolist() == [True, True, False]


def test_largest_component_empty():
    assert g.largest_component(np.zeros((0, 3), int), 0).tolist() == []


def test_compact_reindexes_and_filters_extras():
    v, f = two_blobs()
    cols = np.arange(len(v) * 3).reshape(-1, 3)
    v2, f2, c2 = g.compact(v, f, [False, False, True], cols)
    assert len(v2) == 3 and f2.tolist() == [[0, 1, 2]] and c2.tolist() == cols[4:].tolist()


def test_inside_box():
    v, f = two_blobs()
    assert g.inside_box(v, f, np.array([-1, -1, -1.0]), np.array([2, 2, 2.0])).tolist() == [True, True, False]


def test_near_points_drops_far_fins():
    v, f = two_blobs()
    support = np.array([[0.5, 0.5, 0.0]])
    keep = g.near_points(v, f, support, 1.0)
    assert keep.tolist() == [True, True, False]


def test_box_is_applied_before_the_height_is_set():
    from types import SimpleNamespace
    from scan_studio.pipeline import place_scale_crop
    import trimesh
    # a tall box standing on z = 0 (model frame, z up): 1 x 1 x 4 units
    m = trimesh.creation.box(extents=[1, 1, 4]); m.vertices[:, 2] += 2
    m = m.subdivide().subdivide().subdivide()
    o = SimpleNamespace(height=0.2, longest=None, width=None, points=None, length=None,
                        box="-10,0,-10,10,0.1,10")    # keep the bottom 0.1 m of the 0.2 m tall model
    place, info, V, F, _ = place_scale_crop(o, np.asarray(m.vertices), np.asarray(m.faces), None,
                                            np.array([0, 0, 1.0]), np.array([0, -10.0, 1.0]))
    out = place.apply(V)
    assert out[:, 1].min() == pytest.approx(0.0, abs=1e-9)
    assert out[:, 1].max() == pytest.approx(0.2)          # the delivered part has the height asked for
    assert info["method"] == "height"
    assert len(F) < len(m.faces)


def test_box_with_points_keeps_the_points_scale():
    from types import SimpleNamespace
    from scan_studio.pipeline import place_scale_crop
    import trimesh
    m = trimesh.creation.box(extents=[1, 1, 4]); m.vertices[:, 2] += 2
    m = m.subdivide().subdivide().subdivide()
    o = SimpleNamespace(height=None, longest=None, width=None, points=("0,0,0", "0,1,0"), length=0.05,
                        box="-10,0,-10,10,0.1,10")
    place, info, V, _, _ = place_scale_crop(o, np.asarray(m.vertices), np.asarray(m.faces), None,
                                            np.array([0, 0, 1.0]), np.array([0, -10.0, 1.0]))
    assert place.scale == pytest.approx(0.05) and info["method"] == "points"
    assert place.apply(V)[:, 1].max() == pytest.approx(0.1, abs=0.01)
