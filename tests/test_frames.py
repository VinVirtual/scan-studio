import numpy as np
import pytest

from scan_studio import frames as fr
from conftest import blurred, textured


def test_blur_score_drops_as_blur_grows():
    img = textured()
    s = [fr.blur_score(blurred(img, sig)) if sig else fr.blur_score(img) for sig in (0, 1, 2, 4)]
    assert s == sorted(s, reverse=True)
    assert s[0] > 5 * s[3]


def test_blur_score_is_comparable_across_resolutions():
    import cv2
    img = textured(720, 1280)
    small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    a, b = fr.blur_score(img), fr.blur_score(small)
    assert 0.5 < a / b < 2.0


def test_blur_score_accepts_grey_and_rejects_empty():
    img = textured()[:, :, 0]
    assert fr.blur_score(img) > 0
    with pytest.raises(ValueError):
        fr.blur_score(np.zeros((0, 0), np.uint8))


def test_flat_image_scores_zero():
    assert fr.blur_score(np.full((100, 200, 3), 128, np.uint8)) == pytest.approx(0.0, abs=1e-6)


def test_select_spreads_over_the_whole_capture():
    scores = np.full(300, 100.0)
    sel = fr.select_frames(scores, 30)
    assert len(sel.chosen) == 30
    gaps = np.diff(sel.chosen)
    assert gaps.min() >= 5 and gaps.max() <= 15
    assert sel.chosen[0] < 10 and sel.chosen[-1] > 289
    assert fr.spread(sel.chosen, 300) > 0.7


def test_select_takes_the_sharpest_in_each_part():
    scores = np.full(100, 50.0)
    scores[[3, 14, 25]] = 90.0
    sel = fr.select_frames(scores, 10)
    assert {3, 14, 25} <= set(sel.chosen)


def test_select_rejects_blurry_frames_relative_to_neighbours():
    scores = np.full(120, 100.0)
    scores[40:46] = 10.0          # a quick jerk of the hand
    sel = fr.select_frames(scores, 120)
    assert not set(range(40, 46)) & set(sel.chosen)
    assert set(range(40, 46)) <= set(sel.blurry)


def test_a_dim_but_steady_capture_is_not_rejected():
    scores = np.full(80, 3.0)     # low absolute sharpness everywhere
    sel = fr.select_frames(scores, 20)
    assert len(sel.chosen) == 20 and not sel.blurry


def test_part_with_only_blurry_frames_is_skipped():
    scores = np.full(100, 100.0)
    scores[50:60] = 1.0
    sel = fr.select_frames(scores, 10, radius=30)
    assert all(not 50 <= i < 60 for i in sel.chosen)
    assert len(sel.chosen) == 9


def test_still_camera_duplicates_are_skipped():
    still = np.zeros((fr.THUMB, fr.THUMB), np.float32)
    thumbs = [still + 0.00001 * i for i in range(40)]
    sel = fr.select_frames(np.full(40, 10.0), 10, thumbs=thumbs)
    assert len(sel.chosen) == 1
    assert sel.duplicates


def test_moving_camera_is_not_taken_for_still():
    rng = np.random.default_rng(1)
    thumbs = [rng.random((fr.THUMB, fr.THUMB)).astype(np.float32) for _ in range(40)]
    sel = fr.select_frames(np.full(40, 10.0), 10, thumbs=thumbs)
    assert len(sel.chosen) == 10 and not sel.duplicates


def test_target_larger_than_candidates():
    sel = fr.select_frames([5.0, 6.0, 7.0], 50)
    assert sel.chosen == (0, 1, 2)


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [[1.0, 2.0]], [1.0, float("inf")]])
def test_select_refuses_bad_scores(bad):
    with pytest.raises(ValueError):
        fr.select_frames(bad, 3)


def test_select_refuses_bad_target_and_thumb_count():
    with pytest.raises(ValueError):
        fr.select_frames([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        fr.select_frames([1.0, 2.0], 1, thumbs=[np.zeros((2, 2))])


def test_thresholds_follow_local_median():
    scores = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)])
    sel = fr.select_frames(scores, 10, min_relative=0.5, radius=5)
    assert sel.threshold[0] == pytest.approx(5.0)
    assert sel.threshold[-1] == pytest.approx(500.0)


def test_spread_of_clumped_frames_is_low():
    assert fr.spread([0, 1, 2, 3], 1000) < fr.spread([0, 333, 666, 999], 1000)
    assert fr.spread([5], 10) == 0.0
