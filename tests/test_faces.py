"""The people check. Every face here is drawn with OpenCV ellipses (tests/drawn_face.py): no photograph of anyone."""
import numpy as np

from scan_studio import faces
from conftest import textured
from drawn_face import drawn_face


def frame_with_face(seed=0, size=300):
    img = np.full((720, 1280, 3), 90, np.uint8)
    x = 200 + 100 * (seed % 5)
    img[150:150 + size, x:x + size] = drawn_face(size)
    return img


def test_drawn_face_is_detected():
    assert faces.detect(frame_with_face())


def test_detected_box_is_in_original_pixels():
    (x, y, w, h), *_ = faces.detect(frame_with_face(0, 400))
    assert 150 <= x <= 300 and 120 <= y <= 300 and 250 <= w <= 500


def test_plain_and_textured_scenes_have_no_faces():
    assert faces.detect(np.full((480, 640, 3), 128, np.uint8)) == []
    assert faces.detect(textured(480, 640, seed=3)) == []


def test_tiny_faces_are_ignored():
    img = np.full((720, 1280, 3), 90, np.uint8)
    img[10:40, 10:40] = drawn_face(30)
    assert faces.detect(img) == []


def test_refuse_threshold():
    assert faces.refuse_threshold(10) == 2
    assert faces.refuse_threshold(90) == 5
    assert faces.refuse_threshold(0) == 2


def test_judge_clear():
    rep = faces.judge([[]] * 20, confirmed_object=False)
    assert rep.verdict == "clear" and rep.frames_with_faces == []


def test_judge_single_hit_is_a_warning_not_a_refusal():
    per = [[]] * 40
    per[7] = [(1, 2, 50, 50)]
    rep = faces.judge(per, confirmed_object=False)
    assert rep.verdict == "warning" and rep.frames_with_faces == [7]


def test_judge_refuses_people():
    per = [[(0, 0, 60, 60)]] * 10 + [[]] * 10
    rep = faces.judge(per, confirmed_object=False)
    assert rep.verdict == "refused"
    assert "never" in rep.message and "--confirm-object" in rep.message


def test_confirmed_object_passes_but_is_recorded():
    per = [[(0, 0, 60, 60)]] * 10
    rep = faces.judge(per, confirmed_object=True)
    assert rep.verdict == "confirmed-object"
    assert rep.to_json()["frames_with_faces"] == list(range(10))


def test_face_boxes_are_capped():
    per = [[(i, i, 50, 50)] * 30 for i in range(30)]
    rep = faces.judge(per, confirmed_object=True, max_boxes=25)
    assert len(rep.boxes) == 25 and len(rep.frames_with_faces) == 30


def test_check_frames_end_to_end():
    frames = [frame_with_face(i) for i in range(4)] + [textured(720, 1280, i) for i in range(4)]
    rep = faces.check_frames(frames, confirmed_object=False)
    assert rep.frames_checked == 8
    assert rep.verdict == "refused"
    assert set(rep.frames_with_faces) <= set(range(4)) and len(rep.frames_with_faces) >= 3
