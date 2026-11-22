"""Objects only, never people: look for faces in the frames and refuse when a person is in the capture.

Uses OpenCV's Haar cascades (frontal and profile), which ship inside opencv-python. They are fast on a CPU and
miss some faces (turned away, small, dark), so this is a guard against obvious mistakes, not a guarantee."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

DETECT_WIDTH = 640
MIN_FACE = 40          # pixels at DETECT_WIDTH; smaller faces are background noise for this purpose
# (model, neighbours needed). alt2 is the most precise frontal model; the default frontal and the profile model
# fire on busy patterns (a painted jug did it), so they must agree with themselves much more strongly.
CASCADES = (("haarcascade_frontalface_alt2.xml", 6), ("haarcascade_frontalface_default.xml", 15),
            ("haarcascade_profileface.xml", 15))
_cascades: dict[str, cv2.CascadeClassifier] = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _cascades:
        path = os.path.join(cv2.data.haarcascades, name)
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(f"OpenCV face model {name} could not be loaded")
        _cascades[name] = c
    return _cascades[name]


def detect(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h) in the image's own pixels."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = grey.shape[:2]
    s = DETECT_WIDTH / w
    small = cv2.resize(grey, (DETECT_WIDTH, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.equalizeHist(small)
    boxes = []
    for name, neighbours in CASCADES:
        found = _cascade(name).detectMultiScale(small, scaleFactor=1.1, minNeighbors=neighbours,
                                                minSize=(MIN_FACE, MIN_FACE))
        boxes += [tuple(int(round(v / s)) for v in b) for b in (found if len(found) else [])]
    return boxes


@dataclass
class FaceReport:
    frames_checked: int = 0
    frames_with_faces: list[int] = field(default_factory=list)
    boxes: list[dict] = field(default_factory=list)
    verdict: str = "clear"            # clear | warning | refused | confirmed-object
    message: str = ""

    def to_json(self) -> dict:
        return {"frames_checked": self.frames_checked, "frames_with_faces": self.frames_with_faces,
                "boxes": self.boxes, "verdict": self.verdict, "message": self.message}


def refuse_threshold(frames_checked: int) -> int:
    """Faces in this many frames means a person is in the capture: at least 2 frames, or 5 % of them."""
    return max(2, int(np.ceil(0.05 * frames_checked)))


def judge(per_frame: list[list[tuple[int, int, int, int]]], *, confirmed_object: bool, max_boxes: int = 200) -> FaceReport:
    rep = FaceReport(frames_checked=len(per_frame))
    for i, boxes in enumerate(per_frame):
        if boxes:
            rep.frames_with_faces.append(i)
            for b in boxes:
                if len(rep.boxes) < max_boxes:
                    rep.boxes.append({"frame": i, "box": list(b)})
    hits = len(rep.frames_with_faces)
    if hits == 0:
        rep.message = f"no faces found in {rep.frames_checked} frames"
    elif hits < refuse_threshold(rep.frames_checked):
        rep.verdict = "warning"
        rep.message = (f"a face-like pattern in {hits} of {rep.frames_checked} frames; probably part of the object "
                       "or the background, but check the frames")
    elif confirmed_object:
        rep.verdict = "confirmed-object"
        rep.message = (f"faces in {hits} of {rep.frames_checked} frames; the user confirmed they belong to an object "
                       "(a statue, a toy, a print), not a person")
    else:
        rep.verdict = "refused"
        rep.message = (f"faces in {hits} of {rep.frames_checked} frames. scan-studio makes models of objects, never of "
                       "people. If the face belongs to an object (a statue, a doll, a printed picture) and no person is "
                       "in the frames, run again with --confirm-object")
    return rep


def check_frames(frames, *, confirmed_object: bool, max_boxes: int = 200) -> FaceReport:
    return judge([detect(f) for f in frames], confirmed_object=confirmed_object, max_boxes=max_boxes)
