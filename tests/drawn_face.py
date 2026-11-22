"""A face drawn from ellipses with OpenCV: no photograph, no real person. Used to test the face refusal."""
from __future__ import annotations

import cv2
import numpy as np


def drawn_face(size: int = 480, background: int = 70) -> np.ndarray:
    img = np.full((size, size, 3), background, np.uint8)
    c = size // 2
    s = size / 480
    def P(x, y):
        return (int(c + x * s), int(c + y * s))
    def A(a, b):
        return (int(a * s), int(b * s))
    cv2.ellipse(img, P(0, 10), A(120, 160), 0, 0, 360, (205, 170, 150), -1)          # head
    for x in (-50, 50):
        cv2.ellipse(img, P(x, -35), A(32, 9), 0, 180, 360, (60, 45, 40), 7)       # brows
        cv2.ellipse(img, P(x, -5), A(28, 14), 0, 0, 360, (245, 245, 245), -1)      # eye whites
        cv2.circle(img, P(x, -5), int(12 * s), (40, 30, 30), -1)                    # irises
    cv2.ellipse(img, P(0, 40), A(14, 30), 0, 0, 360, (180, 140, 125), -1)          # nose shading
    cv2.ellipse(img, P(0, 95), A(46, 16), 0, 0, 180, (120, 60, 60), 6)             # mouth
    img = cv2.GaussianBlur(img, (0, 0), 2.0 * s)
    return img
