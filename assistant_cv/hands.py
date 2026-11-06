"""MediaPipe HandLandmarker wrapper and scale/translation-invariant normalization."""

from __future__ import annotations

import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

try:
    from .config import HAND_MODEL_PATH, HAND_MODEL_URL
except ImportError:
    from config import HAND_MODEL_PATH, HAND_MODEL_URL

HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
if not HAND_MODEL_PATH.exists():
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

_h = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=2,
    )
)


def landmarks(frame_bgr: np.ndarray) -> np.ndarray:
    """Extract 2-hand 21-point 3D landmarks into shape (2, 21, 3) float32."""
    out = np.zeros((2, 21, 3), np.float32)
    if frame_bgr is None or frame_bgr.size == 0:
        return out

    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _h.detect(img)
    for h_lm, h_cls in zip(res.hand_landmarks, res.handedness):
        slot = 0 if h_cls[0].category_name == "Left" else 1
        if out[slot].any():
            slot = 1 - slot
        if not out[slot].any():
            out[slot] = [[p.x, p.y, p.z] for p in h_lm]
    return out


def norm(x: np.ndarray) -> np.ndarray:
    """Wrist-relative, palm-scale-invariant normalization returning (126,) float32."""
    x = np.asarray(x, dtype=np.float32).reshape(2, 21, 3).copy()
    for h in range(2):
        if x[h].any():
            x[h] -= x[h][0]
            x[h] /= np.linalg.norm(x[h][9]) + 1e-6
    return x.reshape(-1).astype(np.float32)


def extract(frame_bgr: np.ndarray) -> np.ndarray:
    """Convenience helper returning normalized (126,) hand vector from a BGR frame."""
    return norm(landmarks(frame_bgr))
