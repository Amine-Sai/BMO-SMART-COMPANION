"""Standalone Hand Landmark pipeline using OpenCV Zoo MediaPipe ONNX models.

Replaces the crashing C++ MediaPipe Tasks runtime with pure OpenCV DNN inference.
Keeps identical public API:
  - landmarks(frame_bgr) -> (2, 21, 3) float32
  - norm(x) -> (126,) float32
"""

from __future__ import annotations

import pathlib
import numpy as np

try:
    from .config import (
        HANDPOSE_MODEL_PATH,
        PALM_MODEL_PATH,
    )
    from .mp_handpose import MPHandPose
    from .mp_palmdet import MPPalmDet
except ImportError:
    from config import (
        HANDPOSE_MODEL_PATH,
        PALM_MODEL_PATH,
    )
    from mp_handpose import MPHandPose
    from mp_palmdet import MPPalmDet


def _require_model(path: pathlib.Path, model_name: str) -> pathlib.Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{model_name} model not found at {path}. "
            "Runtime downloads are disabled. Run 'python tools/fetch_assets.py' to download pinned model assets."
        )
    return path


_palm_detector: MPPalmDet | None = None
_handpose_detector: MPHandPose | None = None


def get_detectors() -> tuple[MPPalmDet, MPHandPose]:
    """Lazy singleton initializer for palm and hand pose detectors."""
    global _palm_detector, _handpose_detector
    if _palm_detector is None:
        p_path = _require_model(PALM_MODEL_PATH, "Palm detection")
        _palm_detector = MPPalmDet(
            modelPath=str(p_path),
            nmsThreshold=0.3,
            scoreThreshold=0.6,
        )
    if _handpose_detector is None:
        h_path = _require_model(HANDPOSE_MODEL_PATH, "Handpose estimation")
        _handpose_detector = MPHandPose(
            modelPath=str(h_path),
            confThreshold=0.6,
        )
    return _palm_detector, _handpose_detector


def landmarks(frame_bgr: np.ndarray) -> np.ndarray:
    """Extract 2-hand 21-point 3D landmarks into shape (2, 21, 3) float32.

    Slot 0 = Left hand, Slot 1 = Right hand.
    Coordinates are normalized into [0, 1] across image width and height.
    """
    out = np.zeros((2, 21, 3), dtype=np.float32)
    if frame_bgr is None or frame_bgr.size == 0:
        return out

    h_img, w_img = frame_bgr.shape[:2]
    palm_net, hand_net = get_detectors()

    palms = palm_net.infer(frame_bgr)
    if palms is None or len(palms) == 0:
        return out

    for palm in palms[:2]:  # evaluate up to top 2 candidate palms
        handpose = hand_net.infer(frame_bgr, palm)
        if handpose is None:
            continue

        # handpose layout: [bbox(4), landmarks_screen(63), landmarks_world(63), handedness(1), conf(1)]
        handedness = float(handpose[-2])
        conf = float(handpose[-1])
        if conf < 0.5:
            continue

        # 0 = Left (<= 0.5), 1 = Right (> 0.5)
        slot = 0 if handedness <= 0.5 else 1
        if out[slot].any():
            slot = 1 - slot
        if out[slot].any():
            continue  # both slots occupied

        # landmarks_screen is (21, 3) in pixel coordinates (x, y, z)
        lm_px = handpose[4:67].reshape(21, 3).astype(np.float32)
        lm_norm = np.zeros((21, 3), dtype=np.float32)
        lm_norm[:, 0] = lm_px[:, 0] / max(1.0, float(w_img))
        lm_norm[:, 1] = lm_px[:, 1] / max(1.0, float(h_img))
        lm_norm[:, 2] = lm_px[:, 2] / max(1.0, float(w_img))  # depth normalized by width

        out[slot] = lm_norm

    return out


def norm(x: np.ndarray) -> np.ndarray:
    """Wrist-relative, palm-scale-invariant normalization returning (126,) float32."""
    x = np.asarray(x, dtype=np.float32).reshape(2, 21, 3).copy()
    for h in range(2):
        if x[h].any():
            x[h] -= x[h][0]
            scale = np.linalg.norm(x[h][9])  # middle finger base (point 9) distance
            x[h] /= scale + 1e-6
    return x.reshape(-1).astype(np.float32)


def extract(frame_bgr: np.ndarray) -> np.ndarray:
    """Convenience helper returning normalized (126,) hand vector from a BGR frame."""
    return norm(landmarks(frame_bgr))


def has_palms(frame_bgr: np.ndarray) -> bool:
    """Fast probe returning True if at least one candidate palm is present in frame."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    palm_net, _ = get_detectors()
    palms = palm_net.infer(frame_bgr)
    return bool(palms is not None and len(palms) > 0)
