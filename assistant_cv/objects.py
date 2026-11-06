"""Step 1 — Real-Time Object & Person Detection (YOLO26s) for BMO-CV."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from assistant_cv import hud
from assistant_cv.config import CAM_INDEX, DEVICE, OBJ_CONF, YOLO_MODEL_PATH


class ObjectDetector:
    """YOLO26s wrapper for real-time 80-class COCO object and person detection."""

    def __init__(
        self,
        model_path: pathlib.Path | str = YOLO_MODEL_PATH,
        conf: float = OBJ_CONF,
        device: str = DEVICE,
    ) -> None:
        model_path_obj = pathlib.Path(model_path)
        model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path_obj
        self.det = YOLO(str(model_path_obj))
        self.conf = float(conf)
        self.device = str(device)
        self.names: dict[int, str] = self.det.names

    def detect(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        persons_only: bool = False,
        track: bool = False,
    ) -> list[dict[str, Any]]:
        """Detect (and optionally track) objects in a BGR frame.

        Returns a list of dicts:
            {"label": str, "conf": float, "bbox": (x1, y1, x2, y2), "id": int | None}
        """
        if frame is None:
            return []

        conf_thresh = float(conf if conf is not None else self.conf)
        classes = [0] if persons_only else None
        use_half = self.device == "cuda"

        if track:
            results = self.det.track(
                frame,
                conf=conf_thresh,
                classes=classes,
                persist=True,
                verbose=False,
                device=self.device,
            )
        else:
            results = self.det.predict(
                frame,
                conf=conf_thresh,
                classes=classes,
                verbose=False,
                device=self.device,
            )

        if not results:
            return []

        r = results[0]
        if r.boxes is None:
            return []

        out: list[dict[str, Any]] = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls_id = int(b.cls[0])
            name = self.names[cls_id]
            score = round(float(b.conf[0]), 3)
            track_id = int(b.id[0]) if b.id is not None else None
            out.append(
                {
                    "label": name,
                    "conf": score,
                    "bbox": (x1, y1, x2, y2),
                    "id": track_id,
                }
            )
        return out


_detector: ObjectDetector | None = None


def get_detector() -> ObjectDetector:
    """Return the lazily initialized module-level ObjectDetector singleton."""
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector


def detect_objects(
    frame: np.ndarray,
    conf: float | None = None,
    persons_only: bool = False,
    track: bool = False,
) -> list[dict[str, Any]]:
    """Convenience function to run object detection using the shared singleton."""
    return get_detector().detect(
        frame, conf=conf, persons_only=persons_only, track=track
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BMO-CV Step 1: Real-Time Object & Person Detection (YOLO26s)"
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=CAM_INDEX,
        help=f"Camera index (default: {CAM_INDEX})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=OBJ_CONF,
        help=f"Confidence threshold (default: {OBJ_CONF})",
    )
    parser.add_argument(
        "--persons",
        action="store_true",
        help="Start in persons-only mode (COCO class 0)",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Start with multi-object tracking enabled",
    )
    args = parser.parse_args()

    detector = ObjectDetector(conf=args.conf, device=DEVICE)
    persons_only = bool(args.persons)
    track = bool(args.track)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.cam}")

    fps = 0.0
    prev_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t0 = time.perf_counter()
            detections = detector.detect(
                frame,
                conf=args.conf,
                persons_only=persons_only,
                track=track,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

            hud.draw_objects(frame, detections)
            hud.draw_status_bar(
                frame,
                fps,
                device=DEVICE,
                toggles={"P:PERS": persons_only, "T:TRK": track},
                latencies={"OBJ": infer_ms},
            )

            cv2.imshow("BMO — Step 1: Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("p"):
                persons_only = not persons_only
            elif key == ord("t"):
                track = not track
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
