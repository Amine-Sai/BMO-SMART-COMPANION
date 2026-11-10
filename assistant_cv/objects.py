"""Real-Time Object & Person Detection with Open-Vocabulary YOLO-World support."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from typing import Any, Sequence
import warnings

# Suppress library deprecation and empty line warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import cv2
import numpy as np
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from assistant_cv import hud
from assistant_cv.config import (
    CAM_INDEX,
    DEFAULT_OBJECT_CLASSES,
    DEVICE,
    MAX_FPS,
    OBJ_CONF,
    OBJ_FRAME_STRIDE,
    OBJ_MODEL_TYPE,
    YOLO_MODEL_PATH,
    YOLO_WORLD_MODEL_PATH,
)

try:
    from .registry import get_object_registry
except ImportError:
    from registry import get_object_registry


class ObjectDetector:
    """YOLO detector supporting standard COCO and open-vocabulary YOLO-World models."""

    def __init__(
        self,
        model_path: pathlib.Path | str | None = None,
        conf: float = OBJ_CONF,
        device: str = DEVICE,
        model_type: str = OBJ_MODEL_TYPE,
        classes: Sequence[str] | None = None,
        registry: Any = None,
    ) -> None:
        self.device = str(device)
        self.conf = float(conf)
        self.model_type = str(model_type).lower()
        self.registry = registry if registry is not None else get_object_registry()

        # Resolve model path
        if model_path is not None:
            resolved_path = pathlib.Path(model_path)
        elif self.model_type == "world" and YOLO_WORLD_MODEL_PATH.exists():
            resolved_path = YOLO_WORLD_MODEL_PATH
        else:
            resolved_path = YOLO_MODEL_PATH

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = resolved_path
        self.det = YOLO(str(resolved_path))
        self.is_world = "world" in str(resolved_path).lower()

        if self.is_world:
            target_classes = list(classes) if classes is not None else DEFAULT_OBJECT_CLASSES
            self.set_classes(target_classes)
        else:
            self.names = dict(self.det.names)

    @property
    def is_world_model(self) -> bool:
        return self.is_world

    @property
    def class_names(self) -> list[str]:
        return list(self.names.values())

    def set_classes(self, classes: Sequence[str]) -> None:
        """Dynamically set detection classes on open-vocabulary YOLO-World model."""
        clean_classes = [str(c).strip() for c in classes if str(c).strip()]
        if not clean_classes:
            return
        if hasattr(self.det, "set_classes"):
            try:
                self.det.set_classes(clean_classes)
                self.names = dict(self.det.names)
            except Exception as exc:
                print(f"[ObjectDetector] set_classes warning: {exc}", file=sys.stderr)
        else:
            self.names = {i: c for i, c in enumerate(clean_classes)}

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Extract a 512-D L2-normalized visual embedding from an object image crop."""
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        h, w = crop.shape[:2]
        if h < 32 or w < 32:
            crop = cv2.resize(crop, (max(32, w), max(32, h)), interpolation=cv2.INTER_LINEAR)
        try:
            feat = self.det.embed(crop)
            if isinstance(feat, (list, tuple)) and feat:
                vec = feat[0].detach().cpu().numpy().flatten().astype(np.float32)
            elif hasattr(feat, "detach"):
                vec = feat.detach().cpu().numpy().flatten().astype(np.float32)
            else:
                vec = np.asarray(feat, dtype=np.float32).flatten()
            norm_val = float(np.linalg.norm(vec))
            if norm_val > 1e-8:
                vec = vec / norm_val
            return vec
        except Exception as exc:
            print(f"[ObjectDetector] embed_crop warning: {exc}", file=sys.stderr)
            return np.zeros(512, dtype=np.float32)

    def register_class(self, name: str, synonyms: Sequence[str] | None = None) -> list[str]:
        """Register a new class into active vocabulary and update YOLO-World text prompts."""
        clean_name = str(name).strip()
        if not clean_name:
            return self.class_names
        current = list(self.class_names)
        current_lower = {c.lower() for c in current}
        if clean_name.lower() not in current_lower:
            current.append(clean_name)
            current_lower.add(clean_name.lower())
        if synonyms:
            for syn in synonyms:
                s = str(syn).strip()
                if s and s.lower() not in current_lower:
                    current.append(s)
                    current_lower.add(s.lower())
        self.set_classes(current)
        if self.registry is not None:
            self.registry.register_object(clean_name, synonyms=list(synonyms) if synonyms else None)
        return self.class_names

    def reload_classes(self) -> list[str]:
        """Reload vocabulary from everyday_objects.json and visual gallery from ObjectRegistry."""
        try:
            from .config import load_object_classes
        except ImportError:
            from config import load_object_classes
        classes = load_object_classes()
        self.set_classes(classes)
        if self.registry is not None:
            self.registry.reload_gallery()
        return self.class_names

    def detect(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        persons_only: bool = False,
        track: bool = False,
    ) -> list[dict[str, Any]]:
        """Detect (and optionally track) objects in a BGR frame.

        Returns list of dicts:
            {"label": str, "conf": float, "bbox": (x1, y1, x2, y2), "id": int | None}
        """
        if frame is None or frame.size == 0:
            return []

        conf_thresh = float(conf if conf is not None else self.conf)

        # In YOLO-World / custom models, find class ID for 'person' if persons_only is set
        classes_filter = None
        if persons_only:
            person_ids = [idx for idx, name in self.names.items() if str(name).lower() == "person"]
            classes_filter = person_ids if person_ids else [0]

        if track:
            results = self.det.track(
                frame,
                conf=conf_thresh,
                iou=0.45,
                classes=classes_filter,
                persist=True,
                verbose=False,
                device=self.device,
            )
        else:
            results = self.det.predict(
                frame,
                conf=conf_thresh,
                iou=0.45,
                classes=classes_filter,
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
            name = self.names.get(cls_id, f"class_{cls_id}")
            score = round(float(b.conf[0]), 3)

            # Higher bar for person detection to avoid false human tracks
            if str(name).lower() == "person" and score < max(0.35, conf_thresh):
                continue

            track_id = int(b.id[0]) if b.id is not None else None
            registered_name: str | None = None

            # Check if this detection matches a registered visual exemplar
            if (
                self.registry is not None
                and getattr(self.registry, "_gallery_matrix", None) is not None
                and self.registry._gallery_matrix.shape[0] > 0
                and (x2 - x1 >= 24 and y2 - y1 >= 24)
            ):
                crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
                if crop.size > 0:
                    try:
                        crop_emb = self.embed_crop(crop)
                        matched_name, _ = self.registry.match_visual(crop_emb)
                        if matched_name is not None:
                            registered_name = matched_name
                    except Exception:
                        pass

            item: dict[str, Any] = {
                "label": name,
                "conf": score,
                "bbox": (x1, y1, x2, y2),
                "id": track_id,
            }
            if registered_name:
                item["registered_name"] = registered_name
            out.append(item)
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
    """Convenience function using the shared singleton."""
    return get_detector().detect(
        frame, conf=conf, persons_only=persons_only, track=track
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BMO-CV Object & Person Detection (YOLO-World & COCO)"
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
        help="Start in persons-only mode",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Start with multi-object tracking enabled",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OBJ_MODEL_TYPE,
        choices=["world", "yolo26"],
        help="Detection model: 'world' (80+ everyday objects) or 'yolo26' (COCO 80)",
    )
    args = parser.parse_args()

    detector = ObjectDetector(conf=args.conf, device=DEVICE, model_type=args.model)
    persons_only = bool(args.persons)
    track = bool(args.track)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.cam}")

    fps = 0.0
    prev_t = time.perf_counter()
    frame_idx = 0
    cached_detections: list[dict[str, Any]] = []

    try:
        while True:
            t_loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            infer_ms = 0.0
            if (frame_idx % OBJ_FRAME_STRIDE == 0) or not cached_detections:
                t0 = time.perf_counter()
                detections = detector.detect(
                    frame,
                    conf=args.conf,
                    persons_only=persons_only,
                    track=track,
                )
                infer_ms = (time.perf_counter() - t0) * 1000.0
                cached_detections = detections
            else:
                detections = cached_detections

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
                toggles={"P:PERS": persons_only, "T:TRK": track, f"M:{args.model.upper()}": True},
                latencies={"OBJ": infer_ms},
            )

            cv2.imshow("BMO — Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("p"):
                persons_only = not persons_only
            elif key == ord("t"):
                track = not track

            # 60 FPS cap
            elapsed = time.perf_counter() - t_loop_start
            sleep_needed = (1.0 / MAX_FPS) - elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
