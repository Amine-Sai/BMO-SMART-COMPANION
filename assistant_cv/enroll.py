"""Interactive multi-shot face enrollment utility for BMO-CV."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from assistant_cv import hud
from assistant_cv.config import CAM_INDEX, DEVICE, GALLERY_DIR
from assistant_cv.faces import get_face_app

_FONT = hud._FONT


def _draw_banner(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    y_center: int | None = None,
) -> None:
    """Draw a centered translucent banner with a colored border on ``frame``."""
    h, w = frame.shape[:2]
    font_scale = 0.62
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad_x, pad_y = 16, 10
    bw = tw + pad_x * 2
    bh = th + baseline + pad_y * 2

    bx1 = max(0, (w - bw) // 2)
    bx2 = min(w, bx1 + bw)
    cy = y_center if y_center is not None else (h // 2)
    by1 = max(42, cy - bh // 2)
    by2 = min(h - 10, by1 + bh)

    roi = frame[by1:by2, bx1:bx2]
    if roi.size > 0:
        overlay = np.full_like(roi, hud.BMO_DARK, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.80, roi, 0.20, 0.0, dst=roi)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (bx1 + pad_x, by1 + pad_y + th),
        _FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_enroll_progress(
    frame: np.ndarray,
    name: str,
    captured: int,
    shots: int,
    status_msg: str,
    status_color: tuple[int, int, int],
) -> None:
    """Draw bottom progress bar and status message for enrollment."""
    h, w = frame.shape[:2]
    panel_h = 68
    margin = 12
    px1 = margin
    px2 = w - margin
    py2 = h - margin
    py1 = max(42, py2 - panel_h)

    roi = frame[py1:py2, px1:px2]
    if roi.size > 0:
        overlay = np.full_like(roi, hud.BMO_DARK, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.78, roi, 0.22, 0.0, dst=roi)

    cv2.rectangle(frame, (px1, py1), (px2, py2), hud.BMO_TEAL, 1, cv2.LINE_AA)

    header = f"ENROLL: {name}   |   Captured {captured}/{shots}"
    cv2.putText(
        frame,
        header,
        (px1 + 12, py1 + 22),
        _FONT,
        0.50,
        hud.BMO_WHITE,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        status_msg,
        (px1 + 12, py1 + 42),
        _FONT,
        0.46,
        status_color,
        1,
        cv2.LINE_AA,
    )

    bar_x1 = px1 + 12
    bar_x2 = px2 - 12
    bar_y1 = py1 + 48
    bar_y2 = py2 - 8
    bar_w = max(1, bar_x2 - bar_x1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), hud.BMO_GRAY, 1, cv2.LINE_AA)

    ratio = max(0.0, min(1.0, captured / float(max(1, shots))))
    fill_w = int(round((bar_w - 2) * ratio))
    if fill_w > 0:
        bar_color = hud.BMO_GREEN if captured >= shots else hud.BMO_CYAN
        cv2.rectangle(
            frame,
            (bar_x1 + 1, bar_y1 + 1),
            (bar_x1 + 1 + fill_w, bar_y2 - 1),
            bar_color,
            -1,
        )


def enroll(name: str, shots: int = 10, cam_index: int = CAM_INDEX) -> Path | None:
    """Run interactive webcam face enrollment for ``name`` collecting ``shots`` samples."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Enrollment name must not be empty.")
    shots = max(1, int(shots))

    app = get_face_app()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[enroll] Error: could not open camera index {cam_index}", file=sys.stderr)
        return None

    embs: list[np.ndarray] = []
    status_msg = "Position your face in frame and press SPACE to capture."
    status_color = hud.BMO_TEAL
    warned_duplicate = False
    fps = 0.0
    prev_t = time.perf_counter()

    try:
        while len(embs) < shots:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[enroll] Camera frame read failed; aborting.", file=sys.stderr)
                return None

            faces = app.get(frame)
            disp = frame.copy()

            now = time.perf_counter()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

            if len(faces) == 1:
                hud.draw_corner_box(
                    disp,
                    faces[0].bbox,
                    hud.BMO_GREEN,
                    thickness=2,
                    corner_len=16,
                    label="READY - PRESS SPACE",
                )
            elif len(faces) == 0:
                _draw_banner(disp, "No face detected", hud.BMO_AMBER)
            else:
                for f in faces:
                    hud.draw_corner_box(
                        disp,
                        f.bbox,
                        hud.BMO_RED,
                        thickness=2,
                        corner_len=16,
                        label="MULTIPLE FACES",
                    )
                _draw_banner(disp, "Multiple faces! Only 1 person allowed", hud.BMO_RED)

            hud.draw_status_bar(
                disp,
                fps=fps,
                device=DEVICE,
                toggles={"SPACE:CAPTURE": len(faces) == 1, "ESC/Q:ABORT": True},
            )
            _draw_enroll_progress(
                disp,
                name=clean_name,
                captured=len(embs),
                shots=shots,
                status_msg=status_msg,
                status_color=status_color,
            )

            cv2.imshow(f"BMO-CV — Enroll: {clean_name}", disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("[enroll] Enrollment aborted by user. No file saved.")
                return None

            if key == 32:  # SPACE
                if len(faces) == 1:
                    emb = np.asarray(faces[0].normed_embedding, dtype=np.float32).reshape(-1)
                    emb = emb / (np.linalg.norm(emb) + 1e-8)

                    if len(embs) == 0:
                        try:
                            from assistant_cv.faces import _get_identifier
                            ident = _get_identifier()
                            matched_name, score = ident.check_already_registered(emb)
                            if matched_name and matched_name.lower() != clean_name.lower() and not warned_duplicate:
                                status_msg = f"Already registered as '{matched_name}' ({score:.2f})! Press SPACE again to override."
                                status_color = hud.BMO_AMBER
                                warned_duplicate = True
                                print(f"[enroll] Warning: {status_msg}")
                                continue
                        except Exception:
                            pass

                    if embs and float(emb @ embs[-1]) > 0.996 and not warned_duplicate:
                        status_msg = "Tilt head slightly for pose diversity (press SPACE again to keep)"
                        status_color = hud.BMO_AMBER
                        warned_duplicate = True
                        print(f"[enroll] Warning: {status_msg}")
                    else:
                        embs.append(emb)
                        warned_duplicate = False
                        status_msg = f"Captured shot {len(embs)}/{shots}!"
                        status_color = hud.BMO_GREEN
                        print(f"[enroll] Captured {len(embs)}/{shots}")
                elif len(faces) == 0:
                    status_msg = "Cannot capture: No face detected!"
                    status_color = hud.BMO_AMBER
                else:
                    status_msg = "Cannot capture: Multiple faces! Only 1 person allowed."
                    status_color = hud.BMO_RED
    finally:
        cap.release()
        cv2.destroyAllWindows()

    e = np.mean(embs, axis=0)
    e = e / (np.linalg.norm(e) + 1e-8)
    e = e.astype(np.float32)

    try:
        try:
            from .faces import _get_identifier
        except ImportError:
            from faces import _get_identifier
        ident = _get_identifier()
        out_path = ident.save_identity(clean_name, embs)
    except Exception:
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        out_path = GALLERY_DIR / f"{clean_name}.npy"
        np.save(out_path, e)

    sims = [float(shot @ e) for shot in embs]
    mean_sim = float(np.mean(sims))
    min_sim = float(np.min(sims))
    print(f"[enroll] Saved embedding to: {out_path}")
    print(
        f"[enroll] Self-verification cosine similarity across {len(embs)} shots: "
        f"mean={mean_sim:.4f}, min={min_sim:.4f}"
    )
    return out_path


class AutoEnrollSession:
    """Non-blocking automated face enrollment session triggered by voice or MQTT."""

    def __init__(
        self,
        name: str,
        target_shots: int = 8,
        min_interval_sec: float = 0.15,
        timeout_sec: float = 20.0,
        identifier: Any = None,
    ) -> None:
        self.name = name.strip()
        self.target_shots = max(1, int(target_shots))
        self.min_interval_sec = float(min_interval_sec)
        self.timeout_sec = float(timeout_sec)
        self.start_time = time.perf_counter()
        self.last_capture_time: float = 0.0
        self.embeddings: list[np.ndarray] = []
        self.completed: bool = False
        self.failed: bool = False
        self.already_registered: bool = False
        self.already_registered_as: str | None = None
        self.error_msg: str | None = None
        self.saved_path: Path | None = None
        self.status_msg: str = f"Starting auto-enroll for {self.name}..."
        self._identifier = identifier

    @property
    def progress(self) -> float:
        return min(1.0, len(self.embeddings) / float(self.target_shots))

    def step(self, raw_faces: list[Any]) -> bool:
        """Process one frame's detected InsightFace faces. Returns True if active, False if finished."""
        if self.completed or self.failed:
            return False

        now = time.perf_counter()
        if now - self.start_time > self.timeout_sec:
            self.failed = True
            self.error_msg = f"Auto-enroll timeout after {self.timeout_sec:.1f}s ({len(self.embeddings)}/{self.target_shots} captured)"
            self.status_msg = self.error_msg
            return False

        if not raw_faces:
            self.status_msg = f"Looking for {self.name}'s face ({len(self.embeddings)}/{self.target_shots})..."
            return True

        if len(raw_faces) > 1:
            self.status_msg = f"Multiple faces in view ({len(raw_faces)})! Please face camera alone."
            return True

        # Exactly 1 face
        f = raw_faces[0]
        if not hasattr(f, "normed_embedding"):
            return True

        if now - self.last_capture_time < self.min_interval_sec:
            return True

        emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
        norm_val = float(np.linalg.norm(emb))
        if norm_val < 1e-6:
            return True
        emb = emb / norm_val

        # Check if this person is already registered in the gallery before proceeding
        if len(self.embeddings) == 0:
            try:
                try:
                    from .faces import _get_identifier
                except ImportError:
                    from faces import _get_identifier
                ident = self._identifier or _get_identifier()
                if ident is not None and hasattr(ident, "check_already_registered"):
                    matched_name, score = ident.check_already_registered(emb)
                    if matched_name is not None:
                        if matched_name.lower() != self.name.lower():
                            self.failed = True
                            self.already_registered = True
                            self.already_registered_as = matched_name
                            self.error_msg = f"Person is already registered as '{matched_name}' (conf: {score:.2f})"
                            self.status_msg = f"Already registered as '{matched_name}'!"
                            print(f"[auto-enroll] Aborted: {self.error_msg}")
                            return False
                        else:
                            print(f"[auto-enroll] Updating embeddings for registered person '{matched_name}'")
            except Exception as e:
                print(f"[auto-enroll] Pre-check error: {e}", file=sys.stderr)

        # Check pose diversity vs last shot
        if self.embeddings:
            sim = float(emb @ self.embeddings[-1])
            if sim > 0.998:
                return True

        self.embeddings.append(emb)
        self.last_capture_time = now
        self.status_msg = f"Enrolling {self.name}: {len(self.embeddings)}/{self.target_shots}"

        if len(self.embeddings) >= self.target_shots:
            try:
                try:
                    from .faces import _get_identifier
                except ImportError:
                    from faces import _get_identifier
                ident = self._identifier or _get_identifier()
                self.saved_path = ident.save_identity(self.name, self.embeddings)
                self.completed = True
                self.status_msg = f"Successfully enrolled {self.name} ({len(self.embeddings)} shots)!"
                return False
            except Exception as exc:
                self.failed = True
                self.error_msg = str(exc)
                self.status_msg = f"Failed to save {self.name}: {exc}"
                return False

        return True


class AutoObjectEnrollSession:
    """Non-blocking automated object enrollment session triggered by voice, MQTT, or hotkey 'O'."""

    def __init__(
        self,
        name: str,
        target_shots: int = 6,
        min_interval_sec: float = 0.20,
        timeout_sec: float = 20.0,
        synonyms: list[str] | None = None,
        detector: Any = None,
        registry: Any = None,
    ) -> None:
        self.name = name.strip()
        self.target_shots = max(1, int(target_shots))
        self.min_interval_sec = float(min_interval_sec)
        self.timeout_sec = float(timeout_sec)
        self.synonyms = list(synonyms) if synonyms else []
        self.start_time = time.perf_counter()
        self.last_capture_time: float = 0.0
        self.embeddings: list[np.ndarray] = []
        self.crops: list[np.ndarray] = []
        self.completed: bool = False
        self.failed: bool = False
        self.already_registered: bool = False
        self.already_registered_as: str | None = None
        self.error_msg: str | None = None
        self.saved_path: Path | None = None
        self.status_msg: str = f"Starting object enrollment for {self.name}..."
        self._detector = detector
        self._registry = registry
        self.last_target_bbox: tuple[int, int, int, int] | None = None

    @property
    def progress(self) -> float:
        return min(1.0, len(self.embeddings) / float(self.target_shots))

    def step(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> bool:
        """Process one frame: locate target object, extract 512-D visual embedding, and accumulate shots."""
        if self.completed or self.failed:
            return False

        now = time.perf_counter()
        if now - self.start_time > self.timeout_sec:
            self.failed = True
            self.error_msg = f"Object enroll timeout after {self.timeout_sec:.1f}s ({len(self.embeddings)}/{self.target_shots} captured)"
            self.status_msg = self.error_msg
            return False

        if frame is None or frame.size == 0:
            return True

        h, w = frame.shape[:2]

        # Prioritize candidate non-person detections close to center
        candidates = [d for d in detections if str(d.get("label", "")).lower() != "person"]
        target_box: tuple[int, int, int, int] | None = None

        if candidates:
            cx_frame, cy_frame = w / 2.0, h / 2.0
            def dist_to_center(d: dict[str, Any]) -> float:
                bx1, by1, bx2, by2 = d.get("bbox", (0, 0, 0, 0))
                return ((bx1 + bx2) / 2.0 - cx_frame) ** 2 + ((by1 + by2) / 2.0 - cy_frame) ** 2

            best_det = min(candidates, key=dist_to_center)
            raw_box = best_det.get("bbox", (0, 0, 0, 0))
            target_box = (int(raw_box[0]), int(raw_box[1]), int(raw_box[2]), int(raw_box[3]))
        else:
            # Fallback to center region of frame if object is held directly before camera
            cw, ch = int(w * 0.35), int(h * 0.45)
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2
            target_box = (x1, y1, x1 + cw, y1 + ch)

        self.last_target_bbox = target_box
        bx1, by1, bx2, by2 = target_box
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        if bw < 20 or bh < 20:
            self.status_msg = f"Hold {self.name} clearly in view..."
            return True

        if now - self.last_capture_time < self.min_interval_sec:
            return True

        crop = frame[max(0, by1) : min(h, by2), max(0, bx1) : min(w, bx2)]
        if crop.size == 0:
            return True

        det = self._detector
        if det is None:
            try:
                from .objects import get_detector
            except ImportError:
                from objects import get_detector
            det = get_detector()

        reg = self._registry
        if reg is None:
            try:
                from .registry import get_object_registry
            except ImportError:
                from registry import get_object_registry
            reg = get_object_registry()

        emb = det.embed_crop(crop)
        norm_val = float(np.linalg.norm(emb))
        if norm_val < 1e-6:
            return True
        emb = emb / norm_val

        # Pre-check if already registered on first shot
        if len(self.embeddings) == 0 and reg is not None:
            matched_name, sim = reg.check_already_registered(name=self.name, embedding=emb)
            if matched_name and matched_name.lower() != self.name.lower() and sim >= 0.75:
                self.failed = True
                self.already_registered = True
                self.already_registered_as = matched_name
                self.error_msg = f"Object is already registered as '{matched_name}' (conf: {sim:.2f})"
                self.status_msg = f"Already registered as '{matched_name}'!"
                print(f"[auto-object-enroll] Aborted: {self.error_msg}")
                return False

        # Viewpoint diversity check vs previous shot
        if self.embeddings:
            sim = float(emb @ self.embeddings[-1])
            if sim > 0.995:
                self.status_msg = f"Rotate {self.name} slightly for multi-angle view..."
                return True

        self.embeddings.append(emb)
        self.crops.append(crop)
        self.last_capture_time = now
        self.status_msg = f"Enrolling {self.name}: {len(self.embeddings)}/{self.target_shots}"

        if len(self.embeddings) >= self.target_shots:
            try:
                mean_emb = np.mean(self.embeddings, axis=0).astype(np.float32)
                mean_norm = float(np.linalg.norm(mean_emb))
                if mean_norm > 1e-8:
                    mean_emb = mean_emb / mean_norm

                if reg is not None:
                    ok, msg, out_path = reg.register_object(
                        name=self.name,
                        synonyms=self.synonyms,
                        embedding=mean_emb,
                        overwrite=True,
                    )
                    self.saved_path = out_path

                # Save thumbnail crop
                if self.crops:
                    thumb_dir = reg.gallery_dir if reg else GALLERY_DIR / "objects"
                    thumb_dir.mkdir(parents=True, exist_ok=True)
                    thumb_path = thumb_dir / f"{self.name}.jpg"
                    cv2.imwrite(str(thumb_path), self.crops[-1])

                # Hot-reload detector vocabulary and visual gallery
                if det is not None:
                    det.register_class(self.name, self.synonyms)
                    det.reload_classes()

                self.completed = True
                self.status_msg = f"Successfully registered object '{self.name}' ({len(self.embeddings)} shots)!"
                print(f"[auto-object-enroll] {self.status_msg}")
                return False
            except Exception as exc:
                self.failed = True
                self.error_msg = str(exc)
                self.status_msg = f"Failed to save {self.name}: {exc}"
                return False

        return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive multi-shot face enrollment for BMO-CV."
    )
    parser.add_argument("name", type=str, help="Person name (saved as gallery/<name>.npy)")
    parser.add_argument(
        "--shots",
        type=int,
        default=10,
        help="Number of face embeddings to capture and average (default: 10)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    enroll(args.name, shots=args.shots)


if __name__ == "__main__":
    main()
