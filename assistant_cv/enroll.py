"""Interactive multi-shot face enrollment utility for BMO-CV."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from assistant_cv import hud
from assistant_cv.config import CAM_INDEX, DEVICE, GALLERY_DIR
from assistant_cv.faces import get_face_app

_FONT = cv2.FONT_HERSHEY_SIMPLEX


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
