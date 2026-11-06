"""Interactive sign language sequence collector with BMO HUD feedback."""

from __future__ import annotations

import argparse
import pathlib
import time

import cv2
import numpy as np

try:
    from . import hud
    from .config import CAM_INDEX, DATA_DIR, DEVICE, SEQ_LEN
    from .hands import landmarks, norm
except ImportError:
    import hud
    from config import CAM_INDEX, DATA_DIR, DEVICE, SEQ_LEN
    from hands import landmarks, norm


def get_next_idx(out_dir: pathlib.Path) -> int:
    """Return safe monotonic index so deleting an intermediate .npy never overwrites."""
    return max([int(p.stem) for p in out_dir.glob("*.npy") if p.stem.isdigit()], default=-1) + 1


def collect(label: str, seq_len: int = SEQ_LEN, cam_index: int = CAM_INDEX) -> None:
    """Run live webcam loop to record (seq_len, 126) hand landmark sequences."""
    out = pathlib.Path(DATA_DIR) / label
    out.mkdir(parents=True, exist_ok=True)
    SEQ = int(seq_len)

    cap = cv2.VideoCapture(cam_index)
    rec: bool = False
    seq: list[np.ndarray] = []
    events: list[str] = [f"Collector ready for '{label}' (SPACE=record, ESC/q=quit)"]
    t_prev = time.perf_counter()
    fps = 0.0

    try:
        while cap.isOpened():
            ok, f = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = max(1e-6, now - t_prev)
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0.0 else (1.0 / dt)

            raw_lm = landmarks(f)
            x = norm(raw_lm)

            if rec:
                seq.append(x)
                if len(seq) == SEQ:
                    valid_frames = sum(1 for frame_vec in seq if np.any(frame_vec))
                    if label.lower() != "none" and valid_frames < 5:
                        msg = f"WARN: Skipped '{label}' ({valid_frames}/{SEQ} hand frames < 5)"
                        print(msg)
                        events.append(msg)
                    else:
                        next_idx = get_next_idx(out)
                        save_path = out / f"{next_idx:03d}.npy"
                        np.save(save_path, np.array(seq, dtype=np.float32))
                        msg = f"Saved {save_path.name} ({valid_frames}/{SEQ} hand frames)"
                        print(msg)
                        events.append(msg)
                    rec, seq = False, []

            count = len(list(out.glob("*.npy")))
            hud.draw_hand_skeleton(f, raw_lm)
            hud.draw_sign_gauge(
                f,
                len(seq) if rec else 0,
                seq_len=SEQ,
                active_sign=f"REC {label}" if rec else f"READY: {label} (saved: {count})",
            )
            hud.draw_event_feed(f, events)
            hud.draw_status_bar(
                f,
                fps=fps,
                device=DEVICE,
                toggles={"HANDS": bool(np.any(raw_lm)), "REC": rec},
            )

            cv2.imshow("Collect Sign", f)
            k = cv2.waitKey(1) & 0xFF
            if k == 32 and not rec:
                rec, seq = True, []
            elif k in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sign language landmark sequences.")
    parser.add_argument("label", help="Sign class label (e.g. hello, thanks, none)")
    parser.add_argument("--seq", type=int, default=SEQ_LEN, help="Sequence length in frames")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index")
    args = parser.parse_args()
    collect(args.label, seq_len=args.seq, cam_index=args.cam)


if __name__ == "__main__":
    main()
