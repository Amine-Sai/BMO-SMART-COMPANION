"""Real-time sign language recognition with dropout tolerance and probability smoothing."""

from __future__ import annotations

from collections import deque
import pathlib
import time
from typing import Any

import cv2
import numpy as np
import torch

try:
    from . import hud
    from .config import CAM_INDEX, DEVICE, SEQ_LEN, SIGN_CONF, SIGN_MODEL_PATH
    from .hands import landmarks, norm
    from .model import SignNet
except ImportError:
    import hud
    from config import CAM_INDEX, DEVICE, SEQ_LEN, SIGN_CONF, SIGN_MODEL_PATH
    from hands import landmarks, norm
    from model import SignNet


class SignRecognizer:
    """Stateful sliding-window sign recognizer with 5-frame dropout tolerance."""

    def __init__(
        self,
        model_path: str | pathlib.Path = SIGN_MODEL_PATH,
        seq_len: int = SEQ_LEN,
        conf_thresh: float = SIGN_CONF,
        hold_frames: int = 12,
    ) -> None:
        self.model_path = pathlib.Path(model_path)
        self.seq_len = int(seq_len)
        self.conf_thresh = float(conf_thresh)
        self.hold_frames = int(hold_frames)

        self.buf: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.prob_hist: deque[np.ndarray] = deque(maxlen=3)
        self.missing_frames: int = 0

        self.active_sign: str | None = None
        self.active_conf: float = 0.0
        self.hold_counter: int = 0
        self.last_raw_lm: np.ndarray = np.zeros((2, 21, 3), dtype=np.float32)

        self.model: SignNet | None = None
        self.labels: list[str] = []
        self._loaded_mtime: float | None = None
        self._try_load_model()

    def _resolve_ckpt_path(self) -> pathlib.Path | None:
        if self.model_path.exists():
            return self.model_path
        fallback = pathlib.Path("sign.pt")
        if fallback.exists():
            return fallback
        return None

    def _try_load_model(self) -> bool:
        ckpt_path = self._resolve_ckpt_path()
        if ckpt_path is None:
            return False
        try:
            mtime = ckpt_path.stat().st_mtime
            if self.model is not None and self._loaded_mtime == mtime:
                return True
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            labels = list(ckpt["labels"])
            net = SignNet(len(labels)).to(DEVICE)
            net.load_state_dict(ckpt["m"])
            net.eval()
            self.model = net
            self.labels = labels
            self._loaded_mtime = mtime
            return True
        except Exception as exc:
            print(f"[WARN] Failed to load sign model from '{ckpt_path}': {exc}")
            self.model = None
            self.labels = []
            return False

    def reset(self) -> None:
        """Clear sequence buffer, smoothing history, and active prediction."""
        self.buf.clear()
        self.prob_hist.clear()
        self.missing_frames = 0
        self.active_sign = None
        self.active_conf = 0.0
        self.hold_counter = 0
        self.last_raw_lm = np.zeros((2, 21, 3), dtype=np.float32)

    def _extract_input(self, frame_or_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(frame_or_features)
        if arr.ndim == 1 and arr.size == 126:
            x = arr.astype(np.float32)
            raw_lm = x.reshape(2, 21, 3).copy()
            return raw_lm, x
        if arr.shape == (2, 21, 3):
            raw_lm = arr.astype(np.float32)
            x = norm(raw_lm)
            return raw_lm, x
        raw_lm = landmarks(arr)
        x = norm(raw_lm)
        return raw_lm, x

    def step(
        self, frame_bgr: np.ndarray
    ) -> tuple[str | None, float, int, np.ndarray]:
        """Process one BGR frame (or pre-extracted landmark vector) and update state.

        Returns:
            (active_sign, active_conf, buf_len, raw_lm)
        """
        if self.model is None:
            self._try_load_model()

        raw_lm, x = self._extract_input(frame_bgr)
        self.last_raw_lm = raw_lm

        if bool(np.any(x)):
            self.missing_frames = 0
            self.buf.append(x)
        else:
            self.missing_frames += 1
            if 0 < len(self.buf) and self.missing_frames <= 5:
                self.buf.append(self.buf[-1].copy())
            elif self.missing_frames > 5:
                self.buf.clear()
                self.prob_hist.clear()

        pred_sign: str | None = None
        pred_conf: float = 0.0

        if len(self.buf) == self.seq_len and self.model is not None and self.labels:
            seq_arr = np.array(self.buf, dtype=np.float32)
            inp = torch.tensor(seq_arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self.model(inp), dim=-1)[0].detach().cpu().numpy()
            self.prob_hist.append(probs)
            p_smooth = np.mean(self.prob_hist, axis=0)
            best_idx = int(np.argmax(p_smooth))
            best_conf = float(p_smooth[best_idx])
            best_label = self.labels[best_idx]
            if best_conf > self.conf_thresh and best_label.lower() != "none":
                pred_sign = best_label
                pred_conf = best_conf

        if pred_sign is not None:
            self.active_sign = pred_sign
            self.active_conf = pred_conf
            self.hold_counter = self.hold_frames
        else:
            if self.hold_counter > 0 and self.missing_frames <= 5:
                self.hold_counter -= 1
                if self.hold_counter == 0:
                    self.active_sign = None
                    self.active_conf = 0.0
            else:
                self.hold_counter = 0
                self.active_sign = None
                self.active_conf = 0.0

        return self.active_sign, self.active_conf, len(self.buf), raw_lm

    def get_state(self) -> dict[str, Any]:
        """Return dictionary of latest sign HUD fields for hud.render_hud()."""
        return {
            "sign": self.active_sign,
            "sign_conf": self.active_conf,
            "sign_buf": len(self.buf),
            "seq_len": self.seq_len,
            "raw_hands": self.last_raw_lm,
        }


_recognizer: SignRecognizer | None = None


def get_recognizer() -> SignRecognizer:
    """Return the singleton SignRecognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = SignRecognizer()
    return _recognizer


def sign_step(frame_bgr: np.ndarray) -> str | None:
    """Run one step of sign recognition on frame_bgr and return active_sign."""
    active_sign, _, _, _ = get_recognizer().step(frame_bgr)
    return active_sign


def main() -> None:
    rec = get_recognizer()
    cap = cv2.VideoCapture(CAM_INDEX)
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

            active_sign, active_conf, buf_len, raw_lm = rec.step(f)

            hud.draw_hand_skeleton(f, raw_lm)
            hud.draw_sign_gauge(
                f,
                buf_len=buf_len,
                seq_len=rec.seq_len,
                active_sign=active_sign,
                sign_conf=active_conf,
            )
            hud.draw_status_bar(
                f,
                fps=fps,
                device=DEVICE,
                toggles={
                    "SIGN": rec.model is not None,
                    "HANDS": bool(np.any(raw_lm)),
                },
            )

            cv2.imshow("Sign", f)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
