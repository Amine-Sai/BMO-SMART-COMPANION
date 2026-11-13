"""Load the hand-drawn BMO face SVGs and render animated frames with OpenCV."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from bmo_contracts.models import Emotion

from .state import FaceMode

ART_W, ART_H = 1280, 720
BG = (222, 249, 192)  # BGR of the art's #C0F9DE background
_INK = (20, 20, 20)
_ACCENT = (59, 143, 29)  # BGR of the art's #1D8F3B green

DEFAULT_FILE = "default-face"
TALKING_FILE = "talking-face"
SLEEPING_FILE = "sleeping-face"

EMOTION_FILE: dict[Emotion, str] = {
    Emotion.NEUTRAL: DEFAULT_FILE,
    Emotion.HAPPY: "happy-face",
    Emotion.SAD: "sad-face",
    Emotion.THINKING: "side-look",
    Emotion.SURPRISED: "shocked",
    Emotion.SLEEPY: SLEEPING_FILE,
    Emotion.WINK: "playful-face",
    Emotion.ERROR: "dizzy",
    Emotion.ANGRY: "angry-face",
    Emotion.MAD: "mad-face",
    Emotion.ANNOYED: "annoyed-face",
    Emotion.PLAYFUL: "playful-face",
    Emotion.LOVE: "heart-eyes",
    Emotion.KISS: "kiss",
    Emotion.EXCITED: "starry-eyes",
    Emotion.AMAZED: "amazed-face",
    Emotion.TERRIFIED: "terrified-face",
    Emotion.EMPATHETIC: "empathetic-face",
}

MODE_FILE: dict[FaceMode, str] = {
    FaceMode.BOOT: SLEEPING_FILE,
    FaceMode.SLEEP: SLEEPING_FILE,
    FaceMode.MUTED: "annoyed-face",
    FaceMode.OFFLINE: DEFAULT_FILE,
    FaceMode.ERROR: "dizzy",
    FaceMode.THINKING: "side-look",
    FaceMode.LISTENING: DEFAULT_FILE,
}

_TINTS: dict[FaceMode, np.ndarray | None] = {
    FaceMode.SLEEP: None,
    FaceMode.MUTED: np.array([0.76, 0.76, 0.80], dtype=np.float32),
    FaceMode.OFFLINE: np.array([0.55, 0.58, 0.62], dtype=np.float32),
    FaceMode.ERROR: np.array([0.94, 0.90, 0.90], dtype=np.float32),
}

REQUIRED_FILES = sorted(
    set(EMOTION_FILE.values())
    | set(MODE_FILE.values())
    | {DEFAULT_FILE, TALKING_FILE}
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _fit_size(size: tuple[int, int]) -> tuple[int, int]:
    """Largest 16:9 render size that fits inside ``size``."""
    width, height = size
    scale = min(width / ART_W, height / ART_H)
    return max(1, round(ART_W * scale)), max(1, round(ART_H * scale))


def _ensure_png(
    svg: Path,
    png: Path,
    width: int,
    height: int,
    force: bool = False,
) -> None:
    if (
        not force
        and png.exists()
        and png.stat().st_mtime >= svg.stat().st_mtime
    ):
        return
    subprocess.run(
        [
            "rsvg-convert",
            "-w",
            str(width),
            "-h",
            str(height),
            str(svg),
            "-o",
            str(png),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


def _place(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Fit a 1280x720 art frame into ``size``, padding with the art background."""
    width, height = size
    scale = min(width / ART_W, height / ART_H)
    draw_w = max(1, round(ART_W * scale))
    draw_h = max(1, round(ART_H * scale))
    origin = ((width - draw_w) // 2, (height - draw_h) // 2)
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
    canvas[origin[1] : origin[1] + draw_h, origin[0] : origin[0] + draw_w] = cv.resize(
        img, (draw_w, draw_h), interpolation=interp
    )
    return canvas, scale, origin


@dataclass
class FaceArt:
    size: tuple[int, int]
    frames: dict[str, np.ndarray]
    art_rect: tuple[int, int, int, int]

    def render(
        self,
        mode: FaceMode,
        emotion: Emotion,
        t: float,
        blinking: bool = False,
        speaking_open: bool = False,
    ) -> np.ndarray:
        if mode == FaceMode.SPEAKING:
            # Talk by flipping the closed default face with the open talking face.
            name = TALKING_FILE if speaking_open else DEFAULT_FILE
        elif mode == FaceMode.AWAKE:
            name = EMOTION_FILE.get(emotion, DEFAULT_FILE)
        else:
            name = MODE_FILE.get(mode, DEFAULT_FILE)
        if blinking and name == DEFAULT_FILE:
            # The sleeping art is the same face with the eyes closed.
            name = SLEEPING_FILE
        frame = self.frames.get(name)
        if frame is None:
            frame = self.frames[DEFAULT_FILE]
        frame = frame.copy()

        tint = _TINTS.get(mode)
        if tint is not None:
            frame[:] = np.clip(
                frame.astype(np.float32) * tint, 0, 255
            ).astype(np.uint8)

        if mode == FaceMode.SPEAKING:
            bob = int(round(2.0 * math.sin(t * 9.0)))
            if bob:
                frame = np.roll(frame, bob, axis=0)
                if bob > 0:
                    frame[:bob] = BG
                else:
                    frame[bob:] = BG

        if mode == FaceMode.THINKING:
            self._draw_dots(frame, t)
        elif mode == FaceMode.SLEEP:
            self._draw_zzz(frame, t)
        elif mode == FaceMode.OFFLINE:
            self._draw_offline(frame)
        elif mode == FaceMode.LISTENING:
            self._draw_listening(frame, t)
        elif mode == FaceMode.ERROR:
            self._draw_error(frame)
        return frame

    # -- extras ------------------------------------------------------------
    def _draw_dots(self, frame: np.ndarray, t: float) -> None:
        x0, y0, x1, y1 = self.art_rect
        aw, ah = x1 - x0, y1 - y0
        bx = x0 + int(0.70 * aw)
        by = y0 + int(0.13 * ah)
        step = int(0.035 * aw)
        radius = int(0.008 * aw)
        for i in range(3):
            r = radius + int(0.006 * aw * (0.5 + 0.5 * math.sin(t * 7.0 - i * 0.9)))
            cv.circle(frame, (bx + i * step, by), max(3, r), _ACCENT, -1, cv.LINE_AA)

    def _draw_zzz(self, frame: np.ndarray, t: float) -> None:
        x0, y0, x1, y1 = self.art_rect
        aw, ah = x1 - x0, y1 - y0
        bx = x0 + int(0.86 * aw)
        by = y0 + int(0.22 * ah)
        for i in range(3):
            phase = (t * 0.28 + i * 0.34) % 1.0
            x = bx + int(phase * 0.025 * aw)
            y = by - int(phase * 0.11 * ah)
            scale = (1.0 + phase) * aw / 1280.0
            cv.putText(
                frame,
                "z",
                (x, y),
                cv.FONT_HERSHEY_DUPLEX,
                scale,
                _INK,
                max(1, int(2 * aw / 1280)),
                cv.LINE_AA,
            )

    def _draw_offline(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = self.art_rect
        text = "offline"
        scale = 0.9 * (x1 - x0) / 1280.0
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, scale, 2)
        cv.putText(
            frame,
            text,
            (x0 + (x1 - x0 - tw) // 2, y0 + int((y1 - y0) * 0.78)),
            cv.FONT_HERSHEY_DUPLEX,
            scale,
            (70, 70, 74),
            2,
            cv.LINE_AA,
        )

    def _draw_listening(self, frame: np.ndarray, t: float) -> None:
        x0, y0, x1, y1 = self.art_rect
        scale = (x1 - x0) / 1280.0
        inset = max(4, int(8 * scale))
        thickness = max(2, int((3 + 2 * (0.5 + 0.5 * math.sin(t * 3.0))) * scale))
        cv.rectangle(
            frame,
            (x0 + inset, y0 + inset),
            (x1 - inset, y1 - inset),
            _ACCENT,
            thickness,
            cv.LINE_AA,
        )

    def _draw_error(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = self.art_rect
        inset = max(4, int(8 * (x1 - x0) / 1280.0))
        cv.rectangle(
            frame,
            (x0 + inset, y0 + inset),
            (x1 - inset, y1 - inset),
            (50, 50, 220),
            max(2, inset // 2),
            cv.LINE_AA,
        )


def load_art(
    size: tuple[int, int],
    svg_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> FaceArt | None:
    """Render the provided SVGs into cached PNGs and build a reusable face bank.

    Returns ``None`` when the artwork, ``rsvg-convert`` or any asset is missing,
    so callers can fall back to the procedural face.
    """
    repo = _repo_root()
    svg_dir = svg_dir or Path(os.environ.get("BMO_FACE_ART_DIR", repo / "BMO Face"))
    cache_dir = cache_dir or Path(
        os.environ.get("BMO_FACE_ART_CACHE", repo / "data" / "face_assets")
    )
    if not svg_dir.is_dir():
        print(f"[FACE] art folder not found: {svg_dir}", flush=True)
        return None
    if shutil.which("rsvg-convert") is None:
        print("[FACE] rsvg-convert not found; using procedural face", flush=True)
        return None

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, np.ndarray] = {}
        base_scale = 1.0
        base_origin = (0, 0)
        render_w, render_h = _fit_size(size)
        for name in REQUIRED_FILES:
            svg = svg_dir / f"{name}.svg"
            if not svg.exists():
                raise FileNotFoundError(svg)
            png = cache_dir / f"{name}-{render_w}x{render_h}.png"
            _ensure_png(svg, png, render_w, render_h)
            image = cv.imread(str(png), cv.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read {png}")
            canvas, scale, origin = _place(image, size)
            frames[name] = canvas
            if name == DEFAULT_FILE:
                base_scale, base_origin = scale, origin

        width, height = size
        draw_w = round(ART_W * base_scale)
        draw_h = round(ART_H * base_scale)
        art_rect = (
            base_origin[0],
            base_origin[1],
            base_origin[0] + draw_w,
            base_origin[1] + draw_h,
        )

        art = FaceArt(size=(width, height), frames=frames, art_rect=art_rect)
        print(
            f"[FACE] art loaded: {len(frames)} frames at {width}x{height}",
            flush=True,
        )
        return art
    except Exception as exc:  # noqa: BLE001
        print(f"[FACE] art unavailable ({exc}); using procedural face", flush=True)
        return None


if __name__ == "__main__":  # pragma: no cover - manual preview helper
    out = _repo_root() / "logs" / "face_art_preview.png"
    art = load_art((1280, 720))
    if art is not None:
        cv.imwrite(str(out), art.render(FaceMode.AWAKE, Emotion.NEUTRAL, 0.0))
        print(out)
