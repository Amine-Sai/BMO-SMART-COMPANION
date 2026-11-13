from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2 as cv
import numpy as np

from bmo_contracts.models import Emotion

from .state import FaceMode

INK = (24, 22, 22)
SCREEN_TOP = np.array([214, 250, 190], dtype=np.float32)
SCREEN_BOTTOM = np.array([150, 225, 140], dtype=np.float32)
BEZEL = (32, 32, 36)
BEZEL_EDGE = (18, 18, 20)
SCREW = (78, 78, 84)
SCREW_DARK = (40, 40, 45)
LABEL = (96, 96, 108)


@dataclass
class FaceParams:
    eye_open: float = 1.0
    eye_scale: float = 1.0
    eye_style: str = "normal"  # normal | happy | closed | error
    wink: int = 0  # 0 none, 1 left, 2 right
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    brow: float = 0.0
    mouth_w: float = 0.42
    mouth_open: float = 0.0
    mouth_curve: float = 0.25
    blush: float = 0.0
    head_bob: float = 0.0


EMOTION_PRESETS: dict[Emotion, FaceParams] = {
    Emotion.NEUTRAL: FaceParams(),
    Emotion.HAPPY: FaceParams(
        eye_style="happy", mouth_w=0.48, mouth_curve=0.95, blush=0.6
    ),
    Emotion.SAD: FaceParams(
        eye_open=0.72,
        gaze_y=0.45,
        brow=-0.65,
        mouth_w=0.36,
        mouth_curve=-0.85,
    ),
    Emotion.THINKING: FaceParams(
        eye_open=0.82, gaze_x=0.35, gaze_y=-0.55, mouth_w=0.3, mouth_curve=0.05
    ),
    Emotion.SURPRISED: FaceParams(
        eye_open=1.12,
        eye_scale=1.16,
        brow=0.6,
        mouth_open=0.55,
        mouth_w=0.28,
        mouth_curve=0.0,
    ),
    Emotion.SLEEPY: FaceParams(
        eye_open=0.28, gaze_y=0.15, mouth_w=0.3, mouth_curve=0.1
    ),
    Emotion.WINK: FaceParams(wink=1, mouth_curve=0.6, blush=0.35),
    Emotion.ERROR: FaceParams(eye_style="error", mouth_curve=-0.4, mouth_w=0.34),
    Emotion.ANGRY: FaceParams(
        eye_open=0.8,
        brow=0.85,
        mouth_open=0.35,
        mouth_w=0.34,
        mouth_curve=-0.2,
    ),
    Emotion.MAD: FaceParams(
        eye_open=0.72,
        brow=0.8,
        mouth_w=0.32,
        mouth_curve=-0.9,
    ),
    Emotion.ANNOYED: FaceParams(
        eye_open=0.55, gaze_x=0.25, brow=0.35, mouth_w=0.3, mouth_curve=0.05
    ),
    Emotion.PLAYFUL: FaceParams(
        eye_style="happy", wink=1, mouth_curve=0.8, blush=0.45
    ),
    Emotion.LOVE: FaceParams(
        eye_style="happy", eye_scale=1.06, mouth_curve=0.75, blush=0.95
    ),
    Emotion.KISS: FaceParams(
        eye_open=0.45, mouth_w=0.22, mouth_curve=0.95, blush=0.5
    ),
    Emotion.EXCITED: FaceParams(
        eye_open=1.12,
        eye_scale=1.1,
        brow=0.4,
        mouth_open=0.5,
        mouth_w=0.3,
        blush=0.5,
    ),
    Emotion.AMAZED: FaceParams(
        eye_open=1.2,
        eye_scale=1.2,
        brow=0.5,
        mouth_open=0.6,
        mouth_w=0.3,
        mouth_curve=0.0,
    ),
    Emotion.TERRIFIED: FaceParams(
        eye_open=1.25,
        eye_scale=1.2,
        brow=-0.5,
        gaze_y=-0.3,
        mouth_open=0.6,
        mouth_w=0.26,
    ),
    Emotion.EMPATHETIC: FaceParams(
        eye_open=0.9, brow=-0.45, mouth_w=0.34, mouth_curve=0.25
    ),
}

_SLEEP = FaceParams(
    eye_open=0.05, eye_style="closed", mouth_w=0.28, mouth_curve=0.15
)
_MUTED = FaceParams(
    eye_open=0.05, eye_style="closed", mouth_w=0.28, mouth_curve=0.1
)
_OFFLINE = FaceParams(
    eye_open=0.05, eye_style="closed", mouth_w=0.26, mouth_curve=0.0
)
_LISTENING = FaceParams(
    eye_open=1.08, eye_scale=1.05, mouth_w=0.3, mouth_curve=0.2
)


def target_params(mode: FaceMode, emotion: Emotion, now: float, last_say: float, t: float) -> FaceParams:
    """Build the target face shape for the current mode and emotion."""
    if mode == FaceMode.SLEEP:
        return replace(_SLEEP, head_bob=math.sin(t * 1.2) * 3.0)
    if mode == FaceMode.MUTED:
        return replace(_MUTED, head_bob=math.sin(t * 1.0) * 1.5)
    if mode == FaceMode.OFFLINE:
        return _OFFLINE
    if mode == FaceMode.ERROR:
        return EMOTION_PRESETS[Emotion.ERROR]
    if mode == FaceMode.THINKING:
        return replace(
            EMOTION_PRESETS[Emotion.THINKING], head_bob=math.sin(t * 2.0) * 1.2
        )
    if mode == FaceMode.LISTENING:
        return replace(
            _LISTENING,
            gaze_x=math.sin(t * 0.8) * 0.15,
            gaze_y=math.sin(t * 0.6) * 0.08,
        )
    if mode == FaceMode.SPEAKING:
        params = replace(EMOTION_PRESETS.get(emotion, FaceParams()))
        params.mouth_open = max(
            0.0,
            min(1.0, 0.22 + 0.4 * abs(math.sin(t * 14.0)) + 0.1 * math.sin(t * 9.3)),
        )
        params.head_bob = math.sin(t * 4.0) * 0.8
        return params

    # AWAKE (and BOOT): hold the last emotion for a moment, then look around.
    params = FaceParams()
    if now - last_say < 1.6:
        params = replace(EMOTION_PRESETS.get(emotion, FaceParams()))
    params.gaze_x = 0.28 * math.sin(t * 0.31) + 0.14 * math.sin(t * 0.13)
    params.gaze_y = 0.12 * math.sin(t * 0.23)
    return params


def lerp_params(current: FaceParams, target: FaceParams, alpha: float) -> FaceParams:
    """Interpolate continuous parameters; switch discrete style fields directly."""
    a = max(0.0, min(1.0, alpha))

    def mix(x: float, y: float) -> float:
        return x + (y - x) * a

    return FaceParams(
        eye_open=mix(current.eye_open, target.eye_open),
        eye_scale=mix(current.eye_scale, target.eye_scale),
        eye_style=target.eye_style,
        wink=target.wink,
        gaze_x=mix(current.gaze_x, target.gaze_x),
        gaze_y=mix(current.gaze_y, target.gaze_y),
        brow=mix(current.brow, target.brow),
        mouth_w=mix(current.mouth_w, target.mouth_w),
        mouth_open=mix(current.mouth_open, target.mouth_open),
        mouth_curve=mix(current.mouth_curve, target.mouth_curve),
        blush=mix(current.blush, target.blush),
        head_bob=mix(current.head_bob, target.head_bob),
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    r = max(0, min(radius, w // 2, h // 2))
    if r == 0:
        mask[:] = 255
        return mask
    cv.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _vertical_gradient(
    w: int, h: int, top: np.ndarray, bottom: np.ndarray
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    grad = top[None, :] * (1.0 - t) + bottom[None, :] * t
    grad = np.repeat(grad[:, None, :], w, axis=1)
    return np.clip(grad, 0, 255).astype(np.uint8)


def _tint(mode: FaceMode) -> np.ndarray:
    factors = {
        FaceMode.AWAKE: (1.0, 1.0, 1.0),
        FaceMode.BOOT: (1.0, 1.0, 1.0),
        FaceMode.LISTENING: (1.04, 1.04, 1.0),
        FaceMode.THINKING: (1.0, 1.03, 1.07),
        FaceMode.SPEAKING: (1.0, 1.0, 1.0),
        FaceMode.SLEEP: (0.72, 0.8, 0.9),
        FaceMode.MUTED: (0.66, 0.66, 0.72),
        FaceMode.ERROR: (0.9, 0.8, 0.8),
        FaceMode.OFFLINE: (0.62, 0.64, 0.68),
    }
    return np.array(factors.get(mode, (1.0, 1.0, 1.0)), dtype=np.float32)


def _draw_rounded_outline(
    frame: np.ndarray, rect: tuple[int, int, int, int], radius: int, color, thickness: int
) -> None:
    x1, y1, x2, y2 = rect
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv.LINE_AA)
    cv.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv.LINE_AA)
    cv.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv.LINE_AA)
    cv.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv.LINE_AA)
    cv.ellipse(frame, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thickness, cv.LINE_AA)
    cv.ellipse(frame, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thickness, cv.LINE_AA)
    cv.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv.LINE_AA)
    cv.ellipse(frame, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thickness, cv.LINE_AA)


def _draw_eye(
    frame: np.ndarray,
    center: tuple[int, int],
    w: int,
    h: int,
    open_amt: float,
    style: str,
) -> None:
    cx, cy = int(center[0]), int(center[1])
    w = max(8, int(w))
    h = max(8, int(h))

    if style == "happy":
        axes = (max(6, w // 2), max(8, int(h * 0.38)))
        cv.ellipse(frame, (cx, cy), axes, 0, 180, 360, INK, 10, cv.LINE_AA)
        return

    if style == "error":
        d = max(8, int(w * 0.4))
        cv.line(frame, (cx - d, cy - d), (cx + d, cy + d), INK, 10, cv.LINE_AA)
        cv.line(frame, (cx - d, cy + d), (cx + d, cy - d), INK, 10, cv.LINE_AA)
        return

    if style == "closed" or open_amt < 0.08:
        axes = (max(6, w // 2), max(6, int(h * 0.11)))
        cv.ellipse(frame, (cx, cy), axes, 0, 0, 180, INK, 8, cv.LINE_AA)
        return

    eh = max(8, int(h * max(0.0, min(1.2, open_amt))))
    x1 = cx - w // 2
    y1 = cy - eh // 2
    x2 = x1 + w
    y2 = y1 + eh
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    mask = _rounded_mask(x2 - x1, y2 - y1, min(w, eh) // 2)
    roi = frame[fy1:fy2, fx1:fx2]
    sub = mask[fy1 - y1 : fy2 - y1, fx1 - x1 : fx2 - x1]
    roi[sub > 0] = INK

    glint_r = max(3, int(w * 0.11))
    cv.circle(
        frame,
        (x1 + int(w * 0.28), y1 + int(eh * 0.26)),
        glint_r,
        (245, 255, 255),
        -1,
        cv.LINE_AA,
    )


def _draw_mouth(frame: np.ndarray, center: tuple[int, int], params: FaceParams, w: int) -> None:
    cx, cy = int(center[0]), int(center[1])
    mw = max(20, int(w * 0.18 * (params.mouth_w / 0.42)))
    openness = max(0.0, min(1.0, params.mouth_open))

    if openness > 0.12:
        axes = (mw // 2, int(10 + openness * 46))
        cv.ellipse(frame, (cx, cy), axes, 0, 0, 360, INK, -1, cv.LINE_AA)
        tongue_y = cy + int(axes[1] * 0.45)
        cv.ellipse(
            frame,
            (cx, tongue_y),
            (max(6, mw // 3), max(5, int(6 + openness * 10))),
            0,
            0,
            180,
            (140, 150, 245),
            -1,
            cv.LINE_AA,
        )
        return

    curve = params.mouth_curve
    if curve > 0.12:
        axes = (mw // 2, 18 + int(abs(curve) * 18))
        cv.ellipse(frame, (cx, cy - 6), axes, 0, 0, 180, INK, 7, cv.LINE_AA)
    elif curve < -0.12:
        axes = (mw // 2, 18 + int(abs(curve) * 18))
        cv.ellipse(frame, (cx, cy + 6), axes, 0, 180, 360, INK, 7, cv.LINE_AA)
    else:
        cv.line(frame, (cx - mw // 2, cy), (cx + mw // 2, cy), INK, 6, cv.LINE_AA)


def _draw_blush(frame: np.ndarray, center: tuple[int, int], amount: float, w: int) -> None:
    if amount <= 0.05:
        return
    cx, cy = center
    overlay = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for dx in (-int(w * 0.19), int(w * 0.19)):
        cv.ellipse(overlay, (cx + dx, cy), (48, 28), 0, 0, 360, (120, 130, 255), -1, cv.LINE_AA)
        cv.ellipse(mask, (cx + dx, cy), (48, 28), 0, 0, 360, 255, -1)
    mask = cv.GaussianBlur(mask, (0, 0), 18)
    alpha = (mask.astype(np.float32) / 255.0) * min(1.0, amount) * 0.65
    frame[:] = (
        frame.astype(np.float32) * (1.0 - alpha[..., None])
        + overlay.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)


def _draw_dots(frame: np.ndarray, t: float, w: int) -> None:
    base_x = int(w * 0.68)
    base_y = int(frame.shape[0] * 0.2)
    for i in range(3):
        r = int(7 + 4 * (0.5 + 0.5 * math.sin(t * 7.0 - i * 0.9)))
        cv.circle(frame, (base_x + i * 34, base_y), r, (58, 118, 88), -1, cv.LINE_AA)


def _draw_zzz(frame: np.ndarray, t: float, w: int) -> None:
    base_x = int(w * 0.70)
    base_y = int(frame.shape[0] * 0.28)
    for i in range(3):
        phase = (t * 0.28 + i * 0.34) % 1.0
        x = base_x + int(phase * 34) + i * 4
        y = base_y - int(phase * 90)
        scale = 0.7 + phase * 1.0 + i * 0.1
        cv.putText(
            frame,
            "z",
            (x, y),
            cv.FONT_HERSHEY_DUPLEX,
            scale,
            (58, 118, 88),
            2,
            cv.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------
def render_face(
    params: FaceParams,
    mode: FaceMode,
    t: float,
    size: tuple[int, int] = (800, 480),
) -> np.ndarray:
    W, H = int(size[0]), int(size[1])
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Bezel
    bezel_mask = _rounded_mask(W, H, 46)
    frame[bezel_mask > 0] = BEZEL
    _draw_rounded_outline(frame, (2, 2, W - 3, H - 3), 46, BEZEL_EDGE, 3)

    # Screen
    sx, sy = 58, 54
    ex, ey = W - 58, H - 78
    sw, sh = ex - sx, ey - sy
    screen_mask = _rounded_mask(sw, sh, 26)
    tint = _tint(mode)
    gradient = _vertical_gradient(sw, sh, SCREEN_TOP * tint, SCREEN_BOTTOM * tint)
    roi = frame[sy:ey, sx:ex]
    roi[screen_mask > 0] = gradient[screen_mask > 0]

    # Subtle inner shadow at the top of the screen
    shade = np.tile(
        np.linspace(0.82, 1.0, min(22, sh), dtype=np.float32)[:, None, None],
        (1, sw, 3),
    )
    shadow_rows = min(22, sh)
    screen_roi = frame[sy : sy + shadow_rows, sx:ex]
    screen_roi[:] = np.clip(
        screen_roi.astype(np.float32) * shade, 0, 255
    ).astype(np.uint8)

    # Mode accent outlines
    if mode == FaceMode.LISTENING:
        pulse = int(160 + 60 * (0.5 + 0.5 * math.sin(t * 3.0)))
        _draw_rounded_outline(frame, (sx, sy, ex, ey), 26, (pulse, 255, pulse), 3)
    elif mode == FaceMode.ERROR:
        _draw_rounded_outline(frame, (sx, sy, ex, ey), 26, (60, 60, 230), 4)

    # Face
    oy = int(params.head_bob)
    eye_y = int(H * 0.40) + oy
    gx = int(params.gaze_x * 12)
    gy = int(params.gaze_y * 12)
    ew = int(78 * params.eye_scale)
    eh = int(104 * params.eye_scale)
    left = (int(W * 0.37) + gx, eye_y + gy)
    right = (int(W * 0.63) + gx, eye_y + gy)

    left_style = params.eye_style
    right_style = params.eye_style
    if params.wink == 1:
        left_style = "closed"
    elif params.wink == 2:
        right_style = "closed"

    _draw_eye(frame, left, ew, eh, params.eye_open, left_style)
    _draw_eye(frame, right, ew, eh, params.eye_open, right_style)

    if abs(params.brow) > 0.15:
        brow_y = eye_y - eh // 2 - 22 - int(max(0.0, params.brow) * 12)
        tilt = int(params.brow * 10)  # negative = sad / worried
        for sign in (-1, 1):
            cx = int(W * (0.37 if sign < 0 else 0.63)) + gx
            x1, x2 = cx - ew // 2, cx + ew // 2
            if params.brow < 0:
                y_outer = brow_y - tilt
                y_inner = brow_y + tilt
                if sign < 0:
                    p1, p2 = (x1, y_outer), (x2, y_inner)
                else:
                    p1, p2 = (x1, y_inner), (x2, y_outer)
            else:
                p1, p2 = (x1, brow_y), (x2, brow_y)
            cv.line(frame, p1, p2, INK, 6, cv.LINE_AA)

    mouth_y = int(H * 0.68) + oy
    _draw_mouth(frame, (W // 2, mouth_y), params, W)
    _draw_blush(frame, (W // 2, mouth_y - 14), params.blush, W)

    if mode == FaceMode.THINKING:
        _draw_dots(frame, t, W)
    if mode == FaceMode.SLEEP:
        _draw_zzz(frame, t, W)
    if mode == FaceMode.OFFLINE:
        cv.putText(
            frame,
            "offline",
            (W // 2 - 38, ey - 22),
            cv.FONT_HERSHEY_DUPLEX,
            0.7,
            (70, 74, 78),
            1,
            cv.LINE_AA,
        )

    # Bezel details
    for pos in ((34, 34), (W - 34, 34), (34, H - 34), (W - 34, H - 34)):
        cv.circle(frame, pos, 6, SCREW, -1, cv.LINE_AA)
        cv.circle(frame, pos, 3, SCREW_DARK, -1, cv.LINE_AA)
    cv.putText(
        frame,
        "BMO",
        (W // 2 - 24, H - 24),
        cv.FONT_HERSHEY_DUPLEX,
        0.7,
        LABEL,
        1,
        cv.LINE_AA,
    )

    return frame
