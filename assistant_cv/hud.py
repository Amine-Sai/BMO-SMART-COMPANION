"""BMO HUD Renderer — sleek sci-fi/retro-futuristic OpenCV overlay for BMO-CV."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Color Palette (BGR)
# ---------------------------------------------------------------------------
BMO_TEAL = (196, 224, 112)
BMO_DARK = (38, 46, 28)
BMO_CYAN = (240, 210, 80)
BMO_GREEN = (90, 235, 100)
BMO_RED = (70, 80, 245)
BMO_AMBER = (60, 190, 255)
BMO_MAGENTA = (210, 100, 240)
BMO_WHITE = (245, 250, 245)
BMO_GRAY = (120, 130, 120)

# Standard 21-point MediaPipe hand bone connections
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger & palm base
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _blend_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int] = BMO_DARK,
    alpha: float = 0.65,
) -> None:
    """Alpha-blend a solid rectangle onto ``frame`` in-place (clamped to bounds)."""
    h, w = frame.shape[:2]
    rx1 = max(0, min(w, int(x1)))
    ry1 = max(0, min(h, int(y1)))
    rx2 = max(0, min(w, int(x2)))
    ry2 = max(0, min(h, int(y2)))
    if rx2 <= rx1 or ry2 <= ry1 or alpha <= 0.0:
        return
    if alpha >= 1.0:
        frame[ry1:ry2, rx1:rx2] = color
        return
    roi = frame[ry1:ry2, rx1:rx2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def draw_corner_box(
    frame: np.ndarray,
    bbox: Sequence[float | int],
    color: tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 14,
    label: str | None = None,
    bg_alpha: float = 0.65,
) -> np.ndarray:
    """Draw a bounding box with thick corner brackets, thin outline, and label pill."""
    if frame is None or bbox is None or len(bbox) < 4:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return frame

    # Thin outer bounding outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Thick corner brackets
    cl_x = min(int(corner_len), max(2, box_w // 2))
    cl_y = min(int(corner_len), max(2, box_h // 2))
    t = max(1, int(thickness))

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + cl_x, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + cl_y), color, t, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - cl_x, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + cl_y), color, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + cl_x, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - cl_y), color, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cl_x, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - cl_y), color, t, cv2.LINE_AA)

    # Sleek label pill above y1 (clamped inside frame bounds)
    if label:
        text = str(label)
        font_scale = 0.48
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, font_thick)
        pad_x, pad_y = 6, 4
        pill_w = tw + pad_x * 2
        pill_h = th + baseline + pad_y * 2

        px1 = max(0, min(w - pill_w, x1))
        py2 = y1 - 2
        py1 = py2 - pill_h
        if py1 < 0:
            py1 = max(0, y1 + 2)
            py2 = min(h, py1 + pill_h)
        px2 = min(w, px1 + pill_w)

        _blend_rect(frame, px1, py1, px2, py2, BMO_DARK, alpha=bg_alpha)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)
        text_y = min(h - 4, py1 + pad_y + th)
        cv2.putText(
            frame,
            text,
            (px1 + pad_x, text_y),
            _FONT,
            font_scale,
            BMO_WHITE,
            font_thick,
            cv2.LINE_AA,
        )

    return frame


def draw_objects(frame: np.ndarray, detections: Iterable[Any] | None) -> np.ndarray:
    """Draw object detection boxes with BMO_CYAN for person and BMO_TEAL for others."""
    if frame is None or not detections:
        return frame

    for det in detections:
        label: str | None = None
        conf: float | None = None
        bbox: Sequence[float | int] | None = None
        obj_id: int | str | None = None

        if isinstance(det, Mapping):
            label = det.get("label") or det.get("name") or det.get("class")
            conf = det.get("conf", det.get("score"))
            bbox = det.get("bbox")
            obj_id = det.get("id", det.get("track_id"))
        elif isinstance(det, (tuple, list)):
            if len(det) >= 4:
                label, conf, bbox, obj_id = det[0], det[1], det[2], det[3]
            elif len(det) == 3:
                label, conf, bbox = det[0], det[1], det[2]
            elif len(det) == 2:
                label, conf = det[0], det[1]

        if not label or bbox is None or len(bbox) < 4:
            continue

        label_str = str(label)
        color = BMO_CYAN if label_str.lower() == "person" else BMO_TEAL

        conf_val = float(conf) if conf is not None else 0.0
        if obj_id is not None:
            tag = f"{label_str} #{obj_id} {conf_val:.2f}"
        else:
            tag = f"{label_str} {conf_val:.2f}"

        draw_corner_box(frame, bbox, color, thickness=2, corner_len=14, label=tag)

    return frame


def draw_faces(frame: np.ndarray, faces: Iterable[Any] | None) -> np.ndarray:
    """Draw recognized/unknown face bounding boxes."""
    if frame is None or not faces:
        return frame

    for item in faces:
        name: str | None = None
        score: float = 0.0
        bbox: Sequence[float | int] | None = None

        if isinstance(item, Mapping):
            name = item.get("name") or item.get("label")
            score = float(item.get("score", item.get("conf", 0.0)) or 0.0)
            bbox = item.get("bbox")
        elif isinstance(item, (tuple, list)):
            if len(item) >= 3:
                name = str(item[0]) if item[0] is not None else "unknown"
                score = float(item[1]) if item[1] is not None else 0.0
                bbox = item[2]
            elif len(item) == 2:
                name = str(item[0]) if item[0] is not None else "unknown"
                score = float(item[1]) if item[1] is not None else 0.0

        if not name or bbox is None or len(bbox) < 4:
            continue

        is_known = name.strip().lower() not in ("unknown", "none", "")
        color = BMO_GREEN if is_known else BMO_RED
        tag = f"ID: {name} ({score:.2f})"
        draw_corner_box(frame, bbox, color, thickness=2, corner_len=14, label=tag)

    return frame


def draw_hand_skeleton(frame: np.ndarray, hands_data: Any) -> np.ndarray:
    """Draw 21-point MediaPipe hand skeletons for Left (BMO_CYAN) and Right (BMO_MAGENTA)."""
    if frame is None or hands_data is None:
        return frame

    h, w = frame.shape[:2]

    # Normalize hands_data into a sequence of (hand_array, side_label, color)
    hands_list: list[tuple[np.ndarray, str, tuple[int, int, int]]] = []

    if isinstance(hands_data, np.ndarray):
        arr = hands_data
        if arr.ndim == 1 and arr.size == 126:
            arr = arr.reshape(2, 21, 3)
        elif arr.ndim == 2 and arr.shape[0] == 21:
            arr = arr[np.newaxis, ...]

        if arr.ndim == 3 and arr.shape[1] == 21:
            for idx in range(min(2, arr.shape[0])):
                side = "L" if idx == 0 else "R"
                color = BMO_CYAN if idx == 0 else BMO_MAGENTA
                hands_list.append((arr[idx], side, color))
    elif isinstance(hands_data, (list, tuple)):
        for idx, elem in enumerate(hands_data[:2]):
            if elem is None:
                continue
            arr = np.asarray(elem, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 21:
                side = "L" if idx == 0 else "R"
                color = BMO_CYAN if idx == 0 else BMO_MAGENTA
                hands_list.append((arr, side, color))

    for hand, side, color in hands_list:
        if not np.any(hand):
            continue

        xy = hand[:, :2]
        # Convert normalized [0..1] coordinates to pixel ints if needed
        if float(np.max(np.abs(xy))) <= 1.5:
            pts = np.column_stack((xy[:, 0] * w, xy[:, 1] * h)).astype(np.int32)
        else:
            pts = xy.astype(np.int32)

        # Draw bone connections
        for i, j in HAND_CONNECTIONS:
            p1 = (int(pts[i, 0]), int(pts[i, 1]))
            p2 = (int(pts[j, 0]), int(pts[j, 1]))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        # Draw joint dots
        for idx_pt in range(pts.shape[0]):
            px, py = int(pts[idx_pt, 0]), int(pts[idx_pt, 1])
            radius = 4 if idx_pt in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (px, py), radius, BMO_WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius, color, 1, cv2.LINE_AA)

        # Place "L" or "R" tag near wrist (point 0)
        wx, wy = int(pts[0, 0]), int(pts[0, 1])
        tag_x = max(4, min(w - 26, wx + 8))
        tag_y = max(18, min(h - 6, wy + 6))
        _blend_rect(frame, tag_x - 3, tag_y - 14, tag_x + 16, tag_y + 4, BMO_DARK, 0.75)
        cv2.rectangle(
            frame,
            (tag_x - 3, tag_y - 14),
            (tag_x + 16, tag_y + 4),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            side,
            (tag_x + 1, tag_y - 1),
            _FONT,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_sign_gauge(
    frame: np.ndarray,
    buf_len: int,
    seq_len: int = 30,
    active_sign: str | None = None,
    sign_conf: float = 0.0,
) -> np.ndarray:
    """Draw the bottom-right sign buffer progress gauge and centered active sign banner."""
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    seq_max = max(1, int(seq_len))
    cur_len = max(0, min(seq_max, int(buf_len)))
    ratio = cur_len / float(seq_max)

    # Bottom-right gauge panel
    panel_w, panel_h = 200, 48
    margin = 12
    px2 = w - margin
    px1 = max(0, px2 - panel_w)
    py2 = h - margin
    py1 = max(0, py2 - panel_h)

    _blend_rect(frame, px1, py1, px2, py2, BMO_DARK, alpha=0.72)
    cv2.rectangle(frame, (px1, py1), (px2, py2), BMO_TEAL, 1, cv2.LINE_AA)

    label_text = f"SIGN BUF: {cur_len}/{seq_max}"
    cv2.putText(
        frame,
        label_text,
        (px1 + 10, py1 + 18),
        _FONT,
        0.44,
        BMO_WHITE,
        1,
        cv2.LINE_AA,
    )

    # Segmented progress bar inside panel
    bar_x1 = px1 + 10
    bar_x2 = px2 - 10
    bar_y1 = py1 + 26
    bar_y2 = py2 - 8
    bar_w = max(1, bar_x2 - bar_x1)

    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), BMO_GRAY, 1, cv2.LINE_AA)
    fill_w = int(round((bar_w - 2) * ratio))
    if fill_w > 0:
        bar_color = BMO_GREEN if cur_len >= seq_max else BMO_CYAN
        cv2.rectangle(
            frame,
            (bar_x1 + 1, bar_y1 + 1),
            (bar_x1 + 1 + fill_w, bar_y2 - 1),
            bar_color,
            -1,
        )

    # Prominent centered banner when active_sign is truthy and != "none"
    if active_sign and str(active_sign).strip().lower() != "none":
        sign_str = str(active_sign).strip().upper()
        conf_val = float(sign_conf or 0.0)
        pct = conf_val * 100.0 if conf_val <= 1.0 else conf_val
        if conf_val > 0.0:
            banner_text = f"SIGN: {sign_str}  {pct:.0f}%"
        else:
            banner_text = f"SIGN: {sign_str}"

        font_scale = 0.72
        font_thick = 2
        (tw, th), baseline = cv2.getTextSize(banner_text, _FONT, font_scale, font_thick)
        pad_x, pad_y = 18, 10
        bw = tw + pad_x * 2
        bh = th + baseline + pad_y * 2

        bx1 = max(0, (w - bw) // 2)
        bx2 = min(w, bx1 + bw)
        by2 = h - 18
        by1 = max(0, by2 - bh)

        _blend_rect(frame, bx1, by1, bx2, by2, BMO_DARK, alpha=0.82)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), BMO_GREEN, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            banner_text,
            (bx1 + pad_x, by1 + pad_y + th),
            _FONT,
            font_scale,
            BMO_GREEN,
            font_thick,
            cv2.LINE_AA,
        )

    return frame


def draw_status_bar(
    frame: np.ndarray,
    fps: float,
    device: str = "cuda",
    toggles: Mapping[str, bool] | None = None,
    latencies: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Draw a 38px translucent top status bar with BMO-CV badge, FPS, device, and toggles."""
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    bar_h = min(38, h)

    _blend_rect(frame, 0, 0, w, bar_h, BMO_DARK, alpha=0.78)
    cv2.line(frame, (0, bar_h - 1), (w, bar_h - 1), BMO_TEAL, 1, cv2.LINE_AA)

    # Left: "BMO-CV" badge
    badge_x1, badge_y1, badge_x2, badge_y2 = 10, 7, 84, bar_h - 7
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), BMO_TEAL, -1)
    cv2.putText(
        frame,
        "BMO-CV",
        (badge_x1 + 7, badge_y2 - 6),
        _FONT,
        0.48,
        BMO_DARK,
        2,
        cv2.LINE_AA,
    )

    # FPS + [DEVICE]
    fps_val = float(fps or 0.0)
    dev_str = str(device or "cuda").upper()
    info_text = f"FPS: {fps_val:4.1f}  [{dev_str}]"
    if latencies:
        lat_parts = [f"{k}:{v:.0f}ms" for k, v in list(latencies.items())[:3]]
        if lat_parts:
            info_text += "  " + " ".join(lat_parts)

    cv2.putText(
        frame,
        info_text,
        (badge_x2 + 12, 24),
        _FONT,
        0.48,
        BMO_WHITE,
        1,
        cv2.LINE_AA,
    )

    # Right: Toggle indicators
    if toggles:
        cursor_x = w - 10
        font_scale = 0.42
        for key, active in reversed(list(toggles.items())):
            label = str(key)
            (tw, th), _ = cv2.getTextSize(label, _FONT, font_scale, 1)
            pill_w = tw + 12
            px2 = cursor_x
            px1 = px2 - pill_w
            if px1 < badge_x2 + 140:
                break
            py1, py2 = 8, bar_h - 8
            fg_color = BMO_GREEN if active else BMO_GRAY
            _blend_rect(frame, px1, py1, px2, py2, BMO_DARK, alpha=0.6)
            cv2.rectangle(frame, (px1, py1), (px2, py2), fg_color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                label,
                (px1 + 6, py1 + th + 4),
                _FONT,
                font_scale,
                fg_color,
                1,
                cv2.LINE_AA,
            )
            cursor_x = px1 - 6

    return frame


def draw_event_feed(
    frame: np.ndarray,
    events: Sequence[str] | None,
    max_events: int = 5,
) -> np.ndarray:
    """Draw a translucent event log box in the bottom-left listing recent events."""
    if frame is None or not events:
        return frame

    recent = [str(e) for e in list(events)[-max(1, int(max_events)) :] if e]
    if not recent:
        return frame

    h, w = frame.shape[:2]
    font_scale = 0.42
    line_h = 18
    pad_x, pad_y = 10, 8

    max_tw = 0
    for line in recent:
        (tw, _), _ = cv2.getTextSize(line, _FONT, font_scale, 1)
        max_tw = max(max_tw, tw)

    box_w = min(w - 24, max(160, max_tw + pad_x * 2))
    box_h = len(recent) * line_h + pad_y * 2

    x1 = 12
    x2 = x1 + box_w
    y2 = h - 12
    y1 = max(42, y2 - box_h)

    _blend_rect(frame, x1, y1, x2, y2, BMO_DARK, alpha=0.68)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BMO_TEAL, 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y2), BMO_CYAN, 3, cv2.LINE_AA)

    for idx, line in enumerate(recent):
        ty = y1 + pad_y + (idx + 1) * line_h - 5
        if ty > y2 - 4:
            break
        color = BMO_WHITE if idx == len(recent) - 1 else BMO_TEAL
        cv2.putText(
            frame,
            line,
            (x1 + pad_x, ty),
            _FONT,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def render_hud(
    frame: np.ndarray,
    state: Mapping[str, Any] | None,
    events: Sequence[str] | None = None,
    toggles: Mapping[str, bool] | None = None,
) -> np.ndarray:
    """Render the complete BMO HUD overlay from a unified state dictionary."""
    if frame is None:
        return frame

    st = state or {}

    # 1. Object detections
    draw_objects(frame, st.get("objects"))

    # 2. Face detections (accept either `faces` list or `face` item/list)
    faces_data = st.get("faces")
    if faces_data is None and st.get("face") is not None:
        raw_face = st.get("face")
        if isinstance(raw_face, (list, tuple)) and raw_face and isinstance(
            raw_face[0], (list, tuple, Mapping)
        ):
            faces_data = raw_face
        else:
            faces_data = [raw_face]
    draw_faces(frame, faces_data)

    # 3. Hand landmarks skeleton
    draw_hand_skeleton(frame, st.get("raw_hands"))

    # 4. Sign buffer gauge & active sign banner
    draw_sign_gauge(
        frame,
        buf_len=int(st.get("sign_buf", 0) or 0),
        seq_len=int(st.get("seq_len", 30) or 30),
        active_sign=st.get("sign"),
        sign_conf=float(st.get("sign_conf", 0.0) or 0.0),
    )

    # 5. Event feed
    ev = events if events is not None else st.get("events")
    draw_event_feed(frame, ev)

    # 6. Top status bar
    tg = toggles if toggles is not None else st.get("toggles")
    draw_status_bar(
        frame,
        fps=float(st.get("fps", 0.0) or 0.0),
        device=str(st.get("device", "cuda") or "cuda"),
        toggles=tg,
        latencies=st.get("latencies"),
    )

    return frame
