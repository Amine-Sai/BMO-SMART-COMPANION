"""Unified Real-Time BMO Computer Vision Pipeline.

Integrates:
  1. Object & Person Detection / Tracking (YOLO26s, CUDA FP16, ~30 Hz)
  2. Face Identification (InsightFace SCRFD + ArcFace, Async Background Thread ~5 Hz)
  3. Sign Language Recognition (OpenCV Zoo HandPose ONNX + SignNet GRU, ~30 Hz)
  4. Real-time BMO HUD with interactive keyboard toggles and edge-triggered events.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
import time
from typing import Any

import cv2 as cv
import numpy as np

try:
    from . import hud
    from .config import (
        CAM_INDEX,
        DEVICE,
        FACE_TAU,
        GALLERY_DIR,
        OBJ_CONF,
        SEQ_LEN,
        SIGN_CONF,
        SIGN_MODEL_PATH,
    )
    from .faces import FaceIdentifier, get_face_app
    from .live_sign import SignRecognizer, get_recognizer
    from .objects import ObjectDetector, get_detector
except ImportError:
    # Fallback when executed as standalone script
    parent_dir = str(pathlib.Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import hud
    from config import (
        CAM_INDEX,
        DEVICE,
        FACE_TAU,
        GALLERY_DIR,
        OBJ_CONF,
        SEQ_LEN,
        SIGN_CONF,
        SIGN_MODEL_PATH,
    )
    from faces import FaceIdentifier, get_face_app
    from live_sign import SignRecognizer, get_recognizer
    from objects import ObjectDetector, get_detector


# ---------------------------------------------------------------------------
# Threaded Camera
# ---------------------------------------------------------------------------
class Camera:
    """Threaded OpenCV video capture to prevent frame buffer lag."""

    def __init__(self, cam_index: int = CAM_INDEX) -> None:
        self.cam_index = cam_index
        self.cap = cv.VideoCapture(cam_index)
        self.lock = threading.Lock()
        self.running = False
        self.ok = False
        self.frame: np.ndarray | None = None
        self.thread: threading.Thread | None = None

        if self.cap.isOpened():
            self.ok, self.frame = self.cap.read()
            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def _reader_loop(self) -> None:
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ok = ok
                self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_opened(self) -> bool:
        return self.running and self.cap.isOpened()

    def release(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


# ---------------------------------------------------------------------------
# Asynchronous Face ID Worker (5 Hz)
# ---------------------------------------------------------------------------
class AsyncFaceWorker:
    """Runs Face ID in a background thread so ~300ms face latency never stalls the 30 Hz loop."""

    def __init__(self, tau: float = FACE_TAU, period: float = 0.20) -> None:
        self.tau = tau
        self.period = period
        self.identifier = FaceIdentifier()
        self.lock = threading.Lock()
        self.running = False
        self.latest_frame: np.ndarray | None = None
        self.faces: list[tuple[str, float, tuple[int, int, int, int]]] = []
        self.last_latency_ms: float = 0.0
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.latest_frame = frame

    def get_results(self) -> tuple[list[tuple[str, float, tuple[int, int, int, int]]], float]:
        with self.lock:
            return list(self.faces), self.last_latency_ms

    def reload_gallery(self) -> int:
        return self.identifier.reload_gallery()

    def _worker_loop(self) -> None:
        while self.running:
            t0 = time.perf_counter()
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is not None:
                try:
                    res = self.identifier.identify(frame_to_process, tau=self.tau)
                    lat = (time.perf_counter() - t0) * 1000.0
                    with self.lock:
                        self.faces = res
                        self.last_latency_ms = lat
                except Exception as exc:
                    print(f"[FaceWorker] Inference error: {exc}", file=sys.stderr)

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.01, self.period - elapsed)
            time.sleep(sleep_time)

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)


# ---------------------------------------------------------------------------
# Edge-Triggered Event Detector
# ---------------------------------------------------------------------------
class EventDetector:
    """Detects transitions and emits formatted events for ESP32/serial communication."""

    def __init__(self) -> None:
        self.prev_person_count = 0
        self.prev_known_faces: set[str] = set()
        self.prev_sign: str | None = None
        self.history: list[str] = []

    def update(
        self,
        objects: list[dict[str, Any]],
        faces: list[tuple[str, float, tuple[int, int, int, int]]],
        sign: str | None,
        sign_conf: float,
    ) -> list[str]:
        new_events: list[str] = []
        now_str = time.strftime("%H:%M:%S")

        # 1. Person entry/exit
        persons = [o for o in objects if str(o.get("label", "")).lower() == "person"]
        curr_person_count = len(persons)
        if curr_person_count > self.prev_person_count:
            diff = curr_person_count - self.prev_person_count
            ev = f"[{now_str}] Person entered (+{diff})"
            new_events.append(ev)
            self._emit_json("person_detected", {"count": curr_person_count, "diff": diff})
        self.prev_person_count = curr_person_count

        # 2. Face recognition event
        curr_known = {name for name, score, _ in faces if name.lower() not in ("unknown", "none")}
        for name in curr_known - self.prev_known_faces:
            ev = f"[{now_str}] Face recognized: {name}"
            new_events.append(ev)
            self._emit_json("face_identified", {"name": name})
        self.prev_known_faces = curr_known

        # 3. Sign recognition event
        if sign and sign != self.prev_sign and sign.lower() != "none":
            ev = f"[{now_str}] Sign: {sign.upper()} ({sign_conf*100:.0f}%)"
            new_events.append(ev)
            self._emit_json("sign_recognized", {"sign": sign, "conf": round(sign_conf, 2)})
        self.prev_sign = sign

        for e in new_events:
            self.history.append(e)
            if len(self.history) > 10:
                self.history.pop(0)

        return new_events

    def _emit_json(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {"event": event_type, "timestamp": time.time(), "data": data}
        print(f"[EVENT] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# Main Unified Vision Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    cam_index: int = CAM_INDEX,
    headless: bool = False,
    max_frames: int | None = None,
    mirror: bool = False,
) -> dict[str, Any]:
    """Run the unified multi-rate vision pipeline with HUD and event emission."""
    print("=" * 65)
    print("  BMO Computer Vision System — Unified Pipeline")
    print(f"  Device: {DEVICE.upper()} | Camera: {cam_index} | Headless: {headless}")
    print("=" * 65)

    # Initialize models
    print("[1/3] Loading YOLO26s Object Detector...")
    obj_detector = get_detector()
    print("[2/3] Initializing Async Face Worker (buffalo_s)...")
    face_worker = AsyncFaceWorker()
    face_worker.start()
    print("[3/3] Initializing Standalone Sign Recognizer...")
    sign_recognizer = get_recognizer()

    event_detector = EventDetector()
    events_log: list[str] = ["BMO Vision Pipeline Started"]

    # Active runtime toggles
    toggles = {
        "1:OBJ": True,
        "2:FACE": True,
        "3:SIGN": True,
        "P:PERS": False,
        "T:TRK": False,
    }

    camera = Camera(cam_index=cam_index)
    if not camera.is_opened():
        print(f"[ERROR] Failed to open camera device /dev/video{cam_index}.", file=sys.stderr)
        face_worker.stop()
        return {"error": "Camera open failure"}

    fps: float = 0.0
    t_prev = time.perf_counter()
    frame_count = 0
    start_time = time.perf_counter()

    # Latencies
    obj_latency: float = 0.0
    sign_latency: float = 0.0

    print("\nInteractive Hotkeys:")
    print("  1 / 2 / 3 : Toggle Objects, Face ID, or Sign Language modules")
    print("  p / t     : Toggle Persons-only filter / Object Tracking")
    print("  r         : Hot-reload face gallery and sign model")
    print("  m         : Toggle mirror mode")
    print("  q / ESC   : Quit\n")

    try:
        while camera.is_opened():
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            if mirror:
                frame = cv.flip(frame, 1)

            t_now = time.perf_counter()
            dt = max(1e-6, t_now - t_prev)
            t_prev = t_now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0.0 else (1.0 / dt)

            # 1. Objects module (~30 Hz)
            detections = []
            if toggles["1:OBJ"]:
                t_obj0 = time.perf_counter()
                detections = obj_detector.detect(
                    frame,
                    persons_only=toggles["P:PERS"],
                    track=toggles["T:TRK"],
                )
                obj_latency = (time.perf_counter() - t_obj0) * 1000.0

            # 2. Face ID module (~5 Hz async submission)
            if toggles["2:FACE"]:
                face_worker.submit_frame(frame)
            faces, face_latency = face_worker.get_results() if toggles["2:FACE"] else ([], 0.0)

            # 3. Sign Language module (~30 Hz)
            active_sign, active_conf, buf_len, raw_hands = None, 0.0, 0, None
            if toggles["3:SIGN"]:
                t_sign0 = time.perf_counter()
                active_sign, active_conf, buf_len, raw_hands = sign_recognizer.step(frame)
                sign_latency = (time.perf_counter() - t_sign0) * 1000.0

            # 4. Check for state transition events
            new_events = event_detector.update(
                objects=detections,
                faces=faces,
                sign=active_sign,
                sign_conf=active_conf,
            )
            for ev in new_events:
                events_log.append(ev)
                if len(events_log) > 6:
                    events_log.pop(0)

            # 5. Build unified state dict
            state: dict[str, Any] = {
                "objects": detections,
                "faces": faces,
                "sign": active_sign,
                "sign_conf": active_conf,
                "sign_buf": buf_len,
                "seq_len": SEQ_LEN,
                "raw_hands": raw_hands,
                "fps": fps,
                "device": DEVICE,
                "latencies": {
                    "obj": obj_latency,
                    "sign": sign_latency,
                    "face": face_latency,
                },
            }

            # 6. Render HUD (if not headless)
            if not headless:
                display_frame = frame.copy()
                hud.render_hud(display_frame, state, events=events_log, toggles=toggles)
                cv.imshow("BMO Vision Suite", display_frame)

                k = cv.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                elif k == ord("1"):
                    toggles["1:OBJ"] = not toggles["1:OBJ"]
                    events_log.append(f"Objects: {'ON' if toggles['1:OBJ'] else 'OFF'}")
                elif k == ord("2"):
                    toggles["2:FACE"] = not toggles["2:FACE"]
                    events_log.append(f"Face ID: {'ON' if toggles['2:FACE'] else 'OFF'}")
                elif k == ord("3"):
                    toggles["3:SIGN"] = not toggles["3:SIGN"]
                    events_log.append(f"Sign Lang: {'ON' if toggles['3:SIGN'] else 'OFF'}")
                elif k == ord("p") or k == ord("P"):
                    toggles["P:PERS"] = not toggles["P:PERS"]
                    events_log.append(f"Persons Only: {'ON' if toggles['P:PERS'] else 'OFF'}")
                elif k == ord("t") or k == ord("T"):
                    toggles["T:TRK"] = not toggles["T:TRK"]
                    events_log.append(f"Tracking: {'ON' if toggles['T:TRK'] else 'OFF'}")
                elif k == ord("m") or k == ord("M"):
                    mirror = not mirror
                    events_log.append(f"Mirror: {'ON' if mirror else 'OFF'}")
                elif k == ord("r") or k == ord("R"):
                    count = face_worker.reload_gallery()
                    sign_recognizer._try_load_model()
                    events_log.append(f"Reloaded gallery ({count} faces)")

            if max_frames and frame_count >= max_frames:
                break

    finally:
        total_time = max(1e-6, time.perf_counter() - start_time)
        avg_fps = frame_count / total_time
        print("\n" + "=" * 65)
        print(f"  Session Finished: {frame_count} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")
        print("=" * 65)
        face_worker.stop()
        camera.release()
        if not headless:
            cv.destroyAllWindows()

    return {
        "frames": frame_count,
        "elapsed_sec": total_time,
        "avg_fps": avg_fps,
        "last_fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BMO Computer Vision Unified Pipeline")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index")
    parser.add_argument("--headless", action="store_true", help="Run without graphical HUD")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames (benchmark)")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam view")
    args = parser.parse_args()

    run_pipeline(
        cam_index=args.cam,
        headless=args.headless,
        max_frames=args.frames,
        mirror=args.mirror,
    )


if __name__ == "__main__":
    main()
