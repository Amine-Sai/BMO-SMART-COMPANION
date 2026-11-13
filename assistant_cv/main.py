"""Unified Real-Time BMO Computer Vision Pipeline with Lazy Mode & MQTT Bridge.

Integrates:
  1. Object & Person Detection / Tracking (YOLO26s, CUDA FP16, ~30 Hz)
  2. Synchronized Face Recognition (InsightFace SCRFD + ArcFace, CUDA, Event-Driven Lazy Mode)
  3. Hands-Free Voice Auto-Enrollment (Programmatic multi-shot collector)
  4. Short-Term Visual Memory (Rolling window context for TTS / LLM integration)
  5. Real-Time MQTT JSON Bridge (Sub-millisecond TCP_NODELAY command & event streaming)
  6. Sleek BMO HUD with interactive controls and status overlays.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
import pathlib
import sys
import threading
import time
from typing import Any
import warnings

# Suppress library deprecation and empty line warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import cv2 as cv
import numpy as np

try:
    from . import hud
    from .config import (
        AUTO_ENROLL_SHOTS,
        CAM_INDEX,
        DEVICE,
        FACE_TAU,
        LAZY_MODE_DEFAULT,
        MAX_FPS,
        MEMORY_WINDOW_SEC,
        MQTT_HOST,
        MQTT_PORT,
        OBJ_FRAME_STRIDE,
    )
    from .enroll import AutoEnrollSession, AutoObjectEnrollSession
    from .faces import FaceIdentifier
    from .memory import PersonFaceSynchronizer, ShortTermMemory, get_proximity, get_screen_position
    from .mqtt_bridge import MQTTBridge
    from .objects import get_detector
except ImportError:
    parent_dir = str(pathlib.Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import hud
    from config import (
        AUTO_ENROLL_SHOTS,
        CAM_INDEX,
        DEVICE,
        FACE_TAU,
        LAZY_MODE_DEFAULT,
        MAX_FPS,
        MEMORY_WINDOW_SEC,
        MQTT_HOST,
        MQTT_PORT,
        OBJ_FRAME_STRIDE,
    )
    from enroll import AutoEnrollSession, AutoObjectEnrollSession
    from faces import FaceIdentifier
    from memory import PersonFaceSynchronizer, ShortTermMemory, get_proximity, get_screen_position
    from mqtt_bridge import MQTTBridge
    from objects import get_detector



# ---------------------------------------------------------------------------
# Threaded Camera
# ---------------------------------------------------------------------------
class Camera:
    """Threaded OpenCV video capture to eliminate frame buffer lag."""

    def __init__(self, cam_index: int = CAM_INDEX) -> None:
        self.cam_index = cam_index
        self.cap = cv.VideoCapture(cam_index)
        self.cap.set(cv.CAP_PROP_FPS, MAX_FPS)
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
        frame_interval = 1.0 / MAX_FPS
        while self.running and self.cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.ok = ok
                self.frame = frame
            elapsed = time.perf_counter() - t0
            sleep_needed = frame_interval - elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)

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
# Asynchronous Face ID & Auto-Enroll Worker (Lazy Event-Driven)
# ---------------------------------------------------------------------------
class AsyncFaceWorker:
    """Event-driven Face Recognition and Auto-Enroll worker that sleeps when idle."""

    def __init__(self, tau: float = FACE_TAU) -> None:
        self.tau = tau
        self.identifier = FaceIdentifier()
        self.lock = threading.Lock()
        self.wake_event = threading.Event()
        self.running = False
        self.latest_frame: np.ndarray | None = None
        self.faces: list[tuple[str, float, tuple[int, int, int, int]]] = []
        self.last_latency_ms: float = 0.0
        self.thread: threading.Thread | None = None

        # Auto-enrollment session
        self.auto_enroll_session: AutoEnrollSession | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit a frame and immediately wake the worker thread."""
        with self.lock:
            self.latest_frame = frame
        self.wake_event.set()

    def set_enroll_session(self, session: AutoEnrollSession | None) -> None:
        with self.lock:
            self.auto_enroll_session = session
        if session is not None:
            self.wake_event.set()

    def get_results(self) -> tuple[list[tuple[str, float, tuple[int, int, int, int]]], float, dict[str, Any] | None]:
        """Return (faces, latency_ms, enroll_info_dict_or_None)."""
        with self.lock:
            enroll_info = None
            if self.auto_enroll_session is not None:
                enroll_info = {
                    "name": self.auto_enroll_session.name,
                    "shots": len(self.auto_enroll_session.embeddings),
                    "target": self.auto_enroll_session.target_shots,
                    "msg": self.auto_enroll_session.status_msg,
                    "completed": self.auto_enroll_session.completed,
                    "failed": self.auto_enroll_session.failed,
                    "already_registered": getattr(self.auto_enroll_session, "already_registered", False),
                    "already_registered_as": getattr(self.auto_enroll_session, "already_registered_as", None),
                }
            return list(self.faces), self.last_latency_ms, enroll_info

    def reload_gallery(self) -> int:
        return self.identifier.reload_gallery()

    def _worker_loop(self) -> None:
        while self.running:
            # Sleep until woken up by submit_frame or enroll request
            woken = self.wake_event.wait(timeout=0.5)
            if not self.running:
                break
            if not woken:
                continue

            self.wake_event.clear()
            frame_to_process = None
            session = None

            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None
                session = self.auto_enroll_session

            if frame_to_process is not None:
                t0 = time.perf_counter()
                try:
                    raw_faces = self.identifier.app.get(frame_to_process)
                    lat = (time.perf_counter() - t0) * 1000.0

                    # Handle auto-enrollment stepping
                    if session is not None and not (session.completed or session.failed):
                        session.step(raw_faces)
                        if session.completed:
                            self.identifier.reload_gallery()

                    # Match detected faces against gallery
                    res: list[tuple[str, float, tuple[int, int, int, int]]] = []
                    for f in raw_faces:
                        name = "unknown"
                        score = 0.0
                        if self.identifier._gallery_names and self.identifier._gallery_matrix.shape[0] > 0:
                            emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
                            norm_val = float(np.linalg.norm(emb))
                            if norm_val > 1e-8:
                                emb = emb / norm_val
                            sims = self.identifier._gallery_matrix @ emb
                            best_idx = int(np.argmax(sims))
                            score = float(sims[best_idx])
                            name = self.identifier._gallery_names[best_idx]

                        bbox_coords = tuple(f.bbox.astype(int).tolist())
                        bbox: tuple[int, int, int, int] = (
                            int(bbox_coords[0]),
                            int(bbox_coords[1]),
                            int(bbox_coords[2]),
                            int(bbox_coords[3]),
                        )
                        res.append((name if score >= self.tau else "unknown", round(float(score), 3), bbox))

                    with self.lock:
                        self.faces = res
                        self.last_latency_ms = lat
                        if session is not None and (session.completed or session.failed):
                            # Retain session for one frame then clear
                            pass
                except Exception as exc:
                    print(f"[FaceWorker] Inference error: {exc}", file=sys.stderr)

    def stop(self) -> None:
        self.running = False
        self.wake_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)


# ---------------------------------------------------------------------------
# Edge-Triggered Event Detector with Bridge Forwarding & Debouncing
# ---------------------------------------------------------------------------
class EventDetector:
    """Detects transitions and emits structured events for MQTT / Console with temporal debouncing."""

    def __init__(self, mqtt_bridge: MQTTBridge | None = None) -> None:
        self.mqtt_bridge = mqtt_bridge
        self.prev_person_count = 0
        self.pending_person_count = 0
        self.person_stable_frames = 0
        self.prev_person_tracks: dict[int, dict[str, Any]] = {}

        self.prev_known_faces: set[str] = set()

        self.prev_object_counts: dict[str, int] = {}
        self.pending_object_counts: dict[str, int] = {}
        self.object_stable_frames = 0

        self.history: list[str] = []

    def update(
        self,
        objects: list[dict[str, Any]],
        faces: list[tuple[str, float, tuple[int, int, int, int]]],
    ) -> list[str]:
        new_events: list[str] = []
        now_str = time.strftime("%H:%M:%S")

        # 1. Person entry/exit with 2-frame debounce (seen 2 consecutive times)
        persons = [o for o in objects if str(o.get("label", "")).lower() == "person"]
        curr_person_count = len(persons)

        if curr_person_count != self.pending_person_count:
            self.pending_person_count = curr_person_count
            self.person_stable_frames = 0
        else:
            self.person_stable_frames += 1

        if self.person_stable_frames >= 1 and self.pending_person_count != self.prev_person_count:
            if self.pending_person_count > self.prev_person_count:
                diff = self.pending_person_count - self.prev_person_count
                ev = f"[{now_str}] Person entered (+{diff})"
                new_events.append(ev)
                first_p = persons[0] if persons else {}
                tid = first_p.get("id") or first_p.get("track_id")
                bbox = first_p.get("bbox", (0, 0, 0, 0))
                pos = first_p.get("position") or get_screen_position(bbox)
                prox = first_p.get("proximity") or get_proximity(bbox)
                if prox == "close":
                    prox = "near"
                self._emit(
                    "person_detected",
                    {
                        "track_id": tid,
                        "pos": pos,
                        "proximity": prox,
                        "count": self.pending_person_count,
                        "diff": diff,
                    },
                )
            elif self.pending_person_count < self.prev_person_count:
                diff = self.prev_person_count - self.pending_person_count
                ev = f"[{now_str}] Person left (-{diff})"
                new_events.append(ev)
                curr_ids = {
                    (p.get("id") or p.get("track_id"))
                    for p in persons
                    if (p.get("id") or p.get("track_id")) is not None
                }
                left_ids = set(self.prev_person_tracks.keys()) - curr_ids
                left_tid = next(iter(left_ids)) if left_ids else None
                self._emit(
                    "person_left",
                    {
                        "track_id": left_tid,
                        "count": self.pending_person_count,
                        "diff": diff,
                    },
                )
            self.prev_person_count = self.pending_person_count

        self.prev_person_tracks = {
            (p.get("id") or p.get("track_id")): p for p in persons if (p.get("id") or p.get("track_id")) is not None
        }

        # 2. Face recognition event
        curr_known = {name for name, score, _ in faces if name.lower() not in ("unknown", "none")}
        for name in curr_known - self.prev_known_faces:
            ev = f"[{now_str}] Face recognized: {name}"
            new_events.append(ev)
            face_conf = 1.0
            face_box = (0, 0, 0, 0)
            for fname, fscore, fbbox in faces:
                if fname == name:
                    face_conf = float(fscore)
                    face_box = fbbox
                    break
            tid = None
            fcx = (face_box[0] + face_box[2]) / 2.0
            fcy = (face_box[1] + face_box[3]) / 2.0
            for p in persons:
                px1, py1, px2, py2 = p.get("bbox", (0, 0, 0, 0))
                if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                    tid = p.get("id") or p.get("track_id")
                    break
            self._emit("face_identified", {"track_id": tid, "name": name, "conf": round(face_conf, 2)})
        self.prev_known_faces = curr_known

        # 3. Non-person object detection events with 2-frame debounce
        curr_obj_counts: dict[str, int] = {}
        curr_obj_confs: dict[str, float] = {}
        for o in objects:
            lbl = str(o.get("registered_name") or o.get("label", "")).lower()
            if lbl and lbl != "person":
                curr_obj_counts[lbl] = curr_obj_counts.get(lbl, 0) + 1
                curr_obj_confs[lbl] = max(curr_obj_confs.get(lbl, 0.0), float(o.get("conf", 1.0)))

        if curr_obj_counts != self.pending_object_counts:
            self.pending_object_counts = curr_obj_counts.copy()
            self.object_stable_frames = 0
        else:
            self.object_stable_frames += 1

        if self.object_stable_frames >= 1 and self.pending_object_counts != self.prev_object_counts:
            all_labels = set(self.pending_object_counts.keys()) | set(self.prev_object_counts.keys())
            for lbl in all_labels:
                c_now = self.pending_object_counts.get(lbl, 0)
                c_prev = self.prev_object_counts.get(lbl, 0)
                if c_now > c_prev:
                    conf = curr_obj_confs.get(lbl, 1.0)
                    ev = f"[{now_str}] Object detected: {lbl} (+{c_now - c_prev})"
                    new_events.append(ev)
                    self._emit(
                        "object_seen",
                        {"label": lbl, "conf": round(conf, 2), "count": c_now, "diff": c_now - c_prev},
                    )
                elif c_now < c_prev:
                    ev = f"[{now_str}] Object removed: {lbl} (-{c_prev - c_now})"
                    new_events.append(ev)
                    self._emit("object_removed", {"label": lbl, "count": c_now, "diff": c_prev - c_now})
            self.prev_object_counts = self.pending_object_counts.copy()

        for e in new_events:
            self.history.append(e)
            if len(self.history) > 10:
                self.history.pop(0)

        return new_events

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {"event": event_type, "timestamp": time.time(), "data": data}
        if self.mqtt_bridge is not None:
            self.mqtt_bridge.publish_event(event_type, data)
        print(f"[EVENT] {payload}", flush=True)


# ---------------------------------------------------------------------------
# Main Unified Vision Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    cam_index: int = CAM_INDEX,
    headless: bool = False,
    max_frames: int | None = None,
    mirror: bool = False,
    lazy: bool = LAZY_MODE_DEFAULT,
    mqtt_host: str = MQTT_HOST,
    mqtt_port: int = MQTT_PORT,
    enable_mqtt: bool = True,
) -> dict[str, Any]:
    """Run the unified multi-rate vision pipeline with Lazy Mode and MQTT bridge."""
    print("=" * 68)
    print("  BMO Computer Vision System — Unified Pipeline (Lazy Mode & MQTT)")
    print(f"  Device: {DEVICE.upper()} | Camera: {cam_index} | Lazy Mode: {lazy}")
    print(f"  MQTT Bridge: {'ENABLED (' + mqtt_host + ':' + str(mqtt_port) + ')' if enable_mqtt else 'DISABLED'}")
    print("=" * 68)

    # Initialize MQTT Bridge
    mqtt_bridge: MQTTBridge | None = None
    if enable_mqtt:
        mqtt_bridge = MQTTBridge(host=mqtt_host, port=mqtt_port)
        mqtt_bridge.start()

    # Initialize Core Components
    print("[1/3] Loading YOLO26s Object & Person Detector...")
    obj_detector = get_detector()

    print("[2/3] Initializing Event-Driven Async Face Worker (SCRFD + ArcFace)...")
    face_worker = AsyncFaceWorker()
    face_worker.start()

    print("[3/3] Initializing Person-Face Synchronizer & Short-Term Memory...")
    synchronizer = PersonFaceSynchronizer(registry=face_worker.identifier.registry)
    short_term_memory = ShortTermMemory(window_sec=MEMORY_WINDOW_SEC, registry=face_worker.identifier.registry)
    event_detector = EventDetector(mqtt_bridge=mqtt_bridge)

    events_log: list[str] = ["BMO Vision Pipeline Started"]
    auto_enroll_session: AutoEnrollSession | None = None
    auto_enroll_req_id: str | None = None
    auto_object_enroll_session: AutoObjectEnrollSession | None = None

    # Active runtime toggles
    toggles = {
        "1:OBJ": True,
        "2:FACE": True,
        "L:LAZY": lazy,
        "T:TRK": True,  # Keep tracking enabled for person track IDs
    }

    camera = Camera(cam_index=cam_index)
    if not camera.is_opened():
        print(f"[ERROR] Failed to open camera device /dev/video{cam_index}.", file=sys.stderr)
        face_worker.stop()
        if mqtt_bridge:
            mqtt_bridge.stop()
        return {"error": "Camera open failure"}

    fps: float = 0.0
    t_prev = time.perf_counter()
    frame_count = 0
    start_time = time.perf_counter()

    obj_latency: float = 0.0
    cached_detections: list[dict[str, Any]] = []
    last_context_pub_time = 0.0
    last_scene_pub_time = 0.0
    last_scene_sig: Any = None
    # In lazy mode the heavy detectors stay off until the brain requests a
    # "look". This timestamp keeps them active for a short window.
    vision_active_until: float = 0.0
    # Give freshly-enabled detectors a moment to produce results before
    # publishing (otherwise we publish an empty scene the instant "look" lands).
    vision_grace_until: float = 0.0
    # What the current look is focused on: "people", "objects", or "both".
    vision_target_kind: str | None = None
    # Identifies one logical look. The brain repeats the same id while it
    # waits, so a retry does not suppress the scene publish; a new question
    # gets a new id and therefore a fresh scene even if the view is unchanged.
    vision_request_id: str = ""
    # On-demand camera window: the pipeline is headless, but a "look" request
    # pops up the live HUD for the duration of the look.
    show_window: bool = False
    show_window_until: float = 0.0
    window_opened: bool = False

    print("\nInteractive Hotkeys:")
    print("  1 / 2    : Toggle Objects or Face ID modules")
    print("  l         : Toggle Lazy Mode (sleep idle workers)")
    print("  e         : Trigger hands-free auto-enrollment for 'User'")
    print("  o         : Register / enroll new object held in view")
    print("  r         : Hot-reload face & object galleries")
    print("  m         : Toggle mirror mode")
    print("  q / ESC   : Quit\n")

    try:
        while camera.is_opened():
            t_cycle_start = time.perf_counter()
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

            # ---------------------------------------------------------------
            # 1. Process Incoming MQTT Commands
            # ---------------------------------------------------------------
            if mqtt_bridge is not None:
                for cmd in mqtt_bridge.poll_commands():
                    cmd_type = str(cmd.get("command", "")).lower()
                    req_id = cmd.get("request_id")
                    contract_id = cmd.get("id")
                    args_dict = cmd.get("args") if isinstance(cmd.get("args"), dict) else {}

                    if cmd_type in ("enroll", "enroll_person", "save_person", "remember_person"):
                        target_name = str(args_dict.get("name", cmd.get("name", "User"))).strip()
                        target_shots = int(args_dict.get("shots", cmd.get("shots", AUTO_ENROLL_SHOTS)))
                        auto_enroll_session = AutoEnrollSession(
                            name=target_name,
                            target_shots=target_shots,
                            identifier=face_worker.identifier,
                        )
                        auto_enroll_req_id = str(contract_id) if contract_id else None
                        face_worker.set_enroll_session(auto_enroll_session)
                        msg = f"Starting auto-enroll for '{target_name}'"
                        events_log.append(msg)
                        if not contract_id:
                            mqtt_bridge.publish_response(cmd_type, "started", {"name": target_name}, req_id)

                    elif cmd_type == "reload_gallery":
                        reloaded_count = face_worker.reload_gallery()
                        if contract_id:
                            mqtt_bridge.publish_cmd_result(str(contract_id), True, f"reloaded:{reloaded_count}")
                        else:
                            mqtt_bridge.publish_response(cmd_type, "ok", {"count": reloaded_count}, req_id)

                    elif cmd_type == "set_modules":
                        if "objects" in args_dict:
                            toggles["1:OBJ"] = bool(args_dict["objects"])
                        if "face" in args_dict:
                            toggles["2:FACE"] = bool(args_dict["face"])
                        if contract_id:
                            mqtt_bridge.publish_cmd_result(str(contract_id), True, "modules_updated")
                        else:
                            mqtt_bridge.publish_response(cmd_type, "applied", {"toggles": toggles}, req_id)

                    elif cmd_type == "get_context":
                        ctx_snapshot = short_term_memory.get_llm_context()
                        mqtt_bridge.publish_response(cmd_type, "success", {"context": ctx_snapshot}, req_id)

                    elif cmd_type == "set_mode":
                        new_mode = str(cmd.get("mode", "")).lower()
                        if new_mode == "lazy":
                            toggles["L:LAZY"] = True
                        elif new_mode in ("all", "active"):
                            toggles["L:LAZY"] = False
                            toggles["1:OBJ"] = toggles["2:FACE"] = True
                        elif new_mode == "idle":
                            toggles["2:FACE"] = False
                        mqtt_bridge.publish_response(cmd_type, "applied", {"mode": new_mode}, req_id)

                    elif cmd_type in ("look", "capture", "activate_vision", "snapshot"):
                        try:
                            seconds = float(args_dict.get("seconds", cmd.get("seconds", 6.0)))
                        except (TypeError, ValueError):
                            seconds = 6.0
                        target = str(args_dict.get("target", cmd.get("target", "both"))).lower()
                        request_id = str(
                            args_dict.get(
                                "request_id",
                                cmd.get("request_id", ""),
                            )
                        )
                        now_look = time.monotonic()
                        # The brain repeats a look while the CV process is still
                        # loading. Do not restart the grace window on those
                        # retries, or an empty scene is never published. A new
                        # request id means a new question, even if the view and
                        # target have not changed, so force a fresh publish.
                        if target in ("people", "person", "face", "faces"):
                            next_kind = "people"
                        elif target in (
                            "objects",
                            "object",
                            "thing",
                            "things",
                            "item",
                            "items",
                        ):
                            next_kind = "objects"
                        else:
                            next_kind = "both"
                        continuing = (
                            now_look < vision_active_until
                            and next_kind == vision_target_kind
                            and request_id == vision_request_id
                        )
                        vision_target_kind = next_kind
                        vision_request_id = request_id
                        vision_active_until = now_look + max(1.0, seconds)
                        if not continuing:
                            vision_grace_until = now_look + 1.5
                        if next_kind == "people":
                            toggles["1:OBJ"] = True
                            toggles["2:FACE"] = True
                        elif next_kind == "objects":
                            toggles["1:OBJ"] = True
                            toggles["2:FACE"] = False
                        else:
                            toggles["1:OBJ"] = True
                            toggles["2:FACE"] = True
                        cached_detections = []
                        if not continuing:
                            last_scene_sig = None  # force a publish once detection runs
                        # Keep BMO's face fullscreen: only show the camera HUD
                        # when the pipeline itself is running interactively.
                        show_window = not headless
                        show_window_until = (
                            time.monotonic() + max(8.0, seconds + 2.0)
                            if show_window
                            else 0.0
                        )
                        print(
                            f"[LOOK] vision active for {seconds:.0f}s "
                            f"(target={target})",
                            flush=True,
                        )
                        events_log.append(f"Vision active for {seconds:.0f}s (target={target})")
                        mqtt_bridge.publish_response(
                            cmd_type, "success", {"seconds": seconds, "target": target}, req_id
                        )

                    elif cmd_type == "status":
                        status_data = {
                            "fps": round(fps, 1),
                            "lazy_mode": toggles["L:LAZY"],
                            "toggles": toggles,
                            "persons_in_view": len(short_term_memory.current_persons),
                        }
                        mqtt_bridge.publish_response(cmd_type, "ok", status_data, req_id)

                    elif cmd_type in ("forget", "delete_person", "forget_person"):
                        del_name = str(args_dict.get("name", cmd.get("name", ""))).strip()
                        if not del_name:
                            if contract_id:
                                mqtt_bridge.publish_cmd_result(str(contract_id), False, "Missing person name")
                            else:
                                mqtt_bridge.publish_response(
                                    cmd_type, "error", {"error": "Missing person name"}, req_id
                                )
                        else:
                            ok = face_worker.identifier.delete_identity(del_name)
                            if ok:
                                events_log.append(f"Deleted identity: {del_name}")
                                if contract_id:
                                    mqtt_bridge.publish_cmd_result(str(contract_id), True, f"Deleted {del_name}")
                                else:
                                    mqtt_bridge.publish_response(cmd_type, "success", {"name": del_name}, req_id)
                            else:
                                if contract_id:
                                    mqtt_bridge.publish_cmd_result(
                                        str(contract_id), False, f"Person '{del_name}' not found"
                                    )
                                else:
                                    mqtt_bridge.publish_response(
                                        cmd_type,
                                        "not_found",
                                        {"name": del_name, "error": f"Person '{del_name}' not found"},
                                        req_id,
                                    )

                    elif cmd_type == "rename_person":
                        old_name = str(cmd.get("old_name", "")).strip()
                        new_name = str(cmd.get("new_name", "")).strip()
                        if not old_name or not new_name:
                            mqtt_bridge.publish_response(
                                cmd_type, "error", {"error": "Missing old_name or new_name"}, req_id
                            )
                        else:
                            ok = face_worker.identifier.rename_identity(old_name, new_name)
                            if ok:
                                events_log.append(f"Renamed '{old_name}' -> '{new_name}'")
                                mqtt_bridge.publish_response(
                                    cmd_type, "success", {"old_name": old_name, "new_name": new_name}, req_id
                                )
                            else:
                                mqtt_bridge.publish_response(
                                    cmd_type, "error", {"error": f"Failed to rename '{old_name}'"}, req_id
                                )

                    elif cmd_type in ("list_persons", "list_identities", "get_roster"):
                        roster = face_worker.identifier.registry.list_identities()
                        mqtt_bridge.publish_response(
                            cmd_type, "success", {"persons": roster, "count": len(roster)}, req_id
                        )

                    elif cmd_type == "set_object_classes":
                        classes = cmd.get("classes")
                        if isinstance(classes, (list, tuple)) and classes:
                            new_classes = [str(c).strip() for c in classes if str(c).strip()]
                            obj_detector.set_classes(new_classes)
                            events_log.append(f"Classes updated ({len(new_classes)} items)")
                            mqtt_bridge.publish_response(cmd_type, "success", {"classes": new_classes}, req_id)
                        else:
                            mqtt_bridge.publish_response(
                                cmd_type, "error", {"error": "Missing or invalid classes list"}, req_id
                            )

                    elif cmd_type in ("recheck_face", "reverify_face"):
                        track_id = cmd.get("track_id")
                        if track_id is not None:
                            try:
                                track_id = int(track_id)
                            except (TypeError, ValueError):
                                track_id = None
                        synchronizer.flag_for_recheck(track_id)
                        events_log.append(f"Flagged recheck ({track_id if track_id is not None else 'all'})")
                        mqtt_bridge.publish_response(cmd_type, "success", {"flagged_track": track_id}, req_id)

                    elif cmd_type in ("register_object", "enroll_object", "save_object"):
                        target_name = str(cmd.get("name", "")).strip()
                        if not target_name:
                            mqtt_bridge.publish_response(cmd_type, "error", {"error": "Missing object name"}, req_id)
                        else:
                            shots = int(cmd.get("shots", 6))
                            synonyms = cmd.get("synonyms")
                            if isinstance(synonyms, (list, tuple)):
                                syns = [str(s).strip() for s in synonyms if str(s).strip()]
                            else:
                                syns = []
                            if shots <= 0:
                                obj_detector.register_class(target_name, syns)
                                events_log.append(f"Registered object class: {target_name}")
                                mqtt_bridge.publish_response(
                                    cmd_type, "success", {"name": target_name, "mode": "text"}, req_id
                                )
                            else:
                                auto_object_enroll_session = AutoObjectEnrollSession(
                                    name=target_name,
                                    target_shots=shots,
                                    synonyms=syns,
                                    detector=obj_detector,
                                    registry=obj_detector.registry,
                                )
                                events_log.append(f"Started object enroll for '{target_name}'")
                                mqtt_bridge.publish_response(
                                    cmd_type, "started", {"name": target_name, "shots": shots}, req_id
                                )

                    elif cmd_type in ("delete_object", "forget_object"):
                        target_name = str(cmd.get("name", "")).strip()
                        if not target_name:
                            mqtt_bridge.publish_response(cmd_type, "error", {"error": "Missing object name"}, req_id)
                        else:
                            ok, msg = obj_detector.registry.delete_object(target_name)
                            if ok:
                                obj_detector.reload_classes()
                                events_log.append(f"Deleted object: {target_name}")
                                mqtt_bridge.publish_response(cmd_type, "success", {"name": target_name}, req_id)
                            else:
                                mqtt_bridge.publish_response(
                                    cmd_type, "not_found", {"name": target_name, "error": msg}, req_id
                                )

                    elif cmd_type == "rename_object":
                        old_name = str(cmd.get("old_name", "")).strip()
                        new_name = str(cmd.get("new_name", "")).strip()
                        if not old_name or not new_name:
                            mqtt_bridge.publish_response(
                                cmd_type, "error", {"error": "Missing old_name or new_name"}, req_id
                            )
                        else:
                            ok, msg = obj_detector.registry.rename_object(old_name, new_name)
                            if ok:
                                obj_detector.reload_classes()
                                events_log.append(f"Renamed object: '{old_name}' -> '{new_name}'")
                                mqtt_bridge.publish_response(
                                    cmd_type, "success", {"old_name": old_name, "new_name": new_name}, req_id
                                )
                            else:
                                mqtt_bridge.publish_response(cmd_type, "error", {"error": msg}, req_id)

                    elif cmd_type in ("list_objects", "get_objects"):
                        roster = obj_detector.registry.list_objects()
                        mqtt_bridge.publish_response(
                            cmd_type, "success", {"objects": roster, "count": len(roster)}, req_id
                        )

            # ---------------------------------------------------------------
            # 2. Objects & Person Detection (Decoupled at fps / 3 stride)
            # ---------------------------------------------------------------
            vision_active = time.monotonic() < vision_active_until
            obj_enabled = toggles["1:OBJ"] and (not toggles["L:LAZY"] or vision_active)
            detections: list[dict[str, Any]] = []
            if obj_enabled:
                run_det = (
                    (auto_object_enroll_session is not None)
                    or (frame_count % OBJ_FRAME_STRIDE == 0)
                    or not cached_detections
                    or vision_active
                )
                if run_det:
                    t_obj0 = time.perf_counter()
                    persons_only = vision_active and vision_target_kind == "people"
                    detections = obj_detector.detect(
                        frame,
                        persons_only=persons_only,
                        track=toggles["T:TRK"],
                    )
                    obj_latency = (time.perf_counter() - t_obj0) * 1000.0
                    cached_detections = detections
                else:
                    detections = cached_detections

            # ---------------------------------------------------------------
            # 2b. Temporal Detection Fusion: merge last 3 frames for stability
            # ---------------------------------------------------------------
            # Maintain rolling buffer of last 3 frames' detections
            if not hasattr(run_pipeline, "_det_buffer"):
                run_pipeline._det_buffer = []  # list of list[dict]
            run_pipeline._det_buffer.append(detections)
            if len(run_pipeline._det_buffer) > 3:
                run_pipeline._det_buffer.pop(0)

            # Fuse detections across buffer: 
            # - For persons: keep ALL detections (different track_ids)
            # - For objects: per label, keep highest-conf bbox
            fused_detections: list[dict[str, Any]] = []
            label_best: dict[str, dict[str, Any]] = {}
            for frame_dets in run_pipeline._det_buffer:
                for det in frame_dets:
                    lbl = det.get("label")
                    if not lbl:
                        continue
                    if lbl == "person":
                        # Keep all person detections (different track_ids)
                        fused_detections.append(det)
                    else:
                        conf = det.get("conf", 0.0)
                        if lbl not in label_best or conf > label_best[lbl].get("conf", 0.0):
                            label_best[lbl] = det
            # Add best object detections
            fused_detections.extend(label_best.values())

            # Use fused detections for downstream processing
            detections = fused_detections
            if vision_active and vision_target_kind == "objects":
                detections = [
                    d for d in detections if str(d.get("label", "")).lower() != "person"
                ]

            # Step active object enrollment session if running
            object_enroll_info: dict[str, Any] | None = None
            if auto_object_enroll_session is not None:
                auto_object_enroll_session.step(frame, detections)
                object_enroll_info = {
                    "name": auto_object_enroll_session.name,
                    "shots": len(auto_object_enroll_session.embeddings),
                    "target": auto_object_enroll_session.target_shots,
                    "msg": auto_object_enroll_session.status_msg,
                    "completed": auto_object_enroll_session.completed,
                    "failed": auto_object_enroll_session.failed,
                    "already_registered": auto_object_enroll_session.already_registered,
                    "target_bbox": auto_object_enroll_session.last_target_bbox,
                }
                if auto_object_enroll_session.completed:
                    msg = f"Enrolled object '{auto_object_enroll_session.name}' successfully!"
                    events_log.append(msg)
                    event_detector._emit("object_enrolled", {"name": auto_object_enroll_session.name})
                    if mqtt_bridge:
                        mqtt_bridge.publish_response(
                            "register_object",
                            "completed",
                            {"name": auto_object_enroll_session.name},
                        )
                    auto_object_enroll_session = None
                elif auto_object_enroll_session.failed:
                    msg = f"Object enroll failed: {auto_object_enroll_session.error_msg}"
                    events_log.append(msg)
                    if mqtt_bridge:
                        mqtt_bridge.publish_response(
                            "register_object",
                            "failed",
                            {"error": auto_object_enroll_session.error_msg},
                        )
                    auto_object_enroll_session = None

            persons = [d for d in detections if str(d.get("label", "")).lower() == "person"]
            person_present = len(persons) > 0

            # ---------------------------------------------------------------
            # 3. Face ID Orchestrator (Lazy or Continuous)
            # ---------------------------------------------------------------
            needs_face = False
            if toggles["2:FACE"]:
                if auto_enroll_session is not None and not (
                    auto_enroll_session.completed or auto_enroll_session.failed
                ):
                    needs_face = True
                elif not toggles["L:LAZY"] or vision_active:
                    needs_face = True
                elif person_present:
                    needs_face = synchronizer.needs_face_recognition(persons)

                if needs_face:
                    face_worker.submit_frame(frame)

            faces, face_lat, enroll_info = face_worker.get_results() if toggles["2:FACE"] else ([], 0.0, None)

            # Fallback for close-up faces: YOLO sometimes misses a person when
            # only their face fills the frame. Treat a detected face as a
            # person so the scene still reports who is there.
            if (
                toggles["2:FACE"]
                and faces
                and vision_target_kind in ("people", "both")
                and not any(str(d.get("label", "")).lower() == "person" for d in detections)
            ):
                face = faces[0]
                if isinstance(face, Mapping):
                    fconf = float(face.get("score", face.get("conf", 0.0)))
                    fbox = tuple(map(int, face.get("bbox", (0, 0, 0, 0))))
                else:
                    fconf = float(face[1])
                    fbox = tuple(map(int, face[2]))
                fx1, fy1, fx2, fy2 = fbox
                if fx2 > fx1 and fy2 > fy1:
                    fw, fh = fx2 - fx1, fy2 - fy1
                    pad_x = int(fw * 0.7)
                    bx1 = max(0, fx1 - pad_x)
                    bx2 = min(frame.shape[1], fx2 + pad_x)
                    by1 = max(0, fy1 - int(fh * 0.4))
                    by2 = min(frame.shape[0], fy2 + int(fh * 2.6))
                    detections.append(
                        {
                            "label": "person",
                            "bbox": (bx1, by1, bx2, by2),
                            "conf": fconf,
                            "id": None,
                            "source": "face_fallback",
                        }
                    )

            # Check if auto-enroll finished
            if auto_enroll_session is not None:
                if auto_enroll_session.completed:
                    msg = f"Enrolled {auto_enroll_session.name} successfully!"
                    events_log.append(msg)
                    event_detector._emit("person_enrolled", {"name": auto_enroll_session.name})
                    if mqtt_bridge:
                        if auto_enroll_req_id:
                            mqtt_bridge.publish_cmd_result(auto_enroll_req_id, True, msg)
                        else:
                            mqtt_bridge.publish_response(
                                "enroll_person", "completed", {"name": auto_enroll_session.name}
                            )
                    auto_enroll_req_id = None
                    face_worker.set_enroll_session(None)
                    auto_enroll_session = None
                elif auto_enroll_session.failed:
                    if getattr(auto_enroll_session, "already_registered", False):
                        reg_as = getattr(auto_enroll_session, "already_registered_as", "existing identity")
                        msg = f"Auto-enroll aborted: Already registered as '{reg_as}'"
                        events_log.append(msg)
                        if mqtt_bridge:
                            if auto_enroll_req_id:
                                mqtt_bridge.publish_cmd_result(auto_enroll_req_id, False, msg)
                            else:
                                mqtt_bridge.publish_response(
                                    "enroll_person",
                                    "already_registered",
                                    {
                                        "name": auto_enroll_session.name,
                                        "registered_as": reg_as,
                                        "error": auto_enroll_session.error_msg,
                                    },
                                )
                    else:
                        msg = f"Auto-enroll failed: {auto_enroll_session.error_msg}"
                        events_log.append(msg)
                        if mqtt_bridge:
                            if auto_enroll_req_id:
                                mqtt_bridge.publish_cmd_result(auto_enroll_req_id, False, msg)
                            else:
                                mqtt_bridge.publish_response(
                                    "enroll_person",
                                    "failed",
                                    {"error": auto_enroll_session.error_msg},
                                )
                    auto_enroll_req_id = None
                    face_worker.set_enroll_session(None)
                    auto_enroll_session = None

            # ---------------------------------------------------------------
            # 4. Synchronize Person Tracks with Face Identities
            # ---------------------------------------------------------------
            synced_objects = synchronizer.sync(detections, faces)

            # ---------------------------------------------------------------
            # 5. Update Short-Term Visual Memory & Check Events
            # ---------------------------------------------------------------
            short_term_memory.update(synced_objects, frame.shape)

            new_events = event_detector.update(
                objects=synced_objects,
                faces=faces,
            )
            for ev in new_events:
                events_log.append(ev)
                if len(events_log) > 6:
                    events_log.pop(0)

            # Context & Scene publishing
            now_sec = time.time()
            if mqtt_bridge is not None:
                scene_contract = short_term_memory.get_scene_contract()
                scene_sig = (
                    tuple(
                        (p.get("track_id"), p.get("name"), p.get("pos"), p.get("proximity"))
                        for p in scene_contract["people"]
                    ),
                    tuple(scene_contract["objects"]),
                )
                if scene_sig != last_scene_sig and (now_sec - last_scene_pub_time >= 1.0):
                    has_content = bool(scene_contract["people"]) or bool(scene_contract["objects"])
                    # While idle, keep the last real scene instead of replacing
                    # it with empty detections; only active looks publish blanks.
                    # During the look grace period, wait for detectors to fill in.
                    still_grace = time.monotonic() < vision_grace_until and not has_content
                    if not still_grace and (has_content or vision_active or not toggles["L:LAZY"]):
                        mqtt_bridge.publish_scene(scene_contract)
                        last_scene_sig = scene_sig
                        last_scene_pub_time = now_sec

                if now_sec - last_context_pub_time > 5.0:
                    if person_present or short_term_memory.current_objects:
                        # Legacy context publishing removed; publish_scene handles Scene model
                        last_context_pub_time = now_sec

            # ---------------------------------------------------------------
            # 6. Build HUD State & Render
            # ---------------------------------------------------------------
            state: dict[str, Any] = {
                "objects": synced_objects,
                "faces": faces,
                "fps": fps,
                "device": DEVICE,
                "auto_enroll": enroll_info,
                "auto_object_enroll": object_enroll_info,
                "latencies": {
                    "OBJ": obj_latency,
                    "FACE": face_lat,
                },
            }

            display_active = (not headless) or show_window
            if display_active:
                display_frame = frame.copy()
                hud.render_hud(display_frame, state, events=events_log, toggles=toggles)
                try:
                    cv.imshow("BMO Vision Suite", display_frame)
                    window_opened = True
                except Exception as exc:
                    print(f"[WARN] Camera window unavailable: {exc}", flush=True)
                    show_window = False
                    display_active = False

            if display_active:
                k = cv.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    if headless:
                        show_window = False
                        try:
                            cv.destroyWindow("BMO Vision Suite")
                        except Exception:
                            pass
                    else:
                        break
                elif k == ord("1"):
                    toggles["1:OBJ"] = not toggles["1:OBJ"]
                    events_log.append(f"Objects: {'ON' if toggles['1:OBJ'] else 'OFF'}")
                elif k == ord("2"):
                    toggles["2:FACE"] = not toggles["2:FACE"]
                    events_log.append(f"Face ID: {'ON' if toggles['2:FACE'] else 'OFF'}")
                elif k in (ord("l"), ord("L")):
                    toggles["L:LAZY"] = not toggles["L:LAZY"]
                    events_log.append(f"Lazy Mode: {'ON' if toggles['L:LAZY'] else 'OFF'}")
                elif k in (ord("e"), ord("E")):
                    # Quick hotkey auto-enroll for test user
                    auto_enroll_session = AutoEnrollSession(
                        name="User",
                        target_shots=AUTO_ENROLL_SHOTS,
                        identifier=face_worker.identifier,
                    )
                    face_worker.set_enroll_session(auto_enroll_session)
                    events_log.append("Started auto-enroll for 'User'")
                elif k in (ord("o"), ord("O")):
                    # Quick hotkey auto-enroll for object held in view
                    auto_object_enroll_session = AutoObjectEnrollSession(
                        name="New_Item",
                        target_shots=6,
                        detector=obj_detector,
                        registry=obj_detector.registry,
                    )
                    events_log.append("Started object enroll for 'New_Item' (hold object in view)")
                elif k in (ord("m"), ord("M")):
                    mirror = not mirror
                    events_log.append(f"Mirror: {'ON' if mirror else 'OFF'}")
                elif k in (ord("r"), ord("R")):
                    count = face_worker.reload_gallery()
                    obj_count = len(obj_detector.reload_classes())
                    events_log.append(f"Reloaded galleries ({count} faces, {obj_count} object classes)")
                elif k in (ord("f"), ord("F")):
                    synchronizer.flag_for_recheck()
                    events_log.append("Forced face re-check for active tracks")

            if headless and show_window and time.monotonic() > show_window_until:
                show_window = False
                print("[WINDOW] closing live camera view", flush=True)
                try:
                    cv.destroyWindow("BMO Vision Suite")
                except Exception:
                    pass

            # Enforce 60 FPS cap
            cycle_elapsed = time.perf_counter() - t_cycle_start
            sleep_needed = (1.0 / MAX_FPS) - cycle_elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)

            if max_frames and frame_count >= max_frames:
                break

    finally:
        total_time = max(1e-6, time.perf_counter() - start_time)
        avg_fps = frame_count / total_time
        print("\n" + "=" * 68)
        print(f"  Session Finished: {frame_count} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")
        print("=" * 68)
        face_worker.stop()
        camera.release()
        if mqtt_bridge is not None:
            mqtt_bridge.stop()
        if window_opened or not headless:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass

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
    parser.add_argument("--no-lazy", action="store_true", help="Disable Lazy Mode (run all models continuously)")
    parser.add_argument("--no-mqtt", action="store_true", help="Disable MQTT bridge")
    parser.add_argument("--mqtt-host", type=str, default=MQTT_HOST, help=f"MQTT broker host (default: {MQTT_HOST})")
    parser.add_argument("--mqtt-port", type=int, default=MQTT_PORT, help=f"MQTT broker port (default: {MQTT_PORT})")
    args = parser.parse_args()

    run_pipeline(
        cam_index=args.cam,
        headless=args.headless,
        max_frames=args.frames,
        mirror=args.mirror,
        lazy=not args.no_lazy,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        enable_mqtt=not args.no_mqtt,
    )


if __name__ == "__main__":
    main()
