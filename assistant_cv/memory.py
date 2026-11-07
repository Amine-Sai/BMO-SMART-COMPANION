"""Person-Face Track Synchronization and Short-Term Visual Memory for BMO."""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Mapping, Sequence


def _get_box_center(bbox: Sequence[int | float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox[:4]
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _box_area(bbox: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def get_screen_position(bbox: Sequence[int | float], frame_w: int = 640) -> str:
    """Return 'left', 'center', or 'right' based on bounding box horizontal midpoint."""
    cx, _ = _get_box_center(bbox)
    w = max(1.0, float(frame_w))
    ratio = cx / w
    if ratio < 0.38:
        return "left"
    if ratio > 0.62:
        return "right"
    return "center"


def get_proximity(bbox: Sequence[int | float], frame_area: float = 640 * 480) -> str:
    """Return 'close', 'medium', or 'far' based on relative bounding box coverage."""
    area = _box_area(bbox)
    ratio = area / max(1.0, frame_area)
    if ratio > 0.25:
        return "close"
    if ratio > 0.08:
        return "medium"
    return "far"


class PersonFaceSynchronizer:
    """Binds YOLO person track IDs to ArcFace identity embeddings with temporal persistence."""

    def __init__(
        self,
        track_ttl_sec: float = 6.0,
        reverify_interval_sec: float = 3.0,
        registry: Any = None,
    ) -> None:
        self.track_ttl = float(track_ttl_sec)
        self.reverify_interval = float(reverify_interval_sec)
        self.registry = registry
        # track_id -> {"name": str, "face_conf": float, "last_seen": float, "last_verified": float, "bbox": tuple}
        self.tracks: dict[int, dict[str, Any]] = {}
        self.force_recheck_ids: set[int] = set()

    def flag_for_recheck(self, track_id: int | None = None) -> None:
        """Explicitly request face re-checking for a track ID, or all tracks if None."""
        if track_id is not None:
            self.force_recheck_ids.add(track_id)
        else:
            self.force_recheck_ids.update(self.tracks.keys())

    def sync(
        self,
        person_detections: list[dict[str, Any]],
        faces: list[tuple[str, float, tuple[int, int, int, int]]] | list[dict[str, Any]],
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronize person detections with face identities and enrich detections in-place."""
        t_now = time.perf_counter() if now is None else float(now)

        # Normalize faces input into list of (name, conf, (fx1, fy1, fx2, fy2))
        norm_faces: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for f in faces:
            if isinstance(f, Mapping):
                name = str(f.get("name", "unknown"))
                conf = float(f.get("score", f.get("conf", 0.0)))
                bbox = tuple(map(int, f.get("bbox", (0, 0, 0, 0))))
                norm_faces.append((name, conf, bbox))
            elif isinstance(f, (list, tuple)) and len(f) >= 3:
                norm_faces.append((str(f[0]), float(f[1]), tuple(map(int, f[2]))))

        # Purge stale tracks
        stale_ids = [tid for tid, data in self.tracks.items() if (t_now - data["last_seen"]) > self.track_ttl]
        for tid in stale_ids:
            del self.tracks[tid]

        # Match faces to person bounding boxes
        for p in person_detections:
            p_box = p.get("bbox", (0, 0, 0, 0))
            px1, py1, px2, py2 = p_box
            p_h = max(1, py2 - py1)
            tid = p.get("id")

            # Check if any face center lies in upper 65% of this person's bounding box
            best_match: tuple[str, float] | None = None
            best_conf = -1.0

            for fname, fconf, fbox in norm_faces:
                fcx, fcy = _get_box_center(fbox)
                if px1 <= fcx <= px2 and py1 <= fcy <= (py1 + p_h * 0.65):
                    if fconf > best_conf:
                        best_conf = fconf
                        best_match = (fname, fconf)

            if tid is not None:
                track_info = self.tracks.setdefault(
                    tid,
                    {
                        "name": "unknown",
                        "face_conf": 0.0,
                        "last_seen": t_now,
                        "last_verified": 0.0,
                        "bbox": p_box,
                    },
                )
                track_info["last_seen"] = t_now
                track_info["bbox"] = p_box

                if best_match is not None:
                    matched_name, matched_conf = best_match
                    if matched_name.lower() not in ("unknown", "none"):
                        track_info["name"] = matched_name
                        track_info["face_conf"] = matched_conf
                        track_info["last_verified"] = t_now
                        self.force_recheck_ids.discard(tid)
                        if self.registry is not None:
                            try:
                                self.registry.record_encounter(matched_name)
                            except Exception:
                                pass
                    elif track_info["name"] == "unknown":
                        track_info["face_conf"] = matched_conf
                        track_info["last_verified"] = t_now
                        self.force_recheck_ids.discard(tid)

                p["identity"] = track_info["name"]
                p["face_conf"] = track_info["face_conf"]
            else:
                # No track ID: attach one-shot best match if available
                if best_match is not None:
                    p["identity"] = best_match[0]
                    p["face_conf"] = best_match[1]
                else:
                    p["identity"] = "unknown"
                    p["face_conf"] = 0.0

        return person_detections

    def needs_face_recognition(self, person_detections: list[dict[str, Any]], now: float | None = None) -> bool:
        """Return True if any visible person needs face recognition.

        Known/registered persons on active tracks are remembered and will NOT trigger
        redundant face re-checks unless explicitly forced via flag_for_recheck().
        """
        if not person_detections:
            return False

        for p in person_detections:
            tid = p.get("id")
            if tid is None:
                # Untracked person requires identification
                return True
            if tid in self.force_recheck_ids:
                return True
            info = self.tracks.get(tid)
            if info is None:
                # New track not yet checked
                return True
            if info["name"].lower() in ("unknown", "none"):
                # Unknown person track needs identification
                return True

        return False


class ShortTermMemory:
    """Maintains a rolling temporal memory of scene entities, actions, and events for LLM queries."""

    def __init__(self, window_sec: float = 60.0, registry: Any = None) -> None:
        self.window_sec = float(window_sec)
        self.registry = registry
        self.events: deque[dict[str, Any]] = deque(maxlen=30)
        self.window_sec = float(window_sec)
        self.events: deque[dict[str, Any]] = deque(maxlen=30)
        self.recent_signs: deque[dict[str, Any]] = deque(maxlen=20)
        # identity -> {"first_seen": float, "last_seen": float, "position": str, "proximity": str}
        self.persons_history: dict[str, dict[str, Any]] = {}
        # label -> {"last_seen": float, "count": int, "positions": list[str]}
        self.objects_history: dict[str, dict[str, Any]] = {}
        self.current_persons: list[dict[str, Any]] = []
        self.current_objects: list[dict[str, Any]] = []
        self.last_update_time = time.time()

    def record_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Record a timestamped discrete transition event."""
        item = {
            "event": event_type,
            "timestamp": time.time(),
            "data": data or {},
        }
        self.events.append(item)

    def record_sign(self, sign: str, conf: float, signer: str | None = None) -> None:
        """Record a recognized sign language gesture."""
        item = {
            "sign": sign.upper(),
            "conf": round(float(conf), 2),
            "signer": signer,
            "timestamp": time.time(),
        }
        self.recent_signs.append(item)
        self.record_event("sign_recognized", item)

    def update(
        self,
        objects: list[dict[str, Any]],
        frame_shape: tuple[int, int] = (480, 640),
    ) -> None:
        """Update rolling visual memory with latest synced detections."""
        now = time.time()
        self.last_update_time = now
        fh, fw = frame_shape[:2]
        frame_area = float(fh * fw)

        curr_persons: list[dict[str, Any]] = []
        curr_objs_map: dict[str, dict[str, Any]] = {}

        for obj in objects:
            lbl = str(obj.get("label", "")).lower()
            bbox = obj.get("bbox", (0, 0, 0, 0))
            pos = get_screen_position(bbox, fw)
            prox = get_proximity(bbox, frame_area)

            if lbl == "person":
                identity = str(obj.get("identity", "unknown"))
                tid = obj.get("id")
                face_conf = float(obj.get("face_conf", 0.0))
                p_entry = {
                    "track_id": tid,
                    "identity": identity,
                    "conf": float(obj.get("conf", 0.0)),
                    "face_conf": face_conf,
                    "position": pos,
                    "proximity": prox,
                    "bbox": bbox,
                }
                curr_persons.append(p_entry)

                # Update history for this person
                hist_key = identity if identity != "unknown" else f"person_{tid}"
                if hist_key not in self.persons_history:
                    self.persons_history[hist_key] = {
                        "identity": identity,
                        "first_seen": now,
                        "last_seen": now,
                        "position": pos,
                        "proximity": prox,
                    }
                else:
                    self.persons_history[hist_key]["last_seen"] = now
                    self.persons_history[hist_key]["position"] = pos
                    self.persons_history[hist_key]["proximity"] = prox
                    if identity != "unknown":
                        self.persons_history[hist_key]["identity"] = identity
            else:
                if lbl not in curr_objs_map:
                    curr_objs_map[lbl] = {"label": lbl, "count": 1, "positions": [pos]}
                else:
                    curr_objs_map[lbl]["count"] += 1
                    if pos not in curr_objs_map[lbl]["positions"]:
                        curr_objs_map[lbl]["positions"].append(pos)

                self.objects_history[lbl] = {
                    "last_seen": now,
                    "count": curr_objs_map[lbl]["count"],
                    "positions": curr_objs_map[lbl]["positions"],
                }

        self.current_persons = curr_persons
        self.current_objects = list(curr_objs_map.values())

        # Evict history items older than window_sec
        cutoff = now - self.window_sec
        self.persons_history = {k: v for k, v in self.persons_history.items() if v["last_seen"] >= cutoff}
        self.objects_history = {k: v for k, v in self.objects_history.items() if v["last_seen"] >= cutoff}

    def get_llm_context(self) -> dict[str, Any]:
        """Generate compact, structured JSON visual context designed for TTS/LLM system prompt injection."""
        now = time.time()

        # 1. Active people
        active_p = []
        known_in_view = []
        for p in self.current_persons:
            ident = p["identity"]
            active_p.append(
                {
                    "track_id": p["track_id"],
                    "identity": ident,
                    "position": p["position"],
                    "proximity": p["proximity"],
                    "face_conf": p["face_conf"],
                }
            )
            if ident.lower() not in ("unknown", "none"):
                known_in_view.append(f"{ident} ({p['position']}, {p['proximity']})")
            else:
                known_in_view.append(f"someone ({p['position']}, {p['proximity']})")

        # 2. Recently seen people who left
        recently_left = []
        curr_keys = {p["identity"] if p["identity"] != "unknown" else f"person_{p['track_id']}" for p in self.current_persons}
        for k, v in self.persons_history.items():
            if k not in curr_keys:
                ago = round(now - v["last_seen"], 1)
                recently_left.append({"identity": v["identity"], "ago_sec": ago})

        # 3. Active objects
        obj_summary = [f"{o['count']} {o['label']}" for o in self.current_objects]

        # 4. Recent signs
        signs_summary = []
        cutoff = now - self.window_sec
        for s in reversed(self.recent_signs):
            if s["timestamp"] >= cutoff:
                ago = round(now - s["timestamp"], 1)
                signs_summary.append(
                    {
                        "sign": s["sign"],
                        "conf": s["conf"],
                        "signer": s["signer"],
                        "ago_sec": ago,
                    }
                )

        # 5. Natural language one-line summary
        summary_parts = []
        if known_in_view:
            summary_parts.append(f"Seeing {', '.join(known_in_view)}.")
        else:
            summary_parts.append("No person currently in view.")

        if obj_summary:
            summary_parts.append(f"Visible objects: {', '.join(obj_summary)}.")

        if signs_summary:
            latest_s = signs_summary[0]
            summary_parts.append(f"Recent sign: {latest_s['sign']} ({latest_s['conf']*100:.0f}%, {latest_s['ago_sec']}s ago).")

        summary_text = " ".join(summary_parts)

        # 6. Recent events
        events_list = []
        for ev in list(self.events)[-10:]:
            ago = round(now - ev["timestamp"], 1)
            events_list.append({"event": ev["event"], "ago_sec": ago, "data": ev.get("data", {})})

        # 7. Permanent known persons roster
        known_persons_roster: list[str] = []
        if self.registry is not None:
            try:
                known_persons_roster = self.registry.get_known_names()
            except Exception:
                pass
        else:
            try:
                from .registry import get_identity_registry
                known_persons_roster = get_identity_registry().get_known_names()
            except Exception:
                try:
                    from assistant_cv.registry import get_identity_registry
                    known_persons_roster = get_identity_registry().get_known_names()
                except Exception:
                    pass

        # 8. Permanent registered objects roster
        known_objects_roster: list[str] = []
        try:
            from .registry import get_object_registry
            known_objects_roster = get_object_registry().get_known_names()
        except Exception:
            try:
                from assistant_cv.registry import get_object_registry
                known_objects_roster = get_object_registry().get_known_names()
            except Exception:
                pass

        return {
            "timestamp": now,
            "summary": summary_text,
            "current_persons": active_p,
            "recent_persons": recently_left,
            "current_objects": self.current_objects,
            "recent_signs": signs_summary[:5],
            "recent_events": events_list,
            "known_persons": known_persons_roster,
            "registered_objects": known_objects_roster,
        }
