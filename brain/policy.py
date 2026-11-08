"""Proactive event policy engine for greetings and interactions."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from bmo_contracts.models import (
    Event,
    FaceIdentified,
    PersonDetected,
    PersonLeft,
    parse_event_json,
)
from brain.settings import PolicySettings
from brain.state import BrainState, BrainStateMachine


class ProactivePolicy:
    """Evaluates incoming perception events and triggers greetings when idle."""

    def __init__(
        self,
        settings: PolicySettings,
        state_machine: BrainStateMachine,
        on_greet: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._state_machine = state_machine
        self._on_greet = on_greet
        self._clock = clock
        self._last_person_ts: dict[str, float] = {}
        self._last_global_ts: float | None = None
        self._pending_person: dict[Any, float] = {}

    def handle_event(self, event: Event | bytes | str) -> str | None:
        """Handle perception event; returns greeting text if triggered."""
        if isinstance(event, (bytes, str)):
            parsed = parse_event_json(event)
        else:
            parsed = event

        if self._state_machine.state != BrainState.IDLE:
            self._pending_person.clear()
            return None

        if isinstance(parsed, FaceIdentified):
            name = parsed.data.name.strip()
            if parsed.data.conf < self._settings.face_min_conf or name.lower() in ("", "unknown", "person"):
                return None

            track_id = getattr(parsed.data, "track_id", None)
            if track_id is not None:
                self._pending_person.pop(track_id, None)
            else:
                self._pending_person.clear()

            now = self._clock()
            name_key = name.lower()
            if (
                name_key in self._last_person_ts
                and (now - self._last_person_ts[name_key]) < self._settings.greeting_cooldown_s
            ):
                return None
            if self._last_global_ts is not None and (now - self._last_global_ts) < self._settings.global_cooldown_s:
                return None

            self._last_person_ts[name_key] = now
            self._last_global_ts = now
            greeting = f"Hi {parsed.data.name}!"
            self._on_greet(greeting)
            return greeting

        if isinstance(parsed, PersonDetected):
            now = self._clock()
            track_id = getattr(parsed.data, "track_id", None)
            key = track_id if track_id is not None else "anonymous"
            if key not in self._pending_person:
                self._pending_person[key] = now
            return None

        if isinstance(parsed, PersonLeft):
            track_id = getattr(parsed.data, "track_id", None)
            if track_id is None:
                self._pending_person.clear()
            else:
                self._pending_person.pop(track_id, None)
                self._pending_person.pop("anonymous", None)
            return None

        return None

    def tick(self) -> str | None:
        """Check pending detections periodically; returns greeting text if triggered."""
        now = self._clock()
        if self._state_machine.state != BrainState.IDLE:
            self._pending_person.clear()
            return None

        expired_tracks = [
            track_id
            for track_id, first_seen in self._pending_person.items()
            if (now - first_seen) >= self._settings.walk_by_window_s
        ]
        if not expired_tracks:
            return None

        for track_id in expired_tracks:
            self._pending_person.pop(track_id, None)

        if self._last_global_ts is not None and (now - self._last_global_ts) < self._settings.global_cooldown_s:
            return None

        self._last_global_ts = now
        greeting = "Hello there!"
        self._on_greet(greeting)
        return greeting
