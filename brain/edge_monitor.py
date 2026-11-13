"""Local mute state monitor for BMO Brain service."""

from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any

import numpy as np
from brain.audio.io import AudioSink, AudioSource
from brain.state import BrainState, BrainStateMachine


class EdgeMonitor:
    """Manages local mute state and audio source/sink selection.

    No external edge hardware - runs entirely on the local laptop.
    """

    def __init__(
        self,
        state_machine: BrainStateMachine,
        fallback_source: AudioSource,
        fallback_sink: AudioSink,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state_machine = state_machine
        self._fallback_source = fallback_source
        self._fallback_sink = fallback_sink
        self._clock = clock

        self._muted: bool = False

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def edge_online(self) -> bool:
        # No external edge hardware - always "online" locally
        return True

    @property
    def active_source(self) -> AudioSource | None:
        if self._muted:
            return None
        return self._fallback_source

    @property
    def active_sink(self) -> AudioSink:
        return self._fallback_sink

    def set_muted(self, muted: bool = True) -> None:
        """Set local mute state."""
        self._muted = bool(muted)
        if self._muted:
            self.active_sink.stop()
            self._state_machine.transition(BrainState.MUTED)
        else:
            if self._state_machine.state == BrainState.MUTED:
                self._state_machine.transition(BrainState.IDLE)

    def is_muted(self) -> bool:
        """Return whether mute state is active."""
        return self._muted

    def handle_hw_event(self, event: Any) -> None:
        """Handle mute switch events from local input (e.g., keyboard shortcut)."""
        if isinstance(event, dict):
            kind = event.get("kind")
            value = event.get("value")
        else:
            try:
                data = json.loads(event) if isinstance(event, (bytes, str)) else {}
                kind = data.get("kind")
                value = data.get("value")
            except Exception:
                return

        if kind == "mute_switch":
            if value is True:
                self.set_muted(True)
            elif value is False:
                self.set_muted(False)

    def write_samples(self, pcm: np.ndarray | bytes) -> None:
        """Write audio samples to the active sink."""
        self.active_sink.write_samples(pcm)

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        if self._muted:
            return None
        source = self.active_source
        if source is None:
            return None
        try:
            return source.read_frame(timeout_s=timeout_s)
        except TypeError:
            return source.read_frame()
