"""Edge hardware monitor managing mute switch state and fallback audio I/O."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from bmo_contracts.models import HwEvent, HwStatus
from bmo_contracts.topics import HW_EVENT, HW_STATUS
from brain.audio.io import AudioSink, AudioSource
from brain.settings import EdgeSettings
from brain.state import BrainState, BrainStateMachine


class EdgeMonitor:
    """Monitors edge hardware status and events, managing mute state and audio fallback.

    Failover debounce requires N consecutive opposite status messages before transitioning
    between edge online and fallback state. Note: this debouncing slows failover transition
    by up to N * heartbeat period (e.g. 3 * heartbeat period).
    """

    def __init__(
        self,
        state_machine: BrainStateMachine,
        fallback_source: AudioSource,
        fallback_sink: AudioSink,
        edge_source: AudioSource | None = None,
        edge_sink: AudioSink | None = None,
        heartbeat_timeout_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        failover_debounce: int = EdgeSettings().failover_debounce,
    ) -> None:
        self._state_machine = state_machine
        self._fallback_source = fallback_source
        self._fallback_sink = fallback_sink
        self._edge_source = edge_source
        self._edge_sink = edge_sink
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._clock = clock
        self._failover_debounce = max(1, int(failover_debounce))

        self._muted: bool = False
        self._edge_online: bool = False
        self._has_received_status: bool = False
        self._consecutive_opposite: int = 0
        self._last_heartbeat_ts: float = 0.0

    @property
    def failover_debounce(self) -> int:
        return self._failover_debounce

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def edge_online(self) -> bool:
        return (
            self._edge_online
            and (self._clock() - self._last_heartbeat_ts) <= self._heartbeat_timeout_s
            and self._edge_source is not None
            and self._edge_sink is not None
        )

    @property
    def active_source(self) -> AudioSource | None:
        if self._muted:
            return None
        if self.edge_online and self._edge_source is not None:
            return self._edge_source
        return self._fallback_source

    @property
    def active_sink(self) -> AudioSink:
        if self.edge_online and self._edge_sink is not None:
            return self._edge_sink
        return self._fallback_sink

    def handle_hw_event(self, event: HwEvent | bytes | str) -> None:
        if isinstance(event, (bytes, str)):
            event = HwEvent.model_validate_json(event)
        if event.kind == "mute_switch":
            if event.value is True:
                self._muted = True
                self.active_sink.stop()
                self._state_machine.transition(BrainState.MUTED)
            elif event.value is False:
                self._muted = False
                if self._state_machine.state == BrainState.MUTED:
                    self._state_machine.transition(BrainState.IDLE)

    def handle_hw_status(self, status: HwStatus | bytes | str) -> None:
        """Process incoming hardware status with failover debouncing.

        The initial status received by the monitor is applied promptly. Subsequent status
        changes require N consecutive opposite status messages before transitioning state
        (N * heartbeat period delay).
        """
        if isinstance(status, (bytes, str)):
            status = HwStatus.model_validate_json(status)
        incoming = bool(status.online)
        now = self._clock()
        if incoming:
            self._last_heartbeat_ts = now

        if not self._has_received_status:
            self._has_received_status = True
            self._edge_online = incoming
            self._consecutive_opposite = 0
            return

        if incoming == self._edge_online:
            self._consecutive_opposite = 0
            return

        self._consecutive_opposite += 1
        if self._consecutive_opposite >= self._failover_debounce:
            self._edge_online = incoming
            self._consecutive_opposite = 0

    def handle_mqtt_message(self, topic: str, payload: bytes | str) -> None:
        if topic == HW_EVENT:
            self.handle_hw_event(payload)
        elif topic == HW_STATUS:
            self.handle_hw_status(payload)

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        if self.muted:
            return None
        source = self.active_source
        if source is None:
            return None
        try:
            return source.read_frame(timeout_s=timeout_s)
        except TypeError:
            return source.read_frame()
