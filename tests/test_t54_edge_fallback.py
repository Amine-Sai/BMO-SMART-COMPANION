"""Tests for edge hardware monitoring, mute switch handling, and audio fallback."""

from __future__ import annotations

import json
import numpy as np

from bmo_contracts.models import HwEvent, HwStatus
from bmo_contracts.topics import HW_EVENT, HW_STATUS
from brain.audio.io import AudioSource, MemorySink, MemorySource
from brain.edge_monitor import EdgeMonitor
from brain.state import BrainState, BrainStateMachine


class TrackingSink(MemorySink):
    """MemorySink spy that records stop calls."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        if hasattr(super(), "stop"):
            super().stop()


def _create_source(frame: np.ndarray | None = None) -> AudioSource:
    sample_frame = frame if frame is not None else np.zeros(160, dtype=np.int16)
    try:
        return MemorySource(frames=[sample_frame])
    except TypeError:
        try:
            return MemorySource([sample_frame])
        except TypeError:
            return MemorySource()


def _create_sink() -> TrackingSink:
    try:
        return TrackingSink()
    except TypeError:
        return TrackingSink([])


def _dummy_publish(topic: str, payload: bytes | str, qos: int = 0, retain: bool = False) -> None:
    _ = (topic, payload, qos, retain)


def test_mute_switch_transitions_and_halts_audio() -> None:
    sm = BrainStateMachine(publish_fn=_dummy_publish)
    fallback_source = _create_source()
    fallback_sink = _create_sink()
    edge_source = _create_source()
    edge_sink = _create_sink()

    monitor = EdgeMonitor(
        state_machine=sm,
        fallback_source=fallback_source,
        fallback_sink=fallback_sink,
        edge_source=edge_source,
        edge_sink=edge_sink,
    )

    # Bring edge online first
    status = HwStatus(online=True, ts=1.0, capabilities=["mic", "speaker"])
    monitor.handle_hw_status(status)
    assert monitor.edge_online is True
    assert monitor.active_sink is edge_sink

    # Mute switch pressed (True)
    mute_event = HwEvent(ts=1.0, kind="mute_switch", value=True)
    monitor.handle_hw_event(mute_event)

    assert sm.state == BrainState.MUTED
    assert monitor.muted is True
    assert monitor.active_source is None
    assert monitor.read_frame() is None
    assert edge_sink.stopped is True

    # Mute switch released (False)
    unmute_event = HwEvent(ts=2.0, kind="mute_switch", value=False)
    monitor.handle_hw_event(unmute_event)

    assert sm.state == BrainState.IDLE
    assert monitor.muted is False
    assert monitor.active_source is edge_source
    frame = monitor.read_frame()
    assert frame is not None


def test_edge_status_and_heartbeat_timeout_fallback() -> None:
    clock_time = 100.0

    def fake_clock() -> float:
        return clock_time

    sm = BrainStateMachine(publish_fn=_dummy_publish)
    fallback_source = _create_source()
    fallback_sink = _create_sink()
    edge_source = _create_source()
    edge_sink = _create_sink()

    monitor = EdgeMonitor(
        state_machine=sm,
        fallback_source=fallback_source,
        fallback_sink=fallback_sink,
        edge_source=edge_source,
        edge_sink=edge_sink,
        heartbeat_timeout_s=15.0,
        clock=fake_clock,
    )

    # Initially edge is offline
    assert monitor.edge_online is False
    assert monitor.active_source is fallback_source
    assert monitor.active_sink is fallback_sink

    # Edge sends online status heartbeat
    online_status = HwStatus(online=True, ts=1.0, capabilities=["mic", "speaker"])
    monitor.handle_hw_status(online_status)
    assert monitor.edge_online is True
    assert monitor.active_source is edge_source
    assert monitor.active_sink is edge_sink

    # Advance clock past timeout
    clock_time = 116.0
    assert monitor.edge_online is False
    assert monitor.active_source is fallback_source
    assert monitor.active_sink is fallback_sink

    # Edge sends new online heartbeat
    clock_time = 120.0
    monitor.handle_hw_status(HwStatus(online=True, ts=2.0, capabilities=["mic", "speaker"]))
    assert monitor.edge_online is True
    assert monitor.active_source is edge_source
    assert monitor.active_sink is edge_sink

    # Edge sends offline status explicitly
    clock_time = 121.0
    offline_status = HwStatus(online=False, ts=3.0, capabilities=[])
    monitor.handle_hw_status(offline_status)
    assert monitor.edge_online is False
    assert monitor.active_source is fallback_source
    assert monitor.active_sink is fallback_sink


def test_mqtt_dispatch_and_json_payloads() -> None:
    sm = BrainStateMachine(publish_fn=_dummy_publish)
    fallback_source = _create_source()
    fallback_sink = _create_sink()
    edge_source = _create_source()
    edge_sink = _create_sink()

    monitor = EdgeMonitor(
        state_machine=sm,
        fallback_source=fallback_source,
        fallback_sink=fallback_sink,
        edge_source=edge_source,
        edge_sink=edge_sink,
    )

    # Send HW_STATUS via MQTT with JSON bytes
    status_payload = json.dumps({"online": True, "ts": 1.0, "capabilities": ["mic", "speaker"]}).encode("utf-8")
    monitor.handle_mqtt_message(HW_STATUS, status_payload)
    assert monitor.edge_online is True

    # Send HW_EVENT via MQTT with JSON string
    event_payload = json.dumps({"ts": 2.0, "kind": "mute_switch", "value": True})
    monitor.handle_mqtt_message(HW_EVENT, event_payload)
    assert monitor.muted is True
    assert sm.state == BrainState.MUTED

    # Unknown topic does nothing
    monitor.handle_mqtt_message("unknown/topic", b"{}")
    assert monitor.muted is True


def test_no_edge_devices_provided() -> None:
    sm = BrainStateMachine(publish_fn=_dummy_publish)
    fallback_source = _create_source()
    fallback_sink = _create_sink()

    monitor = EdgeMonitor(
        state_machine=sm,
        fallback_source=fallback_source,
        fallback_sink=fallback_sink,
    )

    # Online status received, but edge_source/edge_sink are None
    monitor.handle_hw_status(HwStatus(online=True, ts=1.0, capabilities=["mic", "speaker"]))
    assert monitor.edge_online is False
    assert monitor.active_source is fallback_source
    assert monitor.active_sink is fallback_sink
