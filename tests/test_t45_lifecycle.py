from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
from paho.mqtt.client import Client

from bmo_contracts.models import FaceData, FaceIdentified, PersonData, PersonDetected
from brain.lifecycle import IdleLifecycleManager
from brain.llm import LLMClient
from brain.main import BrainService
from brain.scene import SceneCache
from brain.settings import load_settings
from brain.state import BrainState, BrainStateMachine, InvalidTransitionError
from brain.vision_client import VisionClient


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, delta: float) -> None:
        self.time += delta


def _create_manager(
    clock: FakeClock, idle_unload_s: float = 300.0
) -> tuple[
    IdleLifecycleManager,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    settings = load_settings().policy.model_copy(update={"idle_unload_s": idle_unload_s})
    scene_cache = MagicMock(spec=SceneCache)
    empty_scene = MagicMock()
    empty_scene.people = []
    scene_cache.current.return_value = empty_scene

    state_machine = MagicMock(spec=BrainStateMachine)
    state_machine.state = BrainState.IDLE

    vision_client = MagicMock(spec=VisionClient)
    llm_client = MagicMock(spec=LLMClient)

    manager = IdleLifecycleManager(
        settings=settings,
        scene_cache=scene_cache,
        state_machine=state_machine,
        vision_client=vision_client,
        llm_client=llm_client,
        clock=clock,
    )
    return manager, scene_cache, state_machine, vision_client, llm_client


def test_tick_unloads_vram_when_idle_timeout_reached() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock, idle_unload_s=300.0)

    # Initial tick at t=0
    assert not manager.tick()
    assert not manager.sleeping
    vision_client.set_modules.assert_not_called()
    llm_client.unload.assert_not_called()

    # Advance fake clock +301.0 s (> idle_unload_s=300.0)
    clock.advance(301.0)
    unloaded = manager.tick()

    assert unloaded is True
    assert manager.sleeping is True
    vision_client.set_modules.assert_called_once_with(face=False)
    llm_client.unload.assert_called_once()

    # Subsequent tick while already sleeping does not re-unload
    assert not manager.tick()
    assert vision_client.set_modules.call_count == 1
    assert llm_client.unload.call_count == 1


def test_person_detected_wakes_manager_and_prewarms() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock, idle_unload_s=300.0)

    # Trigger unload
    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    # PersonDetected event received
    clock.advance(1.0)
    event = PersonDetected(
        id="e1",
        ts=clock(),
        data=PersonData(track_id=1, pos="center", proximity="near"),
    )
    woken = manager.handle_event(event)

    assert woken is True
    assert manager.sleeping is False
    vision_client.set_modules.assert_called_with(face=True)
    llm_client.prewarm.assert_called_once()


def test_person_detected_json_str_and_bytes_wakes_manager() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock)

    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    event = PersonDetected(
        id="e1",
        ts=clock(),
        data=PersonData(track_id=1, pos="center", proximity="near"),
    )
    raw_json = event.model_dump_json()

    # Handle string event
    assert manager.handle_event(raw_json) is True
    assert manager.sleeping is False

    # Sleep again
    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    # Handle bytes event
    raw_bytes = raw_json.encode("utf-8")
    assert manager.handle_event(raw_bytes) is True
    assert manager.sleeping is False


def test_unrelated_event_does_not_wake_manager() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, _, llm_client = _create_manager(clock)

    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    other_event = MagicMock()
    assert manager.handle_event(other_event) is False
    assert manager.sleeping is True
    llm_client.prewarm.assert_not_called()


def test_notify_activity_wakes_sleeping_manager() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock)

    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    # Direct activity notification (e.g. wake word)
    woken = manager.notify_activity()
    assert woken is True
    assert manager.sleeping is False
    vision_client.set_modules.assert_called_with(face=True)
    llm_client.prewarm.assert_called_once()

    # Notifying when already awake returns False
    assert manager.notify_activity() is False


def test_no_unload_when_people_present_in_scene() -> None:
    clock = FakeClock(1000.0)
    manager, scene_cache, _, vision_client, llm_client = _create_manager(clock)

    busy_scene = MagicMock()
    busy_scene.people = [PersonData(track_id=1, pos="center", proximity="near")]
    scene_cache.current.return_value = busy_scene

    clock.advance(301.0)
    assert not manager.tick()
    assert manager.sleeping is False
    vision_client.set_modules.assert_not_called()
    llm_client.unload.assert_not_called()


def test_no_unload_when_state_machine_not_idle() -> None:
    clock = FakeClock(1000.0)
    manager, _, state_machine, vision_client, llm_client = _create_manager(clock)

    non_idle_state = next(s for s in BrainState if s != BrainState.IDLE)
    state_machine.state = non_idle_state

    clock.advance(301.0)
    assert not manager.tick()
    assert manager.sleeping is False
    vision_client.set_modules.assert_not_called()
    llm_client.unload.assert_not_called()


def test_tick_when_scene_is_none() -> None:
    clock = FakeClock(1000.0)
    manager, scene_cache, _, vision_client, llm_client = _create_manager(clock)

    scene_cache.current.return_value = None
    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True
    vision_client.set_modules.assert_called_once_with(face=False)
    llm_client.unload.assert_called_once()


def test_face_identified_wakes_manager_and_prewarms() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock, idle_unload_s=300.0)

    # Trigger unload
    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    # FaceIdentified event received
    clock.advance(1.0)
    event = FaceIdentified(
        id="f1",
        ts=clock(),
        data=FaceData(track_id=1, name="Judas", conf=0.95),
    )
    woken = manager.handle_event(event)

    assert woken is True
    assert manager.sleeping is False
    vision_client.set_modules.assert_called_with(face=True)
    llm_client.prewarm.assert_called_once()


def test_face_identified_json_str_and_bytes_wakes_manager() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock)

    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    event = FaceIdentified(
        id="f1",
        ts=clock(),
        data=FaceData(track_id=1, name="Judas", conf=0.95),
    )
    raw_json = event.model_dump_json()

    # Handle string event
    assert manager.handle_event(raw_json) is True
    assert manager.sleeping is False

    # Sleep again
    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    # Handle bytes event
    raw_bytes = raw_json.encode("utf-8")
    assert manager.handle_event(raw_bytes) is True
    assert manager.sleeping is False


def test_prewarm_exception_does_not_leave_manager_stuck_sleeping() -> None:
    clock = FakeClock(1000.0)
    manager, _, _, vision_client, llm_client = _create_manager(clock)

    clock.advance(301.0)
    assert manager.tick() is True
    assert manager.sleeping is True

    llm_client.prewarm.side_effect = RuntimeError("LLM prewarm glitch")

    woken = manager.notify_activity()
    assert woken is True
    assert manager.sleeping is False
    vision_client.set_modules.assert_called_with(face=True)
    llm_client.prewarm.assert_called_once()


def test_brain_service_tick_thread_lifecycle() -> None:
    mock_mqtt = MagicMock(spec=Client)
    base_settings = load_settings()
    custom_policy = base_settings.policy.model_copy(update={"tick_interval_s": 0.01})
    settings = base_settings.model_copy(update={"policy": custom_policy})

    service = BrainService(settings=settings, mqtt_client=mock_mqtt)
    service.policy = MagicMock()
    service.lifecycle = MagicMock()

    service.start(run_audio_loop=False, run_tick_loop=True)
    assert service._tick_thread is not None
    assert service._tick_thread.is_alive()

    # Wait briefly for tick loop to execute ticks
    time.sleep(0.05)
    assert service.policy.tick.call_count >= 1
    assert service.lifecycle.tick.call_count >= 1

    service.stop()
    assert service._tick_thread is None
    assert service._stop_event.is_set()


def test_brain_service_audio_loop_skips_during_listening() -> None:
    mock_mqtt = MagicMock(spec=Client)
    service = BrainService(mqtt_client=mock_mqtt)

    service.edge_monitor = MagicMock()
    service.turn_manager = MagicMock()
    service.lifecycle = MagicMock()

    # Transition state machine to LISTENING
    service.state_machine.transition(BrainState.LISTENING)

    service._running = True
    service._stop_event.clear()
    thread = threading.Thread(target=service._audio_loop, daemon=True)
    thread.start()

    time.sleep(0.05)
    service._stop_event.set()
    thread.join(timeout=1.0)

    service.edge_monitor.read_frame.assert_not_called()
    service.turn_manager.process_audio_frame.assert_not_called()


def test_brain_service_audio_loop_catches_invalid_transition() -> None:
    mock_mqtt = MagicMock(spec=Client)
    service = BrainService(mqtt_client=mock_mqtt)

    dummy_frame = np.zeros(320, dtype=np.int16)
    service.edge_monitor = MagicMock()
    service.edge_monitor.read_frame.return_value = dummy_frame
    service.edge_monitor.active_source = None

    service.turn_manager = MagicMock()
    call_count = 0

    def fake_process_frame(*_args: object, **_kwargs: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise InvalidTransitionError("Invalid transition")
        return "listening"

    service.turn_manager.process_audio_frame.side_effect = fake_process_frame
    service.lifecycle = MagicMock()

    service._stop_event.clear()
    thread = threading.Thread(target=service._audio_loop, daemon=True)
    thread.start()

    time.sleep(0.06)
    service._stop_event.set()
    thread.join(timeout=1.0)

    assert service.turn_manager.process_audio_frame.call_count >= 2
    service.lifecycle.notify_activity.assert_called()


def test_brain_service_on_proactive_greet_notifies_activity() -> None:
    mock_mqtt = MagicMock(spec=Client)
    service = BrainService(mqtt_client=mock_mqtt)
    service.turn_manager = MagicMock()
    service.lifecycle = MagicMock()

    service._on_proactive_greet("Hello BMO!")
    service.lifecycle.notify_activity.assert_called_once()
    service.turn_manager.start_text_turn.assert_called_once_with("Hello BMO!")
