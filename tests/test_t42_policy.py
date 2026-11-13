"""Tests for proactive event policy (T4.2)."""

from __future__ import annotations

from typing import Any
import pytest

from bmo_contracts.models import (
    FaceData,
    FaceIdentified,
    PersonDetected,
    PersonLeft,
)
from brain.policy import ProactivePolicy
from brain.settings import PolicySettings, load_settings
from brain.state import BrainState, BrainStateMachine


class FakeClock:
    """Deterministic monotonic clock for policy testing."""

    def __init__(self, initial: float = 100.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, delta_s: float) -> None:
        self.now += delta_s


@pytest.fixture
def default_policy_settings() -> PolicySettings:
    return PolicySettings(
        face_min_conf=0.65,
        greeting_cooldown_s=900.0,
        global_cooldown_s=30.0,
        walk_by_window_s=1.5,
        idle_unload_s=300.0,
    )


class StubStateMachine:
    def __init__(self, state: BrainState = BrainState.IDLE) -> None:
        self.state = state


def make_person_detected_data(track_id: int) -> Any:
    annotation = PersonDetected.model_fields["data"].annotation
    for kwargs in [
        {"track_id": track_id, "pos": "center", "proximity": "near"},
        {"track_id": track_id},
    ]:
        try:
            return annotation(**kwargs)
        except Exception:
            pass
    return {"track_id": track_id, "pos": "center", "proximity": "near"}


def make_person_left_data(track_id: int) -> Any:
    annotation = PersonLeft.model_fields["data"].annotation
    for kwargs in [
        {"track_id": track_id},
        {"track_id": track_id, "pos": "center", "proximity": "near"},
    ]:
        try:
            return annotation(**kwargs)
        except Exception:
            pass
    return {"track_id": track_id}


def test_rapid_face_identifications_single_greeting(default_policy_settings: PolicySettings) -> None:
    sm = BrainStateMachine(lambda *_: None)
    clock = FakeClock(initial=100.0)
    greetings: list[str] = []
    policy = ProactivePolicy(default_policy_settings, sm, greetings.append, clock=clock)

    event = FaceIdentified(id="evt1", ts=100.0, data=FaceData(name="Jud", conf=0.88))

    for _ in range(50):
        policy.handle_event(event)

    assert greetings == ["Hi Jud!"]

    # Advance clock by +901.0s and verify second greeting is allowed
    clock.advance(901.0)
    policy.handle_event(event)
    assert greetings == ["Hi Jud!", "Hi Jud!"]


def test_walk_by_suppression(default_policy_settings: PolicySettings) -> None:
    sm = BrainStateMachine(lambda *_: None)
    clock = FakeClock(initial=10.0)
    greetings: list[str] = []
    policy = ProactivePolicy(default_policy_settings, sm, greetings.append, clock=clock)

    # PersonDetected at t=10.0
    policy.handle_event(PersonDetected(id="p_in", ts=10.0, data=make_person_detected_data(1)))

    # PersonLeft at t=10.8
    clock.advance(0.8)
    policy.handle_event(PersonLeft(id="p_out", ts=10.8, data=make_person_left_data(1)))

    # tick at t=12.0
    clock.advance(1.2)
    result = policy.tick()

    assert result is None
    assert greetings == []


def test_non_idle_suppression(default_policy_settings: PolicySettings) -> None:
    clock = FakeClock(initial=100.0)
    greetings: list[str] = []
    sm = StubStateMachine(state=BrainState.IDLE)
    policy = ProactivePolicy(default_policy_settings, sm, greetings.append, clock=clock)  # type: ignore[arg-type]

    non_idle_states = [
        BrainState.LISTENING,
        BrainState.THINKING,
        BrainState.SPEAKING,
        BrainState.MUTED,
        BrainState.ERROR,
    ]

    event = FaceIdentified(id="evt_ni", ts=100.0, data=FaceData(name="Jud", conf=0.88))
    for state in non_idle_states:
        sm.state = state
        assert policy.handle_event(event) is None

    assert greetings == []


def test_walk_by_person_present_triggers_greeting(default_policy_settings: PolicySettings) -> None:
    sm = BrainStateMachine(lambda *_: None)
    clock = FakeClock(initial=10.0)
    greetings: list[str] = []
    policy = ProactivePolicy(default_policy_settings, sm, greetings.append, clock=clock)

    policy.handle_event(PersonDetected(id="p1", ts=10.0, data=make_person_detected_data(1)))
    clock.advance(1.5)
    result = policy.tick()

    assert result == "Hello there!"
    assert greetings == ["Hello there!"]


def test_low_confidence_and_generic_names_ignored(default_policy_settings: PolicySettings) -> None:
    sm = BrainStateMachine(lambda *_: None)
    clock = FakeClock(initial=100.0)
    greetings: list[str] = []
    policy = ProactivePolicy(default_policy_settings, sm, greetings.append, clock=clock)

    # Low confidence
    policy.handle_event(FaceIdentified(id="e1", ts=100.0, data=FaceData(name="Jud", conf=0.50)))
    # Generic names
    policy.handle_event(FaceIdentified(id="e2", ts=100.0, data=FaceData(name="unknown", conf=0.90)))
    policy.handle_event(FaceIdentified(id="e3", ts=100.0, data=FaceData(name="person", conf=0.90)))
    policy.handle_event(FaceIdentified(id="e4", ts=100.0, data=FaceData(name="", conf=0.90)))

    assert greetings == []


def test_settings_load_policy() -> None:
    settings = load_settings("config/settings.d")
    assert settings.policy is not None
    assert settings.policy.face_min_conf == 0.65
    assert settings.policy.greeting_cooldown_s == 900.0
    assert settings.policy.global_cooldown_s == 30.0
    assert settings.policy.walk_by_window_s == 1.5
