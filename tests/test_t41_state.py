import orjson
import pytest
from bmo_contracts.models import BrainStatus
from bmo_contracts.topics import BRAIN_STATUS, TOPICS
from brain.state import (
    VALID_TRANSITIONS,
    BrainState,
    BrainStateMachine,
    InvalidTransitionError,
)


class PublishCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, int, bool]] = []

    def __call__(self, topic: str, payload: bytes, qos: int, retain: bool) -> None:
        self.calls.append((topic, payload, qos, retain))


def test_valid_transitions_map() -> None:
    assert BrainState.IDLE in VALID_TRANSITIONS
    assert BrainState.LISTENING in VALID_TRANSITIONS[BrainState.IDLE]
    assert BrainState.SPEAKING not in VALID_TRANSITIONS[BrainState.MUTED]


def test_last_will_payload() -> None:
    payload = BrainStateMachine.last_will_payload()
    data = orjson.loads(payload)
    status = BrainStatus.model_validate(data)
    assert status.online is False
    assert status.state == "idle"


def test_initial_state_and_publish_current() -> None:
    collector = PublishCollector()
    sm = BrainStateMachine(collector)
    assert sm.state == BrainState.IDLE
    assert len(collector.calls) == 0

    sm.publish_current()
    assert len(collector.calls) == 1
    topic, payload, qos, retain = collector.calls[0]
    assert topic == BRAIN_STATUS
    assert qos == TOPICS[BRAIN_STATUS].qos
    assert retain == TOPICS[BRAIN_STATUS].retain
    status = BrainStatus.model_validate(orjson.loads(payload))
    assert status.online is True
    assert status.state == "idle"


def test_valid_transitions_publish_retained() -> None:
    collector = PublishCollector()
    sm = BrainStateMachine(collector)

    sequence = [
        BrainState.LISTENING,
        BrainState.THINKING,
        BrainState.SPEAKING,
        BrainState.IDLE,
    ]
    for target in sequence:
        res = sm.transition(target)
        assert res == target
        assert sm.state == target
        topic, payload, qos, retain = collector.calls[-1]
        assert topic == BRAIN_STATUS
        assert qos == 1
        assert retain is True
        status = BrainStatus.model_validate(orjson.loads(payload))
        assert status.online is True
        assert status.state == target.value

    # Barge-in: SPEAKING -> LISTENING
    sm.transition(BrainState.SPEAKING)
    res = sm.transition(BrainState.LISTENING)
    assert res == BrainState.LISTENING
    assert sm.state == BrainState.LISTENING
    status = BrainStatus.model_validate(orjson.loads(collector.calls[-1][1]))
    assert status.state == "listening"


def test_muted_and_error_cycles() -> None:
    for initial in (
        BrainState.IDLE,
        BrainState.LISTENING,
        BrainState.THINKING,
        BrainState.SPEAKING,
    ):
        # * -> MUTED -> IDLE
        collector = PublishCollector()
        sm = BrainStateMachine(collector, initial_state=initial)
        sm.transition(BrainState.MUTED)
        assert sm.state == BrainState.MUTED
        sm.transition(BrainState.IDLE)
        assert sm.state == BrainState.IDLE

        # * -> ERROR -> IDLE
        collector_err = PublishCollector()
        sm_err = BrainStateMachine(collector_err, initial_state=initial)
        sm_err.transition(BrainState.ERROR)
        assert sm_err.state == BrainState.ERROR
        sm_err.transition(BrainState.IDLE)
        assert sm_err.state == BrainState.IDLE


def test_self_transition() -> None:
    collector = PublishCollector()
    sm = BrainStateMachine(collector, initial_state=BrainState.IDLE)
    res = sm.transition("idle")
    assert res == BrainState.IDLE
    assert len(collector.calls) == 1
    status = BrainStatus.model_validate(orjson.loads(collector.calls[0][1]))
    assert status.state == "idle"


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (BrainState.LISTENING, BrainState.SPEAKING),
        (BrainState.THINKING, BrainState.LISTENING),
        (BrainState.MUTED, BrainState.LISTENING),
        (BrainState.MUTED, BrainState.THINKING),
        (BrainState.MUTED, BrainState.SPEAKING),
        (BrainState.ERROR, BrainState.SPEAKING),
    ],
)
def test_invalid_transitions_raise(src: BrainState, dst: BrainState) -> None:
    collector = PublishCollector()
    sm = BrainStateMachine(collector, initial_state=src)
    call_count = len(collector.calls)

    with pytest.raises(InvalidTransitionError):
        sm.transition(dst)

    assert sm.state == src
    assert len(collector.calls) == call_count
