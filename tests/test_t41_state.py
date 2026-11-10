import threading

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


def test_publish_callback_invoked_outside_lock() -> None:
    sm: BrainStateMachine | None = None
    thread_saw_state: BrainState | None = None
    publish_entered = threading.Event()
    reader_done = threading.Event()

    def publish_fn(_topic: str, _payload: bytes, _qos: int, _retain: bool) -> None:
        nonlocal thread_saw_state
        publish_entered.set()
        # Wait for reader thread to read sm.state while we are inside publish_fn
        reader_done.wait(timeout=2.0)

    sm = BrainStateMachine(publish_fn, initial_state=BrainState.IDLE)

    def reader_worker() -> None:
        nonlocal thread_saw_state
        publish_entered.wait(timeout=2.0)
        # sm.state acquires sm._lock. If sm._lock was held during publish_fn, this would block/deadlock.
        thread_saw_state = sm.state
        reader_done.set()

    reader = threading.Thread(target=reader_worker)
    reader.start()

    sm.transition(BrainState.LISTENING)
    reader.join(timeout=2.0)

    assert not reader.is_alive()
    assert thread_saw_state == BrainState.LISTENING
    assert sm.state == BrainState.LISTENING


def test_publish_callback_reentrancy() -> None:
    sm: BrainStateMachine | None = None
    reentered = False

    def publish_fn(_topic: str, _payload: bytes, _qos: int, _retain: bool) -> None:
        nonlocal reentered
        if not reentered and sm is not None and sm.state == BrainState.LISTENING:
            reentered = True
            # Re-entrant transition from callback
            sm.transition(BrainState.THINKING)

    sm = BrainStateMachine(publish_fn, initial_state=BrainState.IDLE)
    sm.transition(BrainState.LISTENING)

    assert reentered is True
    assert sm.state == BrainState.THINKING


def test_state_machine_concurrency() -> None:
    num_threads = 12
    iterations = 100
    published: list[tuple[str, bytes]] = []
    publish_lock = threading.Lock()

    def publish_fn(topic: str, payload: bytes, _qos: int, _retain: bool) -> None:
        with publish_lock:
            published.append((topic, payload))

    sm = BrainStateMachine(publish_fn, initial_state=BrainState.IDLE)
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []

    all_states = list(BrainState)

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for i in range(iterations):
                if (tid + i) % 5 == 0:
                    sm.publish_current()
                else:
                    target = all_states[(tid + i) % len(all_states)]
                    try:
                        sm.transition(target)
                    except InvalidTransitionError:
                        pass
                curr = sm.state
                assert isinstance(curr, BrainState)
                assert curr in BrainState
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not any(t.is_alive() for t in threads), "Worker threads hung (possible deadlock)"
    assert not errors, f"Errors during concurrency: {errors}"
    assert sm.state in BrainState

    for topic, payload in published:
        assert topic == BRAIN_STATUS
        status = BrainStatus.model_validate(orjson.loads(payload))
        assert status.online is True
        assert status.state in [s.value for s in BrainState]
