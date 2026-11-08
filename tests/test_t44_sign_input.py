"""Unit tests for T4.4 sign language input path."""

from unittest.mock import MagicMock

from bmo_contracts.models import SignData, SignRecognized
from brain.inputs import SignInputHandler
from brain.settings import load_settings
from brain.state import BrainState, BrainStateMachine


def _create_handler(clock_time: float = 100.0) -> tuple[SignInputHandler, MagicMock, MagicMock]:
    settings = load_settings().policy
    state_machine = MagicMock(spec=BrainStateMachine)
    state_machine.state = BrainState.IDLE
    on_turn = MagicMock()
    clock_fn = MagicMock(return_value=clock_time)
    handler = SignInputHandler(
        settings=settings,
        state_machine=state_machine,
        on_turn=on_turn,
        clock=clock_fn,
    )
    return handler, state_machine, on_turn


def test_sign_recognized_triggers_turn() -> None:
    handler, _, on_turn = _create_handler()
    event = SignRecognized(id="1", ts=1.0, data=SignData(text="HELLO", conf=0.90))

    result = handler.handle_event(event)

    assert result == "[Sign: HELLO]"
    on_turn.assert_called_once_with("[Sign: HELLO]")


def test_sign_recognized_debounce() -> None:
    settings = load_settings().policy
    state_machine = MagicMock(spec=BrainStateMachine)
    state_machine.state = BrainState.IDLE
    on_turn = MagicMock()
    current_time = 100.0

    def clock() -> float:
        return current_time

    handler = SignInputHandler(
        settings=settings,
        state_machine=state_machine,
        on_turn=on_turn,
        clock=clock,
    )

    ev1 = SignRecognized(id="1", ts=1.0, data=SignData(text="HELLO", conf=0.90))
    assert handler.handle_event(ev1) == "[Sign: HELLO]"
    assert on_turn.call_count == 1

    # Within 2.0s (+1.0s) -> ignored
    current_time = 101.0
    ev2 = SignRecognized(id="2", ts=2.0, data=SignData(text="HELLO", conf=0.92))
    assert handler.handle_event(ev2) is None
    assert on_turn.call_count == 1

    # After 2.1s (+2.1s from first trigger) -> triggers again
    current_time = 102.1
    ev3 = SignRecognized(id="3", ts=3.1, data=SignData(text="HELLO", conf=0.92))
    assert handler.handle_event(ev3) == "[Sign: HELLO]"
    assert on_turn.call_count == 2


def test_sign_recognized_low_confidence() -> None:
    handler, _, on_turn = _create_handler()
    event = SignRecognized(id="1", ts=1.0, data=SignData(text="HELLO", conf=0.50))

    result = handler.handle_event(event)

    assert result is None
    on_turn.assert_not_called()


def test_sign_recognized_non_idle_states() -> None:
    for non_idle_state in (
        BrainState.LISTENING,
        BrainState.THINKING,
        BrainState.SPEAKING,
        BrainState.MUTED,
    ):
        handler, state_machine, on_turn = _create_handler()
        state_machine.state = non_idle_state
        event = SignRecognized(id="1", ts=1.0, data=SignData(text="HELLO", conf=0.95))

        result = handler.handle_event(event)

        assert result is None, f"Expected None for state {non_idle_state}"
        on_turn.assert_not_called()


def test_sign_recognized_json_and_edge_cases() -> None:
    handler, _, on_turn = _create_handler()
    event = SignRecognized(id="1", ts=1.0, data=SignData(text="HELLO", conf=0.90))
    json_bytes = event.model_dump_json().encode("utf-8")
    assert handler.handle_event(json_bytes) == "[Sign: HELLO]"
    assert on_turn.call_count == 1

    # Empty text
    empty_event = SignRecognized(id="2", ts=2.0, data=SignData(text="   ", conf=0.90))
    assert handler.handle_event(empty_event) is None
