from collections.abc import Callable
from enum import StrEnum
import threading

import orjson
from bmo_contracts.models import BrainStatus
from bmo_contracts.topics import BRAIN_STATUS, TOPICS


class BrainState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    MUTED = "muted"
    ERROR = "error"


class InvalidTransitionError(RuntimeError):
    pass


VALID_TRANSITIONS: dict[BrainState, frozenset[BrainState]] = {
    BrainState.IDLE: frozenset(
        {
            BrainState.LISTENING,
            BrainState.THINKING,
            BrainState.SPEAKING,
            BrainState.MUTED,
            BrainState.ERROR,
        }
    ),
    BrainState.LISTENING: frozenset(
        {
            BrainState.THINKING,
            BrainState.IDLE,
            BrainState.MUTED,
            BrainState.ERROR,
        }
    ),
    BrainState.THINKING: frozenset(
        {
            BrainState.SPEAKING,
            BrainState.IDLE,
            BrainState.MUTED,
            BrainState.ERROR,
        }
    ),
    BrainState.SPEAKING: frozenset(
        {
            BrainState.IDLE,
            BrainState.LISTENING,
            BrainState.MUTED,
            BrainState.ERROR,
        }
    ),
    BrainState.MUTED: frozenset({BrainState.IDLE, BrainState.ERROR}),
    BrainState.ERROR: frozenset({BrainState.IDLE, BrainState.MUTED}),
}


class BrainStateMachine:
    def __init__(
        self,
        publish_fn: Callable[[str, bytes, int, bool], None],
        initial_state: BrainState = BrainState.IDLE,
    ) -> None:
        self._publish_fn = publish_fn
        self._lock = threading.RLock()
        self._state = BrainState(initial_state)

    @property
    def state(self) -> BrainState:
        with self._lock:
            return self._state

    def _build_status_payload(self, state: BrainState) -> tuple[bytes, int, bool]:
        status = BrainStatus(online=True, state=state.value)
        payload = orjson.dumps(status.model_dump(mode="json"))
        meta = TOPICS[BRAIN_STATUS]
        return payload, meta.qos, meta.retain

    def publish_current(self) -> None:
        with self._lock:
            payload, qos, retain = self._build_status_payload(self._state)
        self._publish_fn(BRAIN_STATUS, payload, qos, retain)

    def transition(self, target: BrainState | str) -> BrainState:
        target_state = BrainState(target)
        with self._lock:
            if target_state == self._state:
                current_state = self._state
            elif target_state not in VALID_TRANSITIONS[self._state]:
                raise InvalidTransitionError(f"Invalid transition: {self._state} -> {target_state}")
            else:
                self._state = target_state
                current_state = self._state
            payload, qos, retain = self._build_status_payload(current_state)

        self._publish_fn(BRAIN_STATUS, payload, qos, retain)
        return current_state

    @staticmethod
    def last_will_payload() -> bytes:
        return orjson.dumps(BrainStatus(online=False, state="idle").model_dump(mode="json"))
