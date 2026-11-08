from collections.abc import Callable
from enum import StrEnum

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
        self._state = BrainState(initial_state)

    @property
    def state(self) -> BrainState:
        return self._state

    def publish_current(self) -> None:
        status = BrainStatus(online=True, state=self._state.value)
        payload = orjson.dumps(status.model_dump(mode="json"))
        meta = TOPICS[BRAIN_STATUS]
        self._publish_fn(BRAIN_STATUS, payload, meta.qos, meta.retain)

    def transition(self, target: BrainState | str) -> BrainState:
        target_state = BrainState(target)
        if target_state == self._state:
            self.publish_current()
            return self._state
        if target_state not in VALID_TRANSITIONS[self._state]:
            raise InvalidTransitionError(f"Invalid transition: {self._state} -> {target_state}")
        self._state = target_state
        self.publish_current()
        return self._state

    @staticmethod
    def last_will_payload() -> bytes:
        return orjson.dumps(BrainStatus(online=False, state="idle").model_dump(mode="json"))
