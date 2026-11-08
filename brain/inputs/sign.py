"""Sign-language input handler for Brain."""

from collections.abc import Callable
import time

from bmo_contracts.models import Event, SignRecognized, parse_event_json
from brain.settings import PolicySettings
from brain.state import BrainState, BrainStateMachine


class SignInputHandler:
    """Handles sign recognition events and injects them as conversation turns."""

    def __init__(
        self,
        settings: PolicySettings,
        state_machine: BrainStateMachine,
        on_turn: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._state_machine = state_machine
        self._on_turn = on_turn
        self._clock = clock
        self._last_sign: str = ""
        self._last_ts: float = -float("inf")

    def handle_event(self, event: Event | bytes | str) -> str | None:
        """Process an incoming event and return the turn text if handled."""
        if isinstance(event, (bytes, str)):
            event = parse_event_json(event)

        if not isinstance(event, SignRecognized):
            return None

        if self._state_machine.state != BrainState.IDLE:
            return None

        text = event.data.text.strip()
        if not text or event.data.conf < self._settings.sign_min_conf:
            return None

        now = self._clock()
        if self._last_sign == text.upper() and (now - self._last_ts) < self._settings.sign_debounce_s:
            return None

        self._last_sign = text.upper()
        self._last_ts = now
        turn_text = f"[Sign: {text}]"
        self._on_turn(turn_text)
        return turn_text
