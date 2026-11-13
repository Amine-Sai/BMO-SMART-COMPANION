from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bmo_contracts.models import Emotion


class FaceMode(StrEnum):
    BOOT = "boot"
    AWAKE = "awake"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    SLEEP = "sleep"
    MUTED = "muted"
    ERROR = "error"
    OFFLINE = "offline"


_STATUS_MAP: dict[str, FaceMode] = {
    "idle": FaceMode.AWAKE,
    "listening": FaceMode.LISTENING,
    "thinking": FaceMode.THINKING,
    "speaking": FaceMode.SPEAKING,
    "muted": FaceMode.MUTED,
    "error": FaceMode.ERROR,
}


def mode_from_status(state: str, online: bool = True) -> FaceMode:
    if not online:
        return FaceMode.OFFLINE
    return _STATUS_MAP.get(str(state).strip().lower(), FaceMode.AWAKE)


@dataclass
class FaceState:
    """Tracks the face mode, last emotion, and idle-to-sleep timing."""

    mode: FaceMode = FaceMode.BOOT
    emotion: Emotion = Emotion.NEUTRAL
    sleep_after_s: float = 60.0
    last_activity: float = 0.0
    last_say: float = 0.0
    last_say_text: str = ""

    def on_status(self, state: str, online: bool, now: float) -> None:
        incoming = mode_from_status(state, online)
        if incoming == FaceMode.AWAKE and self.mode == FaceMode.SLEEP:
            return  # stay asleep until something actually happens
        if incoming in (
            FaceMode.AWAKE,
            FaceMode.LISTENING,
            FaceMode.THINKING,
            FaceMode.SPEAKING,
        ):
            self.last_activity = now
        if incoming in (FaceMode.ERROR, FaceMode.OFFLINE):
            self.mode = incoming
            return
        self.mode = incoming

    def on_say(self, emotion: Emotion, text: str, now: float) -> None:
        self.emotion = emotion
        self.last_say = now
        self.last_say_text = text
        self.last_activity = now
        # A spoken line is the strongest signal that BMO is talking; don't rely
        # on the status message arriving first.
        self.mode = FaceMode.SPEAKING

    def tick(self, now: float) -> None:
        if (
            self.mode == FaceMode.AWAKE
            and self.sleep_after_s > 0
            and now - self.last_activity >= self.sleep_after_s
        ):
            self.mode = FaceMode.SLEEP
