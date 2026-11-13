from __future__ import annotations

from collections.abc import Callable
import logging
import time

from bmo_contracts.models import Event, FaceIdentified, PersonDetected, parse_event_json
from brain.llm import LLMClient
from brain.scene import SceneCache
from brain.settings import PolicySettings
from brain.state import BrainState, BrainStateMachine
from brain.vision_client import VisionClient

logger = logging.getLogger(__name__)


class IdleLifecycleManager:
    """Manages idle lifecycle and VRAM unloading when room is empty and brain is idle."""

    def __init__(
        self,
        settings: PolicySettings,
        scene_cache: SceneCache,
        state_machine: BrainStateMachine,
        vision_client: VisionClient,
        llm_client: LLMClient,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._scene_cache = scene_cache
        self._state_machine = state_machine
        self._vision_client = vision_client
        self._llm_client = llm_client
        self._clock = clock
        self.sleeping: bool = False
        self._last_active_ts: float = self._clock()

    @property
    def _last_activity(self) -> float:
        return self._last_active_ts

    @_last_activity.setter
    def _last_activity(self, value: float) -> None:
        self._last_active_ts = value

    def _has_people(self) -> bool:
        scene = self._scene_cache.current()
        return bool(scene and len(scene.people) > 0)

    def notify_activity(self) -> bool:
        self._last_active_ts = self._clock()
        if self.sleeping:
            self.sleeping = False
            try:
                self._vision_client.set_modules(face=True)
            except Exception:
                logger.exception("Failed to enable face module on wake")
            try:
                self._llm_client.prewarm()
            except Exception:
                logger.exception("Failed to prewarm LLM client on wake")
            return True
        return False

    def handle_event(self, event: Event | bytes | str) -> bool:
        if isinstance(event, (bytes, str)):
            event = parse_event_json(event)
        if isinstance(event, (PersonDetected, FaceIdentified)):
            return self.notify_activity()
        return False

    def tick(self) -> bool:
        now = self._clock()
        if self._state_machine.state != BrainState.IDLE or self._has_people():
            self.notify_activity()
            return False
        if not self.sleeping and (now - self._last_active_ts) >= self._settings.idle_unload_s:
            self.sleeping = True
            try:
                self._vision_client.set_modules(face=False)
            except Exception:
                logger.exception("Failed to disable face module on unload")
            try:
                self._llm_client.unload()
            except Exception:
                logger.exception("Failed to unload LLM client on idle")
            return True
        return False
