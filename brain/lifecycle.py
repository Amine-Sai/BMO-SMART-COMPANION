from __future__ import annotations

from collections.abc import Callable
import time

from bmo_contracts.models import Event, PersonDetected, parse_event_json
from brain.llm import LLMClient
from brain.scene import SceneCache
from brain.settings import PolicySettings
from brain.state import BrainState, BrainStateMachine
from brain.vision_client import VisionClient


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

    def _has_people(self) -> bool:
        scene = self._scene_cache.current()
        return bool(scene and len(scene.people) > 0)

    def notify_activity(self) -> bool:
        self._last_active_ts = self._clock()
        if self.sleeping:
            self.sleeping = False
            self._vision_client.set_modules(face=True, sign=True)
            self._llm_client.prewarm()
            return True
        return False

    def handle_event(self, event: Event | bytes | str) -> bool:
        if isinstance(event, (bytes, str)):
            event = parse_event_json(event)
        if isinstance(event, PersonDetected):
            return self.notify_activity()
        return False

    def tick(self) -> bool:
        now = self._clock()
        if self._state_machine.state != BrainState.IDLE or self._has_people():
            self.notify_activity()
            return False
        if not self.sleeping and (now - self._last_active_ts) >= self._settings.idle_unload_s:
            self.sleeping = True
            self._vision_client.set_modules(face=False, sign=False)
            self._llm_client.unload()
            return True
        return False
