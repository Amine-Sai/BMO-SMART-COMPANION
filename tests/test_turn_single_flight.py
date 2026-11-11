from __future__ import annotations

import logging
import threading
from typing import Any

from brain.llm import FakeLLMBackend, LLMClient
from brain.prompt import PromptBuilder
from brain.settings import LLMSettings
from brain.state import BrainState, BrainStateMachine
from brain.tools.registry import ToolRegistry
from brain.turn import TurnManager


def _noop_publish(_topic: str, _payload: bytes, _qos: int = 1, _retain: bool = False) -> None:
    pass


class _Prompt(PromptBuilder):
    def __init__(self) -> None:
        pass

    def build_messages(self, user_text: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": user_text}]

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        _ = (user_text, assistant_text)


class _TTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak_sentence(self, text: str, sink: Any, cancel_event: threading.Event | None = None) -> None:
        _ = (sink, cancel_event)
        self.spoken.append(text)

    def cancel(self) -> None:
        pass


def _manager(backend: FakeLLMBackend) -> tuple[TurnManager, BrainStateMachine, _TTS]:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    tts = _TTS()
    turn = TurnManager(
        state_machine=sm,
        wake=None,  # type: ignore[arg-type]
        vad=None,  # type: ignore[arg-type]
        stt=None,  # type: ignore[arg-type]
        llm=LLMClient(settings=LLMSettings(), backend=backend),
        prompt_builder=_Prompt(),
        tools=ToolRegistry(),
        tts=tts,  # type: ignore[arg-type]
        sink=None,  # type: ignore[arg-type]
    )
    return turn, sm, tts


def test_second_text_turn_is_dropped_not_queued(caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    backend = FakeLLMBackend(chunks=["One. ", "Two. ", "Three. "], chunk_delay=0.2)
    turn, sm, tts = _manager(backend)

    turn.start_text_turn("first")
    first = turn._active_thread
    assert first is not None and first.is_alive()

    t0 = turn._clock()
    turn.start_text_turn("second")
    elapsed = turn._clock() - t0

    assert elapsed < 0.1, f"start_text_turn blocked for {elapsed:.3f}s"
    assert turn._active_thread is first, "busy turn must not spawn a second worker"
    assert "turn busy, dropping" in caplog.text

    first.join(timeout=5.0)
    assert not first.is_alive()
    assert len(backend.calls) == 1
    assert sm.state == BrainState.IDLE
    assert [m["messages"][-1]["content"] for m in backend.calls] == ["first"]
    assert tts.spoken == ["One.", "Two.", "Three."]

    # Gate is released afterwards, so the next turn runs normally.
    assert turn.run_text_turn("third") == "One. Two. Three. "
    assert len(backend.calls) == 2
