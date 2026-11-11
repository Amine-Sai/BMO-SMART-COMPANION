"""End-to-end text turn through TurnManager with the real llama-cpp model."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("llama_cpp", reason="llama-cpp-python is not installed")

from brain.llm import FakeLLMBackend, LLMClient, LlamaCppBackend
from brain.prompt import PromptBuilder
from brain.scene import SceneCache
from brain.settings import LLMSettings, SceneSettings
from brain.state import BrainStateMachine
from brain.tools.registry import ToolRegistry
from brain.turn import TurnManager

MODEL_PATH = Path(__file__).parents[1] / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
E2E_TTFT_BUDGET_S = 120.0


def _noop_publish(_topic: str, _payload: bytes, _qos: int = 1, _retain: bool = False) -> None:
    pass


class _TTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak_sentence(self, text: str, sink: Any, cancel_event: threading.Event | None = None) -> None:
        _ = (sink, cancel_event)
        self.spoken.append(text)

    def cancel(self) -> None:
        pass


def _manager(llm: LLMClient, prompt_builder: PromptBuilder) -> tuple[TurnManager, BrainStateMachine, _TTS]:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    tts = _TTS()
    turn = TurnManager(
        state_machine=sm,
        wake=None,  # type: ignore[arg-type]
        vad=None,  # type: ignore[arg-type]
        stt=None,  # type: ignore[arg-type]
        llm=llm,
        prompt_builder=prompt_builder,
        tools=ToolRegistry(),
        tts=tts,  # type: ignore[arg-type]
        sink=None,  # type: ignore[arg-type]
        publish_fn=_noop_publish,
    )
    return turn, sm, tts


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason=f"local model file missing: {MODEL_PATH}")
def test_real_backend_text_turn_within_budget() -> None:
    settings = LLMSettings(
        backend="llamacpp",
        host=str(MODEL_PATH),
        n_gpu_layers=0,
        use_flash_attn=False,
    )
    llm = LLMClient(settings=settings, backend=LlamaCppBackend(settings))
    builder = PromptBuilder(
        SceneCache(settings=SceneSettings()),
        max_turns=settings.history_turns,
        max_total_tokens=settings.history_max_tokens,
    )
    turn, sm, tts = _manager(llm, builder)

    try:
        started = time.perf_counter()
        reply = turn.run_text_turn("hi")
        elapsed = time.perf_counter() - started
    finally:
        llm.unload()

    print(f"\nE2E reply in {elapsed:.1f}s: {reply!r}")

    assert reply.strip(), "real backend returned an empty reply"
    assert elapsed < E2E_TTFT_BUDGET_S, f"turn took {elapsed:.1f}s (budget {E2E_TTFT_BUDGET_S}s)"
    assert tts.spoken, "reply was never handed to TTS"
    assert sm.state.value == "idle"


def test_second_concurrent_start_text_turn_is_dropped(caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    backend = FakeLLMBackend(chunks=["One. ", "Two. "], chunk_delay=0.2)
    settings = LLMSettings()
    builder = PromptBuilder(
        SceneCache(settings=SceneSettings()),
        max_turns=settings.history_turns,
        max_total_tokens=settings.history_max_tokens,
    )
    turn, _, tts = _manager(LLMClient(settings=settings, backend=backend), builder)

    turn.start_text_turn("first")
    turn.start_text_turn("second")
    worker = turn._active_thread
    assert worker is not None
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert "turn busy, dropping" in caplog.text
    assert len(backend.calls) == 1
    assert [m["messages"][-1]["content"] for m in backend.calls] == ["first"]
    assert tts.spoken == ["One.", "Two."]
