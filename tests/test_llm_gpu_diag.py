from __future__ import annotations

import logging

import pytest

import brain.llm as llm_mod
from brain.llm import ChatChunk, LlamaCppBackend
from brain.settings import LLMSettings


def make_llamacpp_settings(n_gpu_layers: int) -> LLMSettings:
    return LLMSettings(backend="llamacpp", host="/x", n_gpu_layers=n_gpu_layers)


def test_cpu_only_build_logs_error_when_gpu_layers_requested(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_mod, "Llama", FakeLlama)
    monkeypatch.setattr(llm_mod, "llama_supports_gpu_offload", lambda: False)
    monkeypatch.setattr(llm_mod, "LLAMA_CPP_AVAILABLE", True)
    backend = LlamaCppBackend(make_llamacpp_settings(n_gpu_layers=-1))

    with caplog.at_level(logging.ERROR, logger="brain.llm"):
        backend._load()

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "CPU-only" in errors[0].message
    assert "n_gpu_layers=-1" in errors[0].message


def test_gpu_offload_supported_does_not_log_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @staticmethod
        def supports_gpu_offload() -> bool:
            return True

    monkeypatch.setattr(llm_mod, "Llama", FakeLlama)
    monkeypatch.setattr(llm_mod, "llama_supports_gpu_offload", lambda: False)
    monkeypatch.setattr(llm_mod, "LLAMA_CPP_AVAILABLE", True)
    backend = LlamaCppBackend(make_llamacpp_settings(n_gpu_layers=-1))

    with caplog.at_level(logging.INFO, logger="brain.llm"):
        backend._load()

    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_cpu_only_build_is_quiet_when_cpu_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_mod, "Llama", FakeLlama)
    monkeypatch.setattr(llm_mod, "llama_supports_gpu_offload", lambda: False)
    monkeypatch.setattr(llm_mod, "LLAMA_CPP_AVAILABLE", True)
    backend = LlamaCppBackend(make_llamacpp_settings(n_gpu_layers=0))

    with caplog.at_level(logging.INFO, logger="brain.llm"):
        backend._load()

    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_chat_timings_are_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            return iter(
                [
                    {"choices": [{"delta": {"content": "hel"}}]},
                    {"choices": [{"delta": {"content": "lo"}}]},
                ]
            )

    monkeypatch.setattr(llm_mod, "Llama", FakeLlama)
    monkeypatch.setattr(llm_mod, "llama_supports_gpu_offload", lambda: False)
    monkeypatch.setattr(llm_mod, "LLAMA_CPP_AVAILABLE", True)
    backend = LlamaCppBackend(make_llamacpp_settings(n_gpu_layers=0))

    with caplog.at_level(logging.INFO, logger="brain.llm"):
        chunks = list(backend.chat([{"role": "user", "content": "hi"}]))

    assert [c.text for c in chunks] == ["hel", "lo", ""]
    assert chunks[-1].done is True
    messages = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("TTFT" in m for m in messages)
    assert any("tok/s" in m for m in messages)


def test_chat_error_still_surfaces_as_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(llm_mod, "Llama", FakeLlama)
    monkeypatch.setattr(llm_mod, "llama_supports_gpu_offload", lambda: False)
    monkeypatch.setattr(llm_mod, "LLAMA_CPP_AVAILABLE", True)
    backend = LlamaCppBackend(make_llamacpp_settings(n_gpu_layers=0))

    chunks = list(backend.chat([{"role": "user", "content": "hi"}]))

    assert chunks == [ChatChunk(text="[LLM error: boom]", done=True)]
