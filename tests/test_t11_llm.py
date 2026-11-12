from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
import httpx
from pydantic import ValidationError
import pytest
from brain.llm import (
    ChatChunk,
    FakeLLMBackend,
    LLMClient,
    OllamaBackend,
    ToolCall,
    count_tokens,
)
from brain.settings import LLMSettings, Settings, load_settings


def make_test_settings() -> LLMSettings:
    return LLMSettings(
        model="qwen2.5:3b",
        host="http://127.0.0.1:11434",
        num_ctx=4096,
        keep_alive="5m",
        timeout_s=10.0,
        temperature=0.2,
    )


def test_load_settings_loads_llm_yaml() -> None:
    settings = load_settings("config/settings.d")
    assert isinstance(settings, Settings)
    assert settings.llm.backend == "llamacpp"
    assert settings.llm.model == "Qwen3.5-4B-Q4_K_M.gguf"
    assert settings.llm.host == "models/Qwen3.5-4B-Q4_K_M.gguf"
    assert settings.llm.num_ctx == 4096
    assert settings.llm.keep_alive == "5m"
    assert settings.llm.timeout_s == 30.0
    assert settings.llm.temperature == 0.7
    assert settings.llm.n_gpu_layers == -1
    assert settings.llm.use_flash_attn is False
    assert settings.llm.n_threads == 8


def test_settings_is_frozen_and_rejects_mutation() -> None:
    settings = load_settings("config/settings.d")
    replacement = make_test_settings()
    with pytest.raises(ValidationError):
        settings.llm = replacement  # type: ignore[misc]

    with pytest.raises(ValidationError):
        settings.llm.model = "different-model"  # type: ignore[misc]


def test_load_settings_duplicate_top_level_key_raises_error() -> None:
    with tempfile.TemporaryDirectory() as sandbox_dir:
        sandbox_path = Path(sandbox_dir)
        (sandbox_path / "01_first.yaml").write_text(
            "llm:\n  model: qwen2.5:3b\n  host: http://localhost:11434\n"
            "  num_ctx: 4096\n  keep_alive: 5m\n  timeout_s: 10.0\n  temperature: 0.2\n",
            encoding="utf-8",
        )
        (sandbox_path / "02_second.yaml").write_text(
            "llm:\n  model: another:7b\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate top-level key 'llm'"):
            load_settings(sandbox_path)


def test_load_settings_non_mapping_or_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_settings("non_existent_settings_dir")

    with tempfile.TemporaryDirectory() as sandbox_dir:
        sandbox_path = Path(sandbox_dir)
        (sandbox_path / "scalar.yaml").write_text("scalar_string", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level mapping"):
            load_settings(sandbox_path)


def test_count_tokens_estimator() -> None:
    assert count_tokens("") == 0
    assert count_tokens("a") == 1
    assert count_tokens("abcd") == 1
    assert count_tokens("abcde") == 2
    assert count_tokens("Hello, world!") == 4


def test_llm_client_chat_streaming_chunks() -> None:
    settings = make_test_settings()
    backend = FakeLLMBackend(chunks=["Hello", " world", "!"])
    client = LLMClient(settings, backend=backend)

    chunks = list(client.chat([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["Hello", " world", "!"]
    assert [c.done for c in chunks] == [False, False, True]
    assert len(backend.calls) == 1
    assert backend.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_client_chat_emitting_tool_call() -> None:
    settings = make_test_settings()
    tc = ToolCall(name="query_camera", arguments={"lens": "wide"})
    backend = FakeLLMBackend(chunks=[ChatChunk(text="", tool_call=tc, done=True)])
    client = LLMClient(settings, backend=backend)

    chunks = list(
        client.chat(
            [{"role": "user", "content": "look"}],
            tools=[{"type": "function", "function": {"name": "query_camera"}}],
        )
    )
    assert len(chunks) == 1
    assert chunks[0].tool_call == tc
    assert chunks[0].tool_call.name == "query_camera"
    assert chunks[0].tool_call.arguments == {"lens": "wide"}
    assert chunks[0].done is True


def test_llm_client_chat_timeout_handling() -> None:
    settings = make_test_settings()
    backend = FakeLLMBackend(raise_timeout=True)
    client = LLMClient(settings, backend=backend)

    with pytest.raises(TimeoutError, match="Simulated LLM timeout"):
        list(client.chat([{"role": "user", "content": "ping"}]))


def test_llm_client_cancellation_timing() -> None:
    settings = make_test_settings()
    backend = FakeLLMBackend(chunks=["tok"] * 20, chunk_delay=0.03)
    client = LLMClient(settings, backend=backend)

    cancel_event = threading.Event()
    timer = threading.Timer(0.04, cancel_event.set)
    timer.start()

    start_time = time.monotonic()
    chunks = list(
        client.chat(
            [{"role": "user", "content": "stream"}],
            cancel_event=cancel_event,
        )
    )
    elapsed = time.monotonic() - start_time
    timer.cancel()

    assert elapsed < 0.200
    assert len(chunks) < 20

    # Immediate cancellation check
    cancel_event.set()
    assert (
        list(
            client.chat(
                [{"role": "user", "content": "stream"}],
                cancel_event=cancel_event,
            )
        )
        == []
    )


def test_ollama_backend_mock_transport_chat_streaming() -> None:
    settings = make_test_settings()
    captured_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append(body)

        def generate_lines():
            line1 = json.dumps({"message": {"role": "assistant", "content": "part1"}, "done": False})
            line2 = json.dumps({"message": {"role": "assistant", "content": "part2"}, "done": True})
            yield (line1 + "\n").encode("utf-8")
            yield (line2 + "\n").encode("utf-8")

        return httpx.Response(200, content=generate_lines())

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)

    chunks = list(
        backend.chat(
            [{"role": "user", "content": "hello"}],
            tools=[{"name": "t"}],
            stream=True,
        )
    )
    assert len(chunks) == 2
    assert chunks[0].text == "part1"
    assert chunks[0].done is False
    assert chunks[1].text == "part2"
    assert chunks[1].done is True

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["model"] == "qwen2.5:3b"
    assert req["options"]["num_ctx"] == 4096
    assert req["options"]["temperature"] == 0.2
    assert req["keep_alive"] == "5m"
    assert req["think"] is False
    assert req["tools"] == [{"name": "t"}]


def test_ollama_backend_mock_transport_tool_calls_parsing() -> None:
    settings = make_test_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        resp_data = {
            "model": "qwen2.5:3b",
            "message": {
                "role": "assistant",
                "content": "executing",
                "tool_calls": [
                    {
                        "function": {
                            "name": "say_text",
                            "arguments": '{"phrase": "hello world"}',
                        }
                    }
                ],
            },
            "done": True,
        }
        return httpx.Response(200, json=resp_data)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)

    chunks = list(
        backend.chat(
            [{"role": "user", "content": "speak"}],
            stream=False,
        )
    )
    assert len(chunks) == 1
    assert chunks[0].text == "executing"
    assert chunks[0].tool_call is not None
    assert chunks[0].tool_call.name == "say_text"
    assert chunks[0].tool_call.arguments == {"phrase": "hello world"}
    assert chunks[0].done is True


def test_ollama_backend_mock_transport_stream_cancellation() -> None:
    settings = make_test_settings()
    cancel_event = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"

        def slow_stream():
            for i in range(50):
                line = json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"w{i}",
                        },
                        "done": False,
                    }
                )
                yield (line + "\n").encode("utf-8")
                time.sleep(0.02)

        return httpx.Response(200, content=slow_stream())

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)

    timer = threading.Timer(0.035, cancel_event.set)
    timer.start()
    start_time = time.monotonic()
    chunks = list(
        backend.chat(
            [{"role": "user", "content": "cancel me"}],
            stream=True,
            cancel_event=cancel_event,
        )
    )
    elapsed = time.monotonic() - start_time
    timer.cancel()

    assert elapsed < 0.200
    assert len(chunks) < 50


def test_ollama_backend_timeout_conversion() -> None:
    settings = make_test_settings()

    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    client = httpx.Client(transport=httpx.MockTransport(timeout_handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)

    with pytest.raises(TimeoutError, match="Read timed out"):
        list(backend.chat([{"role": "user", "content": "timeout"}], stream=False))

    with pytest.raises(TimeoutError, match="Read timed out"):
        backend.unload()

    with pytest.raises(TimeoutError, match="Read timed out"):
        backend.prewarm()


def test_ollama_backend_unload_and_prewarm_payloads() -> None:
    settings = make_test_settings()
    recorded_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        recorded_requests.append(body)
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)
    wrapper = LLMClient(settings, backend=backend)

    wrapper.unload()
    wrapper.prewarm()

    assert len(recorded_requests) == 2
    assert recorded_requests[0]["model"] == "qwen2.5:3b"
    assert recorded_requests[0]["messages"] == []
    assert recorded_requests[0]["keep_alive"] == 0

    assert recorded_requests[1]["model"] == "qwen2.5:3b"
    assert recorded_requests[1]["messages"] == []
    assert recorded_requests[1]["keep_alive"] == "5m"


def test_fake_backend_unload_and_prewarm() -> None:
    backend = FakeLLMBackend()
    assert backend.unload_calls == 0
    assert backend.prewarm_calls == 0
    assert backend.unloaded is False
    assert backend.prewarmed is False

    backend.unload()
    assert backend.unload_calls == 1
    assert backend.unloaded is True
    assert backend.prewarm_calls == 0
    assert backend.prewarmed is False

    backend.prewarm()
    assert backend.unload_calls == 1
    assert backend.unloaded is True
    assert backend.prewarm_calls == 1
    assert backend.prewarmed is True

    # Wrapper delegation assertions
    client = LLMClient(settings=make_test_settings(), backend=backend)
    client.unload()
    assert backend.unload_calls == 2
    client.prewarm()
    assert backend.prewarm_calls == 2


def test_ollama_backend_malformed_stream_lines_skipped_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    settings = make_test_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"

        def generate_lines():
            line1 = json.dumps({"message": {"role": "assistant", "content": "hello "}, "done": False})
            bad_line = '{"message": {"role": "assistant", "content": incomplete json...'
            line2 = json.dumps({"message": {"role": "assistant", "content": "world!"}, "done": True})
            yield (line1 + "\n").encode("utf-8")
            yield (bad_line + "\n").encode("utf-8")
            yield b"\n"  # empty line should be safely ignored
            yield (line2 + "\n").encode("utf-8")

        return httpx.Response(200, content=generate_lines())

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)

    with caplog.at_level("WARNING"):
        chunks = list(backend.chat([{"role": "user", "content": "hi"}], stream=True))

    assert len(chunks) == 2
    assert chunks[0].text == "hello "
    assert chunks[0].done is False
    assert chunks[1].text == "world!"
    assert chunks[1].done is True

    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("Malformed JSON in LLM stream line" in r.message and "incomplete json" in r.message for r in warning_records)


def test_ollama_backend_malformed_tool_args_surfaces_tool_error(caplog: pytest.LogCaptureFixture) -> None:
    from brain.tools.registry import ToolRegistry, ToolSpec
    from pydantic import BaseModel

    class SayTextArgs(BaseModel):
        phrase: str

    settings = make_test_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        resp_data = {
            "model": "qwen2.5:3b",
            "message": {
                "role": "assistant",
                "content": "calling function",
                "tool_calls": [
                    {
                        "function": {
                            "name": "say_text",
                            "arguments": '{"phrase": unclosed_string...',
                        }
                    }
                ],
            },
            "done": True,
        }
        return httpx.Response(200, json=resp_data)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=settings.host)
    backend = OllamaBackend(settings, client=client)

    with caplog.at_level("WARNING"):
        chunks = list(backend.chat([{"role": "user", "content": "say something"}], stream=False))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "calling function"
    assert chunk.done is True
    assert chunk.tool_call is not None
    assert chunk.tool_call.name == "say_text"
    assert chunk.tool_call.error is not None
    assert "Malformed tool arguments JSON" in chunk.tool_call.error
    assert "unclosed_string" in chunk.tool_call.error
    assert "error" in chunk.tool_call.arguments

    # Verify warning log with raw excerpt
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("Malformed tool arguments JSON" in r.message and "unclosed_string" in r.message for r in warning_records)

    # Verify existing error path: dispatching to ToolRegistry yields ok=False error result (for the model)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="say_text",
            description="say phrase",
            schema=SayTextArgs,
            handler=lambda _a: {"ok": True},
        )
    )
    dispatch_res = registry.dispatch(chunk.tool_call)
    assert dispatch_res.get("ok") is False
    assert "error" in dispatch_res


def test_parse_chunk_non_dict_tool_args_logged(caplog: pytest.LogCaptureFixture) -> None:
    from brain.llm import _parse_chunk

    with caplog.at_level("WARNING"):
        chunks = list(
            _parse_chunk(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "calc",
                                    "arguments": "42",
                                }
                            }
                        ],
                    },
                    "done": True,
                }
            )
        )

    assert len(chunks) == 1
    assert chunks[0].tool_call is not None
    assert chunks[0].tool_call.name == "calc"
    assert chunks[0].tool_call.error is not None
    assert "expected JSON object" in chunks[0].tool_call.error
    assert any("Malformed tool arguments" in r.message for r in caplog.records if r.levelname == "WARNING")
