from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any, Iterator, Mapping, Protocol, Sequence
import httpx
from brain.settings import LLMSettings

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger(__name__)

MsgList, CancelEvt = Sequence[Mapping[str, Any]], threading.Event | None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class ChatChunk:
    text: str = ""
    tool_call: ToolCall | None = None
    done: bool = False


def count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


class LLMBackend(Protocol):
    def chat(
        self,
        messages: MsgList,
        tools: MsgList | None = None,
        stream: bool = True,
        cancel_event: CancelEvt = None,
    ) -> Iterator[ChatChunk]: ...
    def unload(self) -> None: ...
    def prewarm(self) -> None: ...


def _parse_chunk(data: dict[str, Any]) -> Iterator[ChatChunk]:
    if not isinstance(data, dict):
        logger.warning("Malformed chunk data (expected dict, got %s): %r", type(data).__name__, data)
        return
    msg = data.get("message")
    if not isinstance(msg, dict):
        msg = {}
    done = bool(data.get("done", False))
    content = msg.get("content", "") or ""
    raw_calls = msg.get("tool_calls") or []
    calls = raw_calls if isinstance(raw_calls, list) else []
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "")
        raw = fn.get("arguments", {})
        error: str | None = None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    error = f"Malformed tool arguments (expected JSON object, got {type(parsed).__name__}): {raw[:200]!r}"
                    logger.warning(error)
                    args = {"error": error}
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"Malformed tool arguments JSON: {raw[:200]!r} ({exc})"
                logger.warning(error)
                args = {"error": error}
        elif isinstance(raw, dict):
            args = raw
        else:
            error = f"Malformed tool arguments type (expected str or dict, got {type(raw).__name__}): {raw!r}"
            logger.warning(error)
            args = {"error": error}
        yield ChatChunk(content, ToolCall(name, args, error=error), done)
        content = ""
    if content or (done and not calls):
        yield ChatChunk(content, done=done)


class OllamaBackend:
    def __init__(self, s: LLMSettings, client: httpx.Client | None = None) -> None:
        self._s = s
        self._client = client or httpx.Client(base_url=s.host, timeout=s.timeout_s)

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post("/api/chat", json=payload).raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc

    def chat(
        self,
        messages: MsgList,
        tools: MsgList | None = None,
        stream: bool = True,
        cancel_event: CancelEvt = None,
    ) -> Iterator[ChatChunk]:
        if cancel_event and cancel_event.is_set():
            return
        s = self._s
        p: dict[str, Any] = {
            "model": s.model,
            "messages": list(messages),
            "stream": stream,
            "think": False,
            "options": {"num_ctx": s.num_ctx, "temperature": s.temperature},
            "keep_alive": s.keep_alive,
        }
        if tools is not None:
            p["tools"] = list(tools)
        if not stream:
            try:
                data = self._post(p).json()
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Malformed JSON in non-stream response: %s", exc)
                return
            yield from _parse_chunk(data)
            return
        try:
            with self._client.stream("POST", "/api/chat", json=p) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning(
                            "Malformed JSON in LLM stream line: %r (%s)",
                            line[:200],
                            exc,
                        )
                        continue
                    for c in _parse_chunk(data):
                        if cancel_event and cancel_event.is_set():
                            return
                        yield c
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc

    def unload(self) -> None:
        self._post({"model": self._s.model, "messages": [], "keep_alive": 0})

    def prewarm(self) -> None:
        self._post({"model": self._s.model, "messages": [], "keep_alive": self._s.keep_alive})


class LlamaCppBackend:
    """Local LLM backend using llama-cpp-python (no HTTP server needed)."""

    def __init__(self, s: LLMSettings) -> None:
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        self._s = s
        self._llm: Llama | None = None
        self._model_path = s.host  # reuse host field for model path

    def _load(self) -> Llama:
        if self._llm is None:
            # host field contains the model path for local backend
            started = time.perf_counter()
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._s.num_ctx,
                n_threads=self._s.n_threads,
                n_gpu_layers=self._s.n_gpu_layers,
                flash_attn=self._s.use_flash_attn,
                use_mmap=self._s.use_mmap,
                use_mlock=self._s.use_mlock,
                verbose=False,
            )
            logger.info(
                "llama-cpp model loaded in %.2fs: %s",
                time.perf_counter() - started,
                self._model_path,
            )
            self._check_gpu_offload()
        return self._llm

    def _check_gpu_offload(self) -> None:
        """Warn loudly when n_gpu_layers was requested but this build has no CUDA."""
        # instance lookup also resolves the newer static method; absent in old builds
        probe = getattr(self._llm, "supports_gpu_offload", None)
        try:
            supported = bool(probe()) if callable(probe) else False
        except Exception as exc:  # capability probe must never break loading
            logger.warning("llama-cpp GPU offload probe failed: %s", exc)
            supported = False
        if self._s.n_gpu_layers != 0 and not supported:
            logger.error(
                "llama-cpp-python has no GPU offload support (CPU-only build): "
                "requested n_gpu_layers=%d was ignored and the model will run on CPU. "
                'Fix: rebuild with CUDA, e.g. CMAKE_ARGS="-DGGML_CUDA=on" '
                "pip install --force-reinstall --no-cache-dir llama-cpp-python",
                self._s.n_gpu_layers,
            )

    @staticmethod
    def _log_chat_timings(started: float, first_token_at: float | None, tokens: int) -> None:
        end = time.perf_counter()
        if first_token_at is None:
            logger.info("llama-cpp completion %.2fs (%d tok)", end - started, tokens)
            return
        logger.info("llama-cpp TTFT %.2fs", first_token_at - started)
        decode_s = end - first_token_at
        if tokens and decode_s > 0:
            logger.info(
                "llama-cpp %.1f tok/s (%d tok in %.2fs)", tokens / decode_s, tokens, decode_s
            )

    def chat(
        self,
        messages: MsgList,
        tools: MsgList | None = None,
        stream: bool = True,
        cancel_event: CancelEvt = None,
    ) -> Iterator[ChatChunk]:
        if cancel_event and cancel_event.is_set():
            return
        # measure end-to-end latency: cold calls include model load in TTFT
        started = time.perf_counter()
        llm = self._load()

        # Use proper chat format for Qwen models
        prompt = self._messages_to_chat_format(messages, tools)
        first_token_at: float | None = None
        tokens = 0

        try:
            if stream:
                for chunk in llm.create_chat_completion(
                    messages=prompt,
                    max_tokens=256,
                    temperature=self._s.temperature,
                    repeat_penalty=1.15,
                    stop=["User:", "System:", "Assistant:"],
                    stream=True,
                ):
                    if cancel_event and cancel_event.is_set():
                        return
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        tokens += count_tokens(text)
                        yield ChatChunk(text=text)
                self._log_chat_timings(started, first_token_at, tokens)
                yield ChatChunk(done=True)
            else:
                result = llm.create_chat_completion(
                    messages=prompt,
                    max_tokens=256,
                    temperature=self._s.temperature,
                    repeat_penalty=1.15,
                    stop=["User:", "System:", "Assistant:"],
                    stream=False,
                )
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                self._log_chat_timings(started, None, count_tokens(text))
                yield ChatChunk(text=text, done=True)
        except Exception as exc:
            logger.error("llama-cpp inference error: %s", exc)
            yield ChatChunk(text=f"[LLM error: {exc}]", done=True)

    def _messages_to_chat_format(self, messages: MsgList, tools: MsgList | None) -> list[dict]:
        """Convert messages to llama-cpp chat format (Qwen uses ChatML format)."""
        formatted = []
        system_content = ""
        
        # Build system message content
        if tools:
            system_content = "Available tools: " + json.dumps(tools)
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                if system_content:
                    system_content += "\n" + content
                else:
                    system_content = content
            elif role == "user":
                formatted.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted.append({"role": "assistant", "content": content})
            elif role == "tool":
                formatted.append({"role": "tool", "content": content})
        
        # Add system message at the beginning if it exists
        if system_content:
            formatted.insert(0, {"role": "system", "content": system_content})
        
        return formatted

    def _messages_to_prompt(self, messages: MsgList, tools: MsgList | None) -> str:
        """Legacy method - kept for compatibility."""
        return self._messages_to_chat_format(messages, tools)

    def unload(self) -> None:
        self._llm = None

    def prewarm(self) -> None:
        self._load()


class FakeLLMBackend:
    def __init__(self, chunks: Any = (), chunk_delay: float = 0.0, raise_timeout: bool = False) -> None:
        self.chunks, self.chunk_delay = list(chunks), chunk_delay
        self.raise_timeout, self.calls = raise_timeout, []
        self.unload_calls: int = 0
        self.prewarm_calls: int = 0
        self.unloaded: bool = False
        self.prewarmed: bool = False

    def chat(
        self,
        messages: MsgList,
        tools: MsgList | None = None,
        stream: bool = True,
        cancel_event: CancelEvt = None,
    ) -> Iterator[ChatChunk]:
        self.calls.append({"messages": messages, "tools": tools, "stream": stream})
        if self.raise_timeout:
            raise TimeoutError("Simulated LLM timeout")
        for i, item in enumerate(self.chunks):
            if cancel_event and cancel_event.is_set():
                return
            if cancel_event and self.chunk_delay and cancel_event.wait(self.chunk_delay):
                return
            if not cancel_event and self.chunk_delay:
                time.sleep(self.chunk_delay)
            done = i == len(self.chunks) - 1
            yield ChatChunk(item, done=done) if isinstance(item, str) else item

    def unload(self) -> None:
        self.unload_calls += 1
        self.unloaded = True

    def prewarm(self) -> None:
        self.prewarm_calls += 1
        self.prewarmed = True


def _create_backend(settings: LLMSettings) -> Any:
    if settings.backend == "llamacpp":
        return LlamaCppBackend(settings)
    return OllamaBackend(settings)


class LLMClient:
    def __init__(self, settings: LLMSettings, backend: Any = None) -> None:
        self.settings, self.backend = settings, backend or _create_backend(settings)

    def chat(self, messages: MsgList, **kwargs: Any) -> Iterator[ChatChunk]:
        return self.backend.chat(messages, **kwargs)

    def unload(self) -> None:
        self.backend.unload()

    def prewarm(self) -> None:
        self.backend.prewarm()
