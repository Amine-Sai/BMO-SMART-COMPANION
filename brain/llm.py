from __future__ import annotations

from dataclasses import dataclass
import gc
import inspect
import json
import logging
import threading
import time
from typing import Any, Iterator, Mapping, Protocol, Sequence
import httpx
from brain.settings import LLMSettings

try:
    from llama_cpp import Llama, llama_supports_gpu_offload
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    llama_supports_gpu_offload = None  # type: ignore
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


class ThinkFilter:
    """Filters out <think>...</think> blocks from a stream of text chunks."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def process(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(self.OPEN_TAG)
                if idx != -1:
                    if idx > 0:
                        out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self.OPEN_TAG):]
                    self._in_think = True
                else:
                    matched_len = 0
                    max_check = min(len(self.OPEN_TAG) - 1, len(self._buf))
                    for l in range(max_check, 0, -1):
                        if self.OPEN_TAG.startswith(self._buf[-l:]):
                            matched_len = l
                            break
                    if matched_len > 0:
                        if len(self._buf) > matched_len:
                            out.append(self._buf[:-matched_len])
                            self._buf = self._buf[-matched_len:]
                        break
                    else:
                        out.append(self._buf)
                        self._buf = ""
                        break
            else:
                idx = self._buf.find(self.CLOSE_TAG)
                if idx != -1:
                    self._buf = self._buf[idx + len(self.CLOSE_TAG):]
                    self._in_think = False
                else:
                    matched_len = 0
                    max_check = min(len(self.CLOSE_TAG) - 1, len(self._buf))
                    for l in range(max_check, 0, -1):
                        if self.CLOSE_TAG.startswith(self._buf[-l:]):
                            matched_len = l
                            break
                    if matched_len > 0:
                        self._buf = self._buf[-matched_len:]
                        break
                    else:
                        self._buf = ""
                        break
        return "".join(out)

    def flush(self) -> str:
        if self._in_think:
            self._buf = ""
            return ""
        res = self._buf
        self._buf = ""
        return res


def strip_thinking(text: str) -> str:
    tf = ThinkFilter()
    return tf.process(text) + tf.flush()


def _parse_single_tool_call(tc: dict[str, Any]) -> ToolCall | None:
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return None
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
    return ToolCall(name=name, arguments=args, error=error)


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
    emitted_tc = False
    for raw_tc in calls:
        tc = _parse_single_tool_call(raw_tc)
        if tc is not None:
            yield ChatChunk(content, tc, done)
            content = ""
            emitted_tc = True
    if content or (done and not emitted_tc):
        yield ChatChunk(content, done=done)


def _supports_chat_template_kwargs(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
        return (
            "chat_template_kwargs" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
    except (ValueError, TypeError):
        return False


def _call_create_chat_completion(llm: Any, kwargs: dict[str, Any]) -> Any:
    try:
        return llm.create_chat_completion(**kwargs)
    except TypeError as exc:
        if "chat_template_kwargs" in str(exc) and "chat_template_kwargs" in kwargs:
            kwargs_copy = dict(kwargs)
            kwargs_copy.pop("chat_template_kwargs", None)
            return llm.create_chat_completion(**kwargs_copy)
        raise


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
            yield ChatChunk(done=True)
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
            if cancel_event and cancel_event.is_set():
                yield ChatChunk(done=True)
                return
            yield from _parse_chunk(data)
            return
        try:
            with self._client.stream("POST", "/api/chat", json=p) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        yield ChatChunk(done=True)
                        return
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
                            yield ChatChunk(done=True)
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
        self._decode_lock = threading.Lock()

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
        supported = False
        # 1. instance method (newer llama-cpp-python)
        probe = getattr(self._llm, "supports_gpu_offload", None)
        if callable(probe):
            try:
                supported = bool(probe())
            except Exception as exc:  # capability probe must never break loading
                logger.warning("llama-cpp GPU offload probe failed: %s", exc)
                supported = False
        # 2. module-level function (llama-cpp-python <= 0.3.36)
        elif callable(llama_supports_gpu_offload):
            try:
                supported = bool(llama_supports_gpu_offload())
            except Exception as exc:
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
        # Serialize decodes: the text thread and voice thread share one model.
        with self._decode_lock:
            yield from self._chat_impl(messages, tools, stream, cancel_event)

    def _chat_impl(
        self,
        messages: MsgList,
        tools: MsgList | None = None,
        stream: bool = True,
        cancel_event: CancelEvt = None,
    ) -> Iterator[ChatChunk]:
        if cancel_event and cancel_event.is_set():
            yield ChatChunk(done=True)
            return

        start_time = time.monotonic()
        started = time.perf_counter()
        if time.monotonic() - start_time > self._s.timeout_s:
            raise TimeoutError("Inference timed out")

        llm = self._load()
        if time.monotonic() - start_time > self._s.timeout_s:
            raise TimeoutError("Inference timed out")
        if cancel_event and cancel_event.is_set():
            yield ChatChunk(done=True)
            return

        # Use proper chat format for Qwen models
        prompt = self._messages_to_chat_format(messages, tools)
        kwargs: dict[str, Any] = {
            "messages": prompt,
            "max_tokens": self._s.max_tokens,
            "temperature": self._s.temperature,
            "repeat_penalty": 1.15,
            "stop": ["User:", "System:", "Assistant:"],
            "stream": stream,
        }
        if _supports_chat_template_kwargs(llm.create_chat_completion):
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        think_filter = ThinkFilter()
        first_token_at: float | None = None
        tokens = 0

        try:
            if stream:
                gen = _call_create_chat_completion(llm, kwargs)
                try:
                    for chunk in gen:
                        if time.monotonic() - start_time > self._s.timeout_s:
                            raise TimeoutError("Inference timed out")
                        if cancel_event and cancel_event.is_set():
                            yield ChatChunk(done=True)
                            return
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        raw_content = delta.get("content", "") or ""
                        filtered_text = think_filter.process(raw_content)
                        if filtered_text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            tokens += count_tokens(filtered_text)
                            yield ChatChunk(text=filtered_text)
                        raw_calls = delta.get("tool_calls")
                        if isinstance(raw_calls, list):
                            for raw_tc in raw_calls:
                                tc = _parse_single_tool_call(raw_tc)
                                if tc is not None:
                                    yield ChatChunk(tool_call=tc)
                finally:
                    close_fn = getattr(gen, "close", None)
                    if callable(close_fn):
                        close_fn()

                if cancel_event and cancel_event.is_set():
                    yield ChatChunk(done=True)
                    return

                remaining = think_filter.flush()
                if remaining:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    tokens += count_tokens(remaining)
                    yield ChatChunk(text=remaining)
                self._log_chat_timings(started, first_token_at, tokens)
                yield ChatChunk(done=True)
            else:
                result = _call_create_chat_completion(llm, kwargs)
                if time.monotonic() - start_time > self._s.timeout_s:
                    raise TimeoutError("Inference timed out")
                if cancel_event and cancel_event.is_set():
                    yield ChatChunk(done=True)
                    return
                msg = result.get("choices", [{}])[0].get("message", {})
                raw_content = msg.get("content", "") or ""
                text = strip_thinking(raw_content)
                raw_calls = msg.get("tool_calls")
                calls = raw_calls if isinstance(raw_calls, list) else []
                emitted_tc = False
                for i, raw_tc in enumerate(calls):
                    tc = _parse_single_tool_call(raw_tc)
                    if tc is not None:
                        is_last = (i == len(calls) - 1)
                        yield ChatChunk(text=text, tool_call=tc, done=is_last)
                        text = ""
                        emitted_tc = True
                if text or (not emitted_tc):
                    yield ChatChunk(text=text, done=True)
                self._log_chat_timings(started, None, count_tokens(raw_content))
        except TimeoutError:
            raise
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
        with self._decode_lock:
            if self._llm is not None:
                close_fn = getattr(self._llm, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as exc:
                        logger.warning("Error closing llama-cpp model: %s", exc)
                self._llm = None
                gc.collect()

    def prewarm(self) -> None:
        with self._decode_lock:
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
