from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Iterator, Mapping, Protocol, Sequence
import httpx
from brain.settings import LLMSettings

MsgList, CancelEvt = Sequence[Mapping[str, Any]], threading.Event | None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


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
    msg, done = data.get("message", {}), bool(data.get("done", False))
    content, calls = msg.get("content", ""), msg.get("tool_calls") or []
    for tc in calls:
        fn = tc.get("function", {})
        raw = fn.get("arguments", {})
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        yield ChatChunk(content, ToolCall(fn.get("name", ""), args), done)
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
            yield from _parse_chunk(self._post(p).json())
            return
        try:
            with self._client.stream("POST", "/api/chat", json=p) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    for c in _parse_chunk(json.loads(line)) if line else ():
                        if cancel_event and cancel_event.is_set():
                            return
                        yield c
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc

    def unload(self) -> None:
        self._post({"model": self._s.model, "messages": [], "keep_alive": 0})

    def prewarm(self) -> None:
        self._post({"model": self._s.model, "messages": [], "keep_alive": self._s.keep_alive})


class FakeLLMBackend:
    def __init__(self, chunks: Any = (), chunk_delay: float = 0.0, raise_timeout: bool = False) -> None:
        self.chunks, self.chunk_delay = list(chunks), chunk_delay
        self.raise_timeout, self.calls = raise_timeout, []

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

    def unload(self) -> None: ...
    def prewarm(self) -> None: ...


class LLMClient:
    def __init__(self, settings: LLMSettings, backend: Any = None) -> None:
        self.settings, self.backend = settings, backend or OllamaBackend(settings)

    def chat(self, messages: MsgList, **kwargs: Any) -> Iterator[ChatChunk]:
        return self.backend.chat(messages, **kwargs)

    def unload(self) -> None:
        self.backend.unload()

    def prewarm(self) -> None:
        self.backend.prewarm()
