from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from brain.llm import ToolCall


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]
    risky: bool = False


class ToolRegistry:
    def __init__(
        self,
        confirmation_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.confirmation_timeout_s = confirmation_timeout_s
        self._clock = clock
        self._specs: dict[str, ToolSpec] = {}
        self._pending: tuple[str, BaseModel, float] | None = None

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def reset(self) -> None:
        self._pending = None

    def ollama_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.schema.model_json_schema(),
                },
            }
            for spec in self._specs.values()
        ]

    def dispatch(self, call: ToolCall | Mapping[str, Any], *, confirmed: bool = False) -> dict[str, Any]:
        if isinstance(call, ToolCall):
            name = call.name
            arguments = call.arguments
        elif isinstance(call, Mapping):
            name = call.get("name")
            arguments = call.get("arguments")
        else:
            return {"ok": False, "error": f"Unknown or invalid tool: {call!r}"}

        if not isinstance(name, str) or name not in self._specs or not isinstance(arguments, Mapping):
            return {"ok": False, "error": f"Unknown or invalid tool: {name!r}"}

        spec = self._specs[name]
        try:
            args_obj = spec.schema.model_validate(dict(arguments))
        except ValidationError as exc:
            return {"ok": False, "error": str(exc)}

        if spec.risky and not confirmed:
            self._pending = (spec.name, args_obj, self._clock())
            return {
                "ok": False,
                "needs_confirmation": True,
                "tool": spec.name,
                "detail": f"Confirm execution of {spec.name}",
            }

        if spec.risky and confirmed:
            if (
                self._pending is None
                or self._pending[0] != spec.name
                or self._pending[1] != args_obj
                or (self._clock() - self._pending[2]) > self.confirmation_timeout_s
            ):
                self._pending = None
                return {"ok": False, "error": "Confirmation missing or expired"}
            self._pending = None

        res = spec.handler(args_obj)
        return res if "ok" in res else {"ok": True, **res}


_TOOL_INTENT_RE = re.compile(
    r"\b(?:forget|delete|remove|remember|memorize|memorise|register|enroll|enrol|"
    r"rename|save|list|roster|my name|who am i)\b",
    re.IGNORECASE,
)


def needs_tools(text: str) -> bool:
    """Return True when an utterance likely needs a tool to be fulfilled."""
    return bool(text and _TOOL_INTENT_RE.search(text))
