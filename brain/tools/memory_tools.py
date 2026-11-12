from __future__ import annotations

from typing import Any

from pydantic import Field

from brain.memory import LongTermMemory
from brain.tools.registry import ToolRegistry, ToolSpec
from brain.tools.schemas import EmptyArgs, StrictArgs


class RememberFactArgs(StrictArgs):
    key: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=500)
    category: str = Field(default="fact", max_length=32)


class RecallFactArgs(StrictArgs):
    key: str = Field(min_length=1, max_length=80)


class ForgetFactArgs(StrictArgs):
    key: str = Field(min_length=1, max_length=80)


class MemoryTools:
    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def remember_fact(self, args: RememberFactArgs) -> dict[str, Any]:
        item = self._memory.remember(args.key, args.value, category=args.category)
        return {"ok": True, "item": item}

    def recall_fact(self, args: RecallFactArgs) -> dict[str, Any]:
        item = self._memory.recall(args.key)
        if item is None:
            return {"ok": True, "found": False, "key": args.key, "matches": self._memory.search(args.key)}
        return {"ok": True, "found": True, "item": item}

    def forget_fact(self, args: ForgetFactArgs) -> dict[str, Any]:
        removed = self._memory.forget(args.key)
        return {"ok": True, "forgot": removed, "key": args.key}

    def list_facts(self, _args: EmptyArgs | dict[str, Any] | None = None, **_kwargs: Any) -> dict[str, Any]:
        rows = self._memory.items()
        return {"ok": True, "facts": rows, "count": len(rows)}


def register_memory_tools(registry: ToolRegistry, memory: LongTermMemory) -> MemoryTools:
    tools = MemoryTools(memory)
    registry.register(
        ToolSpec(
            name="remember_fact",
            description=(
                "Save a lasting fact to long-term memory. Use when the user tells you "
                "something worth remembering: their name, preferences, routines, plans, "
                "or important details about people and the home."
            ),
            schema=RememberFactArgs,
            handler=tools.remember_fact,
            risky=False,
        )
    )
    registry.register(
        ToolSpec(
            name="recall_fact",
            description="Look up a previously saved fact by its key.",
            schema=RecallFactArgs,
            handler=tools.recall_fact,
            risky=False,
        )
    )
    registry.register(
        ToolSpec(
            name="forget_fact",
            description="Delete a saved fact from long-term memory.",
            schema=ForgetFactArgs,
            handler=tools.forget_fact,
            risky=False,
        )
    )
    registry.register(
        ToolSpec(
            name="list_facts",
            description="List everything currently saved in long-term memory.",
            schema=EmptyArgs,
            handler=tools.list_facts,
            risky=False,
        )
    )
    return tools
