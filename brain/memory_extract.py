from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.memory import LongTermMemory


def _clean(value: str) -> str:
    value = value.strip().strip(" .,!?;:\"'")
    if value.lower().startswith("that "):
        value = value[5:].strip()
    return value


def _note_key(value: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", value.lower()).split()
    while words and words[0] in {"the", "a", "an", "my", "our", "your"}:
        words.pop(0)
    return " ".join(words[:6]) or "note"


def extract_memories(text: str) -> list[tuple[str, str, str]]:
    """Return (key, value, category) facts explicitly stated by the user."""
    t = (text or "").strip()
    if not t:
        return []

    found: list[tuple[str, str, str]] = []

    m = re.search(r"\b(?:my name is|call me)\s+(.+?)(?:[.!?]|$)", t, re.IGNORECASE)
    if m:
        value = _clean(m.group(1))
        if value:
            found.append(("user name", value, "identity"))

    m = re.search(
        r"\bmy fav(?:ou?rite)?\s+(.+?)\s+is\s+(.+?)(?:[.!?]|$)", t, re.IGNORECASE
    )
    if m:
        topic, value = _clean(m.group(1)).lower(), _clean(m.group(2))
        if topic and value:
            found.append((f"favorite {topic}", value, "preference"))

    m = re.search(r"\bi (?:like|love|prefer)\s+(.+?)(?:[.!?]|$)", t, re.IGNORECASE)
    if m:
        value = _clean(m.group(1))
        if value:
            found.append(("likes", value, "preference"))

    m = re.search(r"\bi live in\s+(.+?)(?:[.!?]|$)", t, re.IGNORECASE)
    if m:
        value = _clean(m.group(1))
        if value:
            found.append(("home", value, "fact"))

    m = re.search(r"\bi work (?:at|for)\s+(.+?)(?:[.!?]|$)", t, re.IGNORECASE)
    if m:
        value = _clean(m.group(1))
        if value:
            found.append(("work", value, "fact"))

    # People: this is <name>, meet <name>, or remember <name>
    m = re.search(r"\b(?:this is|meet|remember)\s+([A-Z][a-z]+)(?:\s+.+?)(?:[.!?]|$)", t)
    if m:
        name = _clean(m.group(1))
        if name and name.lower() not in {"i", "you", "we", "they", "he", "she", "it"}:
            # Extract the rest as details
            rest = t[m.end():].strip(" ,.!?")
            detail = _clean(rest) if rest else f"Met {name}"
            found.append((f"person {name}", detail, "person"))

    # Objects: this is a <object> or remember this <object>
    m = re.search(r"\b(?:this is (?:a|an)|remember (?:this|that))\s+([a-z]+)(?:\s+.+?)(?:[.!?]|$)", t, re.IGNORECASE)
    if m:
        obj = _clean(m.group(1))
        if obj and obj.lower() not in {"i", "you", "we", "they", "he", "she", "it", "person", "name"}:
            rest = t[m.end():].strip(" ,.!?")
            detail = _clean(rest) if rest else f"Learned about {obj}"
            found.append((f"object {obj}", detail, "object"))

    # Generic "remember that ..." / "don't forget ..." only when no more
    # specific pattern already captured the statement.
    if not found:
        m = re.search(
            r"\b(?:remember|note|keep in mind|don'?t forget)(?:\s+that)?\s+(.+?)(?:[.!?]|$)",
            t,
            re.IGNORECASE,
        )
        if m:
            value = _clean(m.group(1))
            if " is " in value:
                topic, detail = value.split(" is ", 1)
                key, detail = _note_key(topic), _clean(detail)
                if key != "note" and detail:
                    found.append((key, detail, "note"))
                else:
                    found.append((_note_key(value), value, "note"))
            elif value and len(value) > 1:
                found.append((_note_key(value), value, "note"))

    return found


def extract_forget(text: str) -> str | None:
    m = re.search(
        r"\b(?:forget|delete|remove)\s+(?:about\s+)?(.+?)(?:\s+from (?:your )?memory)?(?:[.!?]|$)",
        text or "",
        re.IGNORECASE,
    )
    if not m:
        return None
    key = _clean(m.group(1))
    if not key or key.lower() in {"it", "that", "this", "everything", "all"}:
        return None
    return key


def apply_memory_effects(text: str, memory: "LongTermMemory") -> list[str]:
    """Persist explicit remember/forget statements. Returns human-readable notes."""
    notes: list[str] = []
    forget_key = extract_forget(text)
    if forget_key and memory.forget(forget_key):
        notes.append(f"forgot {forget_key}")
    for key, value, category in extract_memories(text):
        if category == "person":
            memory.remember_person(key.replace("person ", ""), value)
        elif category == "object":
            memory.remember_object(key.replace("object ", ""), value)
        else:
            memory.remember(key, value, category=category)
        notes.append(f"remembered {key}")
    return notes
