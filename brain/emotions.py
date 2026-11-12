"""Parse the leading emotion tag the model puts at the start of a reply."""

from __future__ import annotations

import re

from bmo_contracts.models import Emotion

_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")


def split_emotion_tag(text: str) -> tuple[Emotion | None, str]:
    """Return ``(emotion_or_None, text_without_leading_tag)``."""
    match = _TAG_RE.match(text)
    if match is None:
        return None, text
    try:
        emotion = Emotion(match.group(1).lower())
    except ValueError:
        emotion = Emotion.NEUTRAL
    return emotion, text[match.end() :].strip()


def strip_emotion_tag(text: str) -> str:
    """Remove a leading ``[happy]``-style tag, if present."""
    return split_emotion_tag(text)[1]
