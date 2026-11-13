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


_ANY_TAG_RE = re.compile(r"\[[a-zA-Z_]+\]\s*")


def strip_emotion_tags(text: str) -> str:
    """Remove every ``[emotion]`` tag (the model emits one per sentence)."""
    return _ANY_TAG_RE.sub("", text)


class EmotionTagStreamFilter:
    """Strip ``[emotion]`` tags from a streamed reply, holding partial tags."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text or ""
        out: list[str] = []
        while True:
            match = re.match(r"^\s*\[[a-zA-Z_]+\]\s*", self._buf)
            if match is not None:
                self._buf = self._buf[match.end():]
                continue
            partial = re.search(r"\[[a-zA-Z_]*$", self._buf)
            if partial is not None:
                out.append(self._buf[: partial.start()])
                self._buf = self._buf[partial.start():]
                break
            out.append(re.sub(r"\[[a-zA-Z_]+\]\s*", "", self._buf))
            self._buf = ""
            break
        return "".join(out)

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out
