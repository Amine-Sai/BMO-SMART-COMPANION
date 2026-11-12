"""Detects when a user utterance likely needs the camera to answer."""

from __future__ import annotations

import re

# Phrases that imply the user is asking about what the camera can see.
_PATTERNS = [
    r"\bwho(?:'s| is| are| was|se)?\b.*\b(?:this|that|there|here|standing|in front|behind me|at the door)\b",
    r"\bwho(?:'s| is| are|se)?\b\s*(?:this|that|it)\b",
    r"\bwhat\s+(?:am|are)\s+(?:i|you|we)\s+(?:holding|wearing|carrying|looking at)\b",
    r"\bwhat(?:'s| is)\s+(?:this|that|in my hand|in my hands|on the (?:table|desk|screen))\b",
    r"\bwhat\s+do\s+you\s+see\b",
    r"\bcan\s+you\s+see\b",
    r"\bdo\s+you\s+see\b",
    r"\blook\s+(?:at|around|here|there)\b",
    r"\bhow\s+many\s+(?:people|persons|persons are|of us)\b",
    r"\bdescribe\s+(?:me|the (?:scene|room)|what you see)\b",
    r"\b(?:is|are)\s+(?:there|anyone|anybody|someone|somebody)\b",
    r"\bwhat\s+(?:color|colour)\b",
    r"\bam\s+i\s+(?:holding|wearing)\b",
]

_INTENT_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def is_vision_intent(text: str) -> bool:
    """Return True if ``text`` likely requires the camera to answer."""
    if not text:
        return False
    return _INTENT_RE.search(text) is not None
