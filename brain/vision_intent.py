"""Detects when a user utterance likely needs the camera to answer.

Also classifies what the camera should look for so the vision pipeline only
runs the relevant models (people vs. objects).
"""

from __future__ import annotations

import re

_PEOPLE_PATTERNS = [
    r"\bwho(?:['\u2019]?s| is| are| was)?\b",
    r"\b(?:person|people|persons|someone|somebody|anyone|anybody|face|faces)\b",
    r"\bhow\s+many\s+(?:of\s+us|people|persons)\b",
    r"\b(?:is|are)\s+there\s+(?:anyone|anybody|someone|somebody)\b",
    r"\b(?:recognize|recognise|know)\s+(?:this|that|the)\s+person\b",
    r"\b(?:standing|at the door|in front of me|behind me)\b",
]

_OBJECT_PATTERNS = [
    r"\bwhat\s+(?:am|are)\s+(?:i|you|we)\s+(?:holding|wearing|carrying)\b",
    r"\bwhat(?:'s| is)\s+(?:this|that|in my hand|in my hands)\b",
    r"\b(?:holding|holding up|in my hand|in my hands)\b",
    r"\b(?:object|objects|item|items|thing|things)\b",
    r"\bon\s+the\s+(?:table|desk|screen|floor|shelf)\b",
    r"\bwhat\s+(?:color|colour)\b",
    r"\b(?:book|cup|phone|bottle|box|bag|key|keys|laptop|pen)\b",
]

_BOTH_PATTERNS = [
    r"\bwhat\s+do\s+you\s+see\b",
    r"\bcan\s+you\s+see\b",
    r"\bdo\s+you\s+see\b",
    r"\blook\s+(?:at|around|here|there)\b",
    r"\bdescribe\s+(?:me|the scene|the room|what you see)\b",
    r"\bwhat(?:'s| is)\s+(?:going on|around me|in the room)\b",
    r"\bscan\b",
]

_PEOPLE_RE = re.compile("|".join(f"(?:{p})" for p in _PEOPLE_PATTERNS), re.IGNORECASE)
_OBJECT_RE = re.compile("|".join(f"(?:{p})" for p in _OBJECT_PATTERNS), re.IGNORECASE)
_BOTH_RE = re.compile("|".join(f"(?:{p})" for p in _BOTH_PATTERNS), re.IGNORECASE)

Target = str  # "people" | "objects" | "both"


def vision_target(text: str) -> Target | None:
    """Return what to look for: ``people``, ``objects``, ``both``, or None.

    None means the utterance does not need the camera at all.
    """
    if not text:
        return None
    # "Who are you?" is about BMO's identity, not a camera question.
    if re.search(r"\bwho\s+(?:are|r)\s+you\b", text, re.IGNORECASE) and not re.search(
        r"\b(?:see|seeing|looking|person|people|there|this|that|photo|camera|behind|front)\b",
        text,
        re.IGNORECASE,
    ):
        return None
    if _BOTH_RE.search(text):
        return "both"

    wants_people = _PEOPLE_RE.search(text) is not None
    wants_objects = _OBJECT_RE.search(text) is not None
    if wants_people and wants_objects:
        return "both"
    if wants_people:
        return "people"
    if wants_objects:
        return "objects"
    return None


def is_vision_intent(text: str) -> bool:
    """Return True if ``text`` likely requires the camera to answer."""
    return vision_target(text) is not None
