"""BMO Contracts Topic Registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from .models import (
    EVENT_ADAPTER,
    BrainSay,
    BrainStatus,
    Scene,
    VisionCmd,
    VisionCmdResult,
    VisionStatus,
)

Role = Literal["cv", "brain"]


@dataclass(frozen=True)
class TopicSpec:
    name: str
    qos: int
    retain: bool
    model: Any
    pub_roles: tuple[Role, ...]
    sub_roles: tuple[Role, ...]


BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883

VISION_STATUS = "bmo/vision/status"
VISION_EVENTS = "bmo/vision/events"
VISION_CONTEXT = "bmo/vision/context"
VISION_CMD = "bmo/vision/cmd"
VISION_CMD_RESULT = "bmo/vision/cmd_result"
BRAIN_STATUS = "bmo/brain/status"
BRAIN_SAY = "bmo/brain/say"

# Wildcard topic for monitoring tools
VISION_WILDCARD = "bmo/vision/#"

TOPICS: dict[str, TopicSpec] = {
    VISION_STATUS: TopicSpec(
        name=VISION_STATUS,
        qos=1,
        retain=True,
        model=VisionStatus,
        pub_roles=("cv",),
        sub_roles=("brain",),
    ),
    VISION_EVENTS: TopicSpec(
        name=VISION_EVENTS,
        qos=1,
        retain=False,
        model=EVENT_ADAPTER,
        pub_roles=("cv",),
        sub_roles=("brain",),
    ),
    VISION_CONTEXT: TopicSpec(
        name=VISION_CONTEXT,
        qos=0,
        retain=True,
        model=Scene,
        pub_roles=("cv",),
        sub_roles=("brain",),
    ),
    VISION_CMD: TopicSpec(
        name=VISION_CMD,
        qos=1,
        retain=False,
        model=VisionCmd,
        pub_roles=("brain",),
        sub_roles=("cv",),
    ),
    VISION_CMD_RESULT: TopicSpec(
        name=VISION_CMD_RESULT,
        qos=1,
        retain=False,
        model=VisionCmdResult,
        pub_roles=("cv",),
        sub_roles=("brain",),
    ),
    BRAIN_STATUS: TopicSpec(
        name=BRAIN_STATUS,
        qos=1,
        retain=True,
        model=BrainStatus,
        pub_roles=("brain",),
        sub_roles=("cv",),
    ),
    BRAIN_SAY: TopicSpec(
        name=BRAIN_SAY,
        qos=1,
        retain=False,
        model=BrainSay,
        pub_roles=("brain",),
        sub_roles=(),
    ),
}
