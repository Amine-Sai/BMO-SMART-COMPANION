"""BMO Contracts Data Models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    THINKING = "thinking"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    WINK = "wink"
    ERROR = "error"


class VisionStatus(Model):
    online: bool


class BrainStatus(Model):
    online: bool
    state: Literal["idle", "listening", "thinking", "speaking", "muted", "error"]


class HwStatus(Model):
    online: bool
    ts: float
    capabilities: list[str]


class HwEvent(Model):
    ts: float
    kind: Literal["mute_switch"]
    value: bool


class FaceData(Model):
    track_id: int | None = None
    name: str
    conf: float


class PersonData(Model):
    track_id: int | None = None
    pos: Literal["left", "center", "right"]
    proximity: Literal["near", "medium", "far"]


class PersonLeftData(Model):
    track_id: int | None = None


class SignData(Model):
    text: str
    conf: float


class ObjectData(Model):
    label: str
    conf: float


class FaceIdentified(Model):
    v: Literal[1] = 1
    id: str
    ts: float
    type: Literal["face_identified"] = "face_identified"
    data: FaceData


class PersonDetected(Model):
    v: Literal[1] = 1
    id: str
    ts: float
    type: Literal["person_detected"] = "person_detected"
    data: PersonData


class PersonLeft(Model):
    v: Literal[1] = 1
    id: str
    ts: float
    type: Literal["person_left"] = "person_left"
    data: PersonLeftData


class SignRecognized(Model):
    v: Literal[1] = 1
    id: str
    ts: float
    type: Literal["sign_recognized"] = "sign_recognized"
    data: SignData


class ObjectSeen(Model):
    v: Literal[1] = 1
    id: str
    ts: float
    type: Literal["object_seen"] = "object_seen"
    data: ObjectData


Event = Annotated[
    FaceIdentified | PersonDetected | PersonLeft | SignRecognized | ObjectSeen,
    Field(discriminator="type"),
]
EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event_json(raw: bytes | str) -> Event:
    return EVENT_ADAPTER.validate_json(raw)


def validate_event(obj: Any) -> Event:
    return EVENT_ADAPTER.validate_python(obj)


class ScenePerson(Model):
    track_id: int | None = None
    name: str
    pos: Literal["left", "center", "right"]
    proximity: Literal["near", "medium", "far"]
    since_s: float


class LastSign(Model):
    text: str
    age_s: float


class Scene(Model):
    ts: float
    people: list[ScenePerson]
    objects: list[str]
    last_sign: LastSign | None = None
    recent: list[str]


class VisionCmd(Model):
    id: str
    cmd: Literal["enroll", "forget", "reload_gallery", "set_modules"]
    args: dict[str, Any] = Field(default_factory=dict)


class VisionCmdResult(Model):
    id: str
    ok: bool
    detail: str


class BrainSay(Model):
    utt_id: str
    text: str
    emotion: Emotion


class HwCmd(Model):
    id: str
    target: Literal["pi", "esp32"]
    action: Literal["set_expression", "move_servo", "play_sound"]
    args: dict[str, Any] = Field(default_factory=dict)
