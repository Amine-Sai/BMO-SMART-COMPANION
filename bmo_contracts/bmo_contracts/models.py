"""BMO Contracts Data Models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


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
    ANGRY = "angry"
    MAD = "mad"
    ANNOYED = "annoyed"
    PLAYFUL = "playful"
    LOVE = "love"
    KISS = "kiss"
    EXCITED = "excited"
    AMAZED = "amazed"
    TERRIFIED = "terrified"
    EMPATHETIC = "empathetic"


class VisionStatus(Model):
    v: Literal[1] = 1
    online: bool


class BrainStatus(Model):
    v: Literal[1] = 1
    online: bool
    state: Literal["idle", "listening", "thinking", "speaking", "muted", "error"]


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


class ObjectSeen(Model):
    v: Literal[1] = 1
    id: str
    ts: float
    type: Literal["object_seen"] = "object_seen"
    data: ObjectData


Event = Annotated[
    FaceIdentified | PersonDetected | PersonLeft | ObjectSeen,
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


class Scene(Model):
    v: Literal[1] = 1
    ts: float
    people: list[ScenePerson]
    objects: list[str]
    recent: list[str]


PersonName = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9 _-]+$")]


class EnrollArgs(Model):
    name: PersonName
    shots: int = 8


class ForgetArgs(Model):
    name: PersonName


class VisionCmd(Model):
    v: Literal[1] = 1
    id: str
    cmd: Literal["enroll", "forget", "reload_gallery", "set_modules"]
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_args(cls, data: Any) -> Any:
        if isinstance(data, dict) and "args" in data:
            args_val = data["args"]
            if isinstance(args_val, BaseModel):
                data = dict(data)
                data["args"] = args_val.model_dump()
        return data

    @model_validator(mode="after")
    def _validate_args(self) -> Self:
        if self.cmd in ("enroll", "forget"):
            if "name" in self.args:
                TypeAdapter(PersonName).validate_python(self.args["name"])
        return self


class VisionCmdResult(Model):
    v: Literal[1] = 1
    id: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] | None = None


class BrainSay(Model):
    v: Literal[1] = 1
    utt_id: str
    text: str
    emotion: Emotion
