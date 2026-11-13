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


class HwStatus(Model):
    v: Literal[1] = 1
    online: bool
    ts: float
    capabilities: list[str]


class HwEvent(Model):
    v: Literal[1] = 1
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


ServoId = Literal["pan", "tilt", "left_arm", "right_arm"]


class SetExpressionArgs(Model):
    emotion: Emotion

    @model_validator(mode="before")
    @classmethod
    def _normalize_expression(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "emotion" not in data and "expression" in data:
                data = dict(data)
                data["emotion"] = data.pop("expression")
        return data

    def get_emotion(self) -> Emotion:
        return self.emotion

    def get_expression(self) -> str:
        return self.emotion.value

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class MoveServoArgs(Model):
    id: ServoId
    angle: float

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class PlaySoundArgs(Model):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.]+$")

    @model_validator(mode="before")
    @classmethod
    def _normalize_sound(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "name" not in data and "sound" in data:
                data = dict(data)
                data["name"] = data.pop("sound")
        return data

    def get_sound(self) -> str:
        return self.name

    def get_name(self) -> str:
        return self.name

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class HwCmd(Model):
    v: Literal[1] = 1
    id: str
    target: Literal["pi", "esp32"]
    action: Literal["set_expression", "move_servo", "play_sound"]
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
        if not self.args:
            return self
        if self.action == "set_expression":
            validated_expr = SetExpressionArgs.model_validate(self.args)
            self.args["emotion"] = validated_expr.emotion
        elif self.action == "move_servo":
            validated_servo = MoveServoArgs.model_validate(self.args)
            self.args["id"] = validated_servo.id
            self.args["angle"] = validated_servo.angle
        elif self.action == "play_sound":
            validated_sound = PlaySoundArgs.model_validate(self.args)
            self.args["name"] = validated_sound.name
        return self
