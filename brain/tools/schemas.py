from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bmo_contracts.models import Emotion


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyArgs(StrictArgs):
    pass


class FindObjectArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9 _-]+$")


class PersonNameArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9 _-]+$")


class SetExpressionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emotion: Emotion


class MoveServoArgs(StrictArgs):
    id: Literal["pan", "tilt", "left_arm", "right_arm"]
    angle: float | int


class PlaySoundArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_]+$")
