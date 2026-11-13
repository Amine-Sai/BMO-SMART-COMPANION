from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyArgs(StrictArgs):
    pass


class FindObjectArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9 _-]+$")


class PersonNameArgs(StrictArgs):
    name: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9 _-]+$")
