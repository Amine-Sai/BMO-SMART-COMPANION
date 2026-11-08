from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from pydantic import BaseModel, Field
import yaml

from bmo_contracts.models import HwCmd
from bmo_contracts.topics import HW_CMD
from brain.tools.registry import ToolRegistry, ToolSpec

__all__ = [
    "ActuatorLimits",
    "ActuatorTools",
    "MoveServoArgs",
    "PlaySoundArgs",
    "SetExpressionArgs",
    "load_limits",
    "register_actuator_tools",
]


class SetExpressionArgs(BaseModel):
    expression: str = Field(default="", description="Facial expression name")
    emotion: str = Field(default="", description="Facial expression name")

    def get_expression(self) -> str:
        return self.expression or self.emotion


class MoveServoArgs(BaseModel):
    id: str = Field(description="Servo identifier, e.g. pan or tilt")
    angle: float = Field(description="Target angle in degrees")


class PlaySoundArgs(BaseModel):
    sound: str = Field(default="", description="Sound effect name to play")
    name: str = Field(default="", description="Sound effect name to play")

    def get_sound(self) -> str:
        return self.sound or self.name


class ActuatorLimits(BaseModel):
    emotions: list[str] = Field(default_factory=list)
    servos: dict[str, Any] = Field(default_factory=dict)
    min_interval_s: float = 0.25
    max_deg_per_sec: float = 180.0
    watchdog_timeout_ms: int = 1000
    sounds: list[str] = Field(default_factory=list)


def load_limits(path: Path | str = "config/limits.yaml") -> ActuatorLimits:
    p = Path(path)
    if not p.is_file():
        return ActuatorLimits()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rate = data.get("rate_limits", {})
    return ActuatorLimits(
        emotions=list(data.get("emotions", [])),
        servos=dict(data.get("servos", {})),
        min_interval_s=float(rate.get("min_interval_s", 0.25)),
        max_deg_per_sec=float(rate.get("max_deg_per_sec", 180.0)),
        watchdog_timeout_ms=int(rate.get("watchdog_timeout_ms", 1000)),
        sounds=list(data.get("sounds", [])),
    )


def _get_servo_range(servo_cfg: Any) -> tuple[float, float]:
    if isinstance(servo_cfg, dict):
        min_val = servo_cfg.get(
            "min_deg",
            servo_cfg.get("min", servo_cfg.get("min_angle", -90.0)),
        )
        max_val = servo_cfg.get(
            "max_deg",
            servo_cfg.get("max", servo_cfg.get("max_angle", 90.0)),
        )
        return float(min_val), float(max_val)
    if isinstance(servo_cfg, (list, tuple)) and len(servo_cfg) >= 2:
        return float(servo_cfg[0]), float(servo_cfg[1])
    return -90.0, 90.0


class ActuatorTools:
    def __init__(
        self,
        bus: Any = None,
        limits: ActuatorLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.limits = limits or load_limits()
        self.clock = clock
        self._last_servo_time: float | None = None

    def move_servo(
        self,
        id: str | MoveServoArgs | dict[str, Any] | None = None,
        angle: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(id, MoveServoArgs):
            servo_id = id.id
            target_angle = id.angle
        elif isinstance(id, dict):
            servo_id = str(id.get("id", ""))
            target_angle = float(id.get("angle", 0.0))
        elif id is not None:
            servo_id = str(id)
            target_angle = float(angle if angle is not None else kwargs.get("angle", 0.0))
        else:
            servo_id = str(kwargs.get("id", ""))
            target_angle = float(kwargs.get("angle", 0.0))

        now = self.clock()
        if self._last_servo_time is not None and (now - self._last_servo_time) < self.limits.min_interval_s:
            return {"ok": False, "error": "rate_limited"}

        servo_cfg = self.limits.servos.get(servo_id)
        if servo_cfg is None:
            return {"ok": False, "error": f"unknown servo: {servo_id}"}

        min_angle, max_angle = _get_servo_range(servo_cfg)

        clamped = False
        clamped_angle = target_angle
        if target_angle < min_angle:
            clamped_angle = float(min_angle)
            clamped = True
        elif target_angle > max_angle:
            clamped_angle = float(max_angle)
            clamped = True

        self._last_servo_time = now

        if self.bus is not None:
            cmd = HwCmd(
                id=f"servo-{servo_id}-{int(now * 1000)}",
                target="esp32",
                action="move_servo",
                args={"id": servo_id, "angle": clamped_angle},
            )
            self.bus.publish(HW_CMD, cmd)

        return {
            "ok": True,
            "id": servo_id,
            "angle": clamped_angle,
            "clamped": clamped,
        }

    def set_expression(
        self,
        expression: str | SetExpressionArgs | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(expression, SetExpressionArgs):
            expr_val = expression.get_expression()
        elif isinstance(expression, dict):
            expr_val = str(expression.get("expression") or expression.get("emotion") or "")
        elif expression is not None:
            expr_val = str(expression)
        else:
            expr_val = str(kwargs.get("expression") or kwargs.get("emotion") or "")

        if expr_val not in self.limits.emotions:
            return {"ok": False, "error": f"unknown expression: {expr_val}"}

        if self.bus is not None:
            cmd = HwCmd(
                id=f"expr-{expr_val}-{int(self.clock() * 1000)}",
                target="esp32",
                action="set_expression",
                args={"expression": expr_val},
            )
            self.bus.publish(HW_CMD, cmd)

        return {"ok": True, "expression": expr_val}

    def play_sound(
        self,
        sound: str | PlaySoundArgs | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(sound, PlaySoundArgs):
            sound_val = sound.get_sound()
        elif isinstance(sound, dict):
            sound_val = str(sound.get("sound") or sound.get("name") or "")
        elif sound is not None:
            sound_val = str(sound)
        else:
            sound_val = str(kwargs.get("sound") or kwargs.get("name") or "")

        if sound_val not in self.limits.sounds:
            return {"ok": False, "error": f"unknown sound: {sound_val}"}

        if self.bus is not None:
            cmd = HwCmd(
                id=f"sound-{sound_val}-{int(self.clock() * 1000)}",
                target="esp32",
                action="play_sound",
                args={"sound": sound_val},
            )
            self.bus.publish(HW_CMD, cmd)

        return {"ok": True, "sound": sound_val}


def register_actuator_tools(
    registry: ToolRegistry,
    bus: Any = None,
    limits: ActuatorLimits | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ActuatorTools:
    service = ActuatorTools(bus=bus, limits=limits, clock=clock)
    registry.register(
        ToolSpec(
            name="set_expression",
            description="Set facial expression",
            schema=SetExpressionArgs,
            handler=service.set_expression,
        )
    )
    registry.register(
        ToolSpec(
            name="move_servo",
            description="Move a servo to target angle",
            schema=MoveServoArgs,
            handler=service.move_servo,
        )
    )
    registry.register(
        ToolSpec(
            name="play_sound",
            description="Play a sound effect",
            schema=PlaySoundArgs,
            handler=service.play_sound,
        )
    )
    return service
