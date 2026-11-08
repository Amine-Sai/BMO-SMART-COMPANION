from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bmo_contracts.models import HwCmd  # noqa: E402
from bmo_contracts.topics import HW_CMD  # noqa: E402
from brain.tools.actuator_tools import (  # noqa: E402
    ActuatorTools,
    load_limits,
    register_actuator_tools,
)
from brain.tools.registry import ToolRegistry  # noqa: E402
from tools.gen_limits_header import generate_limits_header  # noqa: E402

__all__ = [
    "test_expression_and_sound_via_dispatch",
    "test_limits_header_matches",
    "test_move_servo_clamping_and_publish",
    "test_move_servo_rate_limiting",
]


class DummyBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    def publish(self, topic: str, msg: Any) -> None:
        self.published.append((topic, msg))


def test_move_servo_clamping_and_publish() -> None:
    bus = DummyBus()
    limits = load_limits()
    service = ActuatorTools(bus=bus, limits=limits)
    res = service.move_servo(id="pan", angle=500.0)
    assert res["clamped"] is True
    assert res["angle"] == 60.0
    assert res["ok"] is True
    assert len(bus.published) == 1
    topic, msg = bus.published[0]
    assert topic == HW_CMD
    assert isinstance(msg, HwCmd)
    assert msg.action == "move_servo"
    assert msg.args["id"] == "pan"
    assert msg.args["angle"] == 60.0


def test_move_servo_rate_limiting() -> None:
    bus = DummyBus()
    limits = load_limits()
    fake_now = 0.0

    def fake_clock() -> float:
        return fake_now

    service = ActuatorTools(bus=bus, limits=limits, clock=fake_clock)
    success_count = 0
    rate_limited_count = 0

    for _ in range(20):
        res = service.move_servo(id="pan", angle=0.0)
        if res.get("ok"):
            success_count += 1
        elif res.get("error") == "rate_limited":
            rate_limited_count += 1
        fake_now += 0.05

    assert success_count == 4
    assert rate_limited_count == 16
    assert len(bus.published) == 4


def test_expression_and_sound_via_dispatch() -> None:
    bus = DummyBus()
    limits = load_limits()
    registry = ToolRegistry()
    register_actuator_tools(registry, bus=bus, limits=limits)

    ok_sound = registry.dispatch({"name": "play_sound", "arguments": {"sound": "chime"}})
    assert ok_sound.get("ok") is True

    fail_sound = registry.dispatch({"name": "play_sound", "arguments": {"sound": "laser"}})
    assert fail_sound.get("ok") is False

    valid_expr = limits.emotions[0] if limits.emotions else "happy"
    ok_expr = registry.dispatch({"name": "set_expression", "arguments": {"expression": valid_expr}})
    assert ok_expr.get("ok") is True

    fail_expr = registry.dispatch(
        {
            "name": "set_expression",
            "arguments": {"expression": "unknown_emotion_xyz"},
        }
    )
    assert fail_expr.get("ok") is False


def test_limits_header_matches() -> None:
    header = generate_limits_header()
    expected = Path("edge/esp32_firmware/limits.h").read_text()
    assert header == expected
