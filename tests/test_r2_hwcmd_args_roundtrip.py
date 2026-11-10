"""Test R2: HW_CMD args end-to-end round-trip (F4 fix)."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bmo_contracts.models import HwCmd  # noqa: E402
from brain.tools.actuator_tools import ActuatorTools, load_limits  # noqa: E402


class DummyBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, HwCmd]] = []

    def publish(self, topic: str, msg: HwCmd) -> None:
        self.published.append((topic, msg))


def test_set_expression_roundtrip_no_keyerror() -> None:
    """Producer sends set_expression, Pi parses without KeyError/extra_forbidden."""
    bus = DummyBus()
    limits = load_limits()
    service = ActuatorTools(bus=bus, limits=limits)

    # Valid emotion from limits
    valid_emotion = limits.emotions[0] if limits.emotions else "happy"
    res = service.set_expression(valid_emotion)
    assert res["ok"] is True

    assert len(bus.published) == 1
    _, msg = bus.published[0]
    assert msg.action == "set_expression"

    # Serialize and parse as Pi would
    json_str = msg.model_dump_json()
    parsed = HwCmd.model_validate_json(json_str)

    # Pi code expects args["emotion"] - should not raise KeyError
    assert "emotion" in parsed.args, f"Missing 'emotion' key in {parsed.args}"
    assert parsed.args["emotion"] == valid_emotion


def test_play_sound_roundtrip_no_keyerror() -> None:
    """Producer sends play_sound, Pi parses without KeyError/extra_forbidden."""
    bus = DummyBus()
    limits = load_limits()
    service = ActuatorTools(bus=bus, limits=limits)

    valid_sound = limits.sounds[0] if limits.sounds else "chime"
    res = service.play_sound(valid_sound)
    assert res["ok"] is True

    assert len(bus.published) == 1
    _, msg = bus.published[0]
    assert msg.action == "play_sound"

    # Serialize and parse as Pi would
    json_str = msg.model_dump_json()
    parsed = HwCmd.model_validate_json(json_str)

    # Pi code expects args["name"] - should not raise KeyError
    assert "name" in parsed.args, f"Missing 'name' key in {parsed.args}"
    assert parsed.args["name"] == valid_sound


def test_no_extra_keys_in_serialized_args() -> None:
    """Serialized args should not contain deprecated keys (expression, sound)."""
    bus = DummyBus()
    limits = load_limits()
    service = ActuatorTools(bus=bus, limits=limits)

    valid_emotion = limits.emotions[0] if limits.emotions else "happy"
    service.set_expression(valid_emotion)
    _, msg = bus.published[0]
    json_str = msg.model_dump_json()

    # Should not have deprecated 'expression' key
    assert '"expression"' not in json_str, f"Deprecated 'expression' key in JSON: {json_str}"

    bus.published.clear()
    valid_sound = limits.sounds[0] if limits.sounds else "chime"
    service.play_sound(valid_sound)
    _, msg = bus.published[0]
    json_str = msg.model_dump_json()

    # Should not have deprecated 'sound' key
    assert '"sound"' not in json_str, f"Deprecated 'sound' key in JSON: {json_str}"