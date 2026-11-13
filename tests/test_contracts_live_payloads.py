"""Contract tests with real producer payloads (no mock_cv).

Validates that:
1. MQTTBridge publishes Scene and VisionCmdResult conforming to bmo_contracts models
2. ActuatorTools produces HwCmd payloads that PiClient can parse
3. PiClient._on_message handles real payloads correctly
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from paho.mqtt.client import CallbackAPIVersion, Client as MQTTClient, MQTTMessage

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant_cv.mqtt_bridge import MQTTBridge  # noqa: E402
from bmo_contracts.models import (  # noqa: E402
    Emotion,
    HwCmd,
    Scene,
    ScenePerson,
    VisionCmdResult,
    VisionStatus,
    parse_event_json,
    EVENT_ADAPTER,
)
from bmo_contracts.topics import (  # noqa: E402
    HW_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_STATUS,
    BRAIN_SAY,
    BRAIN_STATUS,
)
from brain.tools.actuator_tools import ActuatorTools, load_limits  # noqa: E402
from edge.pi_client import PiClient  # noqa: E402
from tools.simple_broker import SimpleMQTTBroker, parse_passwords_file  # noqa: E402


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@contextmanager
def _running_broker(port: int, require_auth: bool = True) -> Generator[SimpleMQTTBroker, None, None]:
    users = parse_passwords_file("infra/passwords.example") if require_auth else {}
    broker = SimpleMQTTBroker(
        host="127.0.0.1",
        port=port,
        require_auth=require_auth,
        users=users,
        acl_path="infra/acl" if require_auth else None,
    )
    loop = asyncio.new_event_loop()

    def run_broker() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broker.start())
        loop.run_forever()

    thread = threading.Thread(target=run_broker, daemon=True)
    thread.start()
    time.sleep(0.3)
    try:
        yield broker
    finally:
        try:
            asyncio.run_coroutine_threadsafe(broker.stop(), loop).result(timeout=2.0)
        finally:
            loop.call_soon_threadsafe(loop.stop)


class DummyBus:
    """Capture bus publications for testing."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    def publish(self, topic: str, msg: Any) -> None:
        self.published.append((topic, msg))


def test_mqttbridge_publish_scene_validates_against_contract() -> None:
    """MQTTBridge.publish_scene() validates payload against Scene model."""
    bridge = MQTTBridge(clock=lambda: 1_700_000_000.0)
    bridge.connected = True
    bridge.client = MagicMock()
    published: list[tuple[str, bytes, int, bool]] = []

    def mock_pub(topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        published.append((topic, payload, qos, retain))

    bridge.client.publish = mock_pub

    # Valid scene dict
    valid_scene = {
        "v": 1,
        "ts": 1_700_000_000.0,
        "people": [
            {"track_id": 1, "name": "Finn", "pos": "center", "proximity": "near", "since_s": 10.0}
        ],
        "objects": ["cup", "book"],
        "recent": ["face_identified:Finn", "object_seen:cup"],
    }

    # Should not raise
    bridge.publish_scene(valid_scene)

    # Verify published
    assert len(published) == 1
    topic, payload, qos, retain = published[0]
    assert topic == VISION_CONTEXT
    assert qos == 0
    assert retain is True

    # Validate against Scene model
    validated_scene = Scene.model_validate_json(payload)
    assert validated_scene.v == 1
    assert len(validated_scene.people) == 1
    assert validated_scene.people[0].name == "Finn"
    assert validated_scene.objects == ["cup", "book"]
    assert "face_identified:Finn" in validated_scene.recent

    # Invalid scene (missing required field) - should log error but not crash
    published.clear()
    invalid_scene = {
        "v": 1,
        "ts": 1_700_000_000.0,
        "people": [],
        "objects": [],
        "recent": [],
        "extra_field": "disallowed",  # Extra field should be rejected
    }
    bridge.publish_scene(invalid_scene)
    # Should not publish invalid scene (extra fields forbidden)
    assert len(published) == 0


def test_mqttbridge_publish_cmd_result_validates_against_contract() -> None:
    """MQTTBridge.publish_cmd_result() validates payload against VisionCmdResult model."""
    bridge = MQTTBridge(clock=lambda: 1_700_000_000.0)
    bridge.connected = True
    bridge.client = MagicMock()
    published: list[tuple[str, bytes, int, bool]] = []

    def mock_pub(topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        published.append((topic, payload, qos, retain))

    bridge.client.publish = mock_pub

    # Should not raise
    bridge.publish_cmd_result(cmd_id="cmd-123", ok=True, detail="ack:enroll")

    # Verify published
    assert len(published) == 1
    topic, payload, qos, retain = published[0]
    assert topic == VISION_CMD_RESULT
    assert qos == 1

    # Validate against VisionCmdResult model
    validated_result = VisionCmdResult.model_validate_json(payload)
    assert validated_result.v == 1
    assert validated_result.id == "cmd-123"
    assert validated_result.ok is True
    assert validated_result.detail == "ack:enroll"

    # Failed result
    published.clear()
    bridge.publish_cmd_result(cmd_id="cmd-456", ok=False, detail="enroll failed: no face")
    validated_fail = VisionCmdResult.model_validate_json(published[0][1])
    assert validated_fail.ok is False
    assert "enroll failed" in validated_fail.detail


def test_actuator_tools_set_expression_produces_valid_hwcmd() -> None:
    """ActuatorTools.set_expression() produces HwCmd that parses on Pi side."""
    bus = DummyBus()
    limits = load_limits()
    service = ActuatorTools(bus=bus, limits=limits, clock=lambda: 1_700_000_000.0)

    # Valid expression from limits
    for expr in limits.emotions:
        result = service.set_expression(expr)
        assert result["ok"] is True
        assert result["expression"] == expr

        # Verify HwCmd was published
        assert len(bus.published) > 0
        topic, msg = bus.published[-1]
        assert topic == HW_CMD
        assert isinstance(msg, HwCmd)
        assert msg.action == "set_expression"
        assert msg.args["emotion"] == expr
        assert msg.target == "esp32"

        # Validate HwCmd model
        hwcmd = HwCmd.model_validate(msg.model_dump())
        assert hwcmd.action == "set_expression"
        assert hwcmd.args["emotion"] == expr

    # Invalid expression
    result = service.set_expression("unknown_emotion_xyz")
    assert result["ok"] is False
    assert "unknown expression" in result["error"]


def test_actuator_tools_play_sound_produces_valid_hwcmd() -> None:
    """ActuatorTools.play_sound() produces HwCmd that parses on Pi side."""
    bus = DummyBus()
    limits = load_limits()
    service = ActuatorTools(bus=bus, limits=limits, clock=lambda: 1_700_000_000.0)

    # Valid sounds from limits
    for sound in limits.sounds:
        result = service.play_sound(sound)
        assert result["ok"] is True
        assert result["sound"] == sound

        # Verify HwCmd was published
        assert len(bus.published) > 0
        topic, msg = bus.published[-1]
        assert topic == HW_CMD
        assert isinstance(msg, HwCmd)
        assert msg.action == "play_sound"
        assert msg.args["name"] == sound
        assert msg.target == "esp32"

        # Validate HwCmd model
        hwcmd = HwCmd.model_validate(msg.model_dump())
        assert hwcmd.action == "play_sound"
        assert hwcmd.args["name"] == sound

    # Invalid sound
    result = service.play_sound("laser")
    assert result["ok"] is False
    assert "unknown sound" in result["error"]


def test_pi_client_on_message_handles_real_hwcmd_set_expression() -> None:
    """PiClient._on_message parses real HwCmd set_expression payload."""
    pi = PiClient(sim=True)
    pi.client = MQTTClient(CallbackAPIVersion.VERSION2)

    # Build a real HwCmd from ActuatorTools path
    hwcmd = HwCmd(
        id="expr-happy-123456",
        target="pi",
        action="set_expression",
        args={"emotion": "happy"},
    )
    payload_bytes = hwcmd.model_dump_json().encode("utf-8")

    # Create mock message
    msg = MagicMock(spec=MQTTMessage)
    msg.topic = HW_CMD
    msg.payload = payload_bytes

    # Call handler
    pi._on_message(pi.client, None, msg)

    # Verify state updated
    assert pi.current_emotion == Emotion.HAPPY
    assert pi.last_frame["emotion"] == "happy"
    assert pi.last_frame["mouth_open"] is False


def test_pi_client_on_message_handles_real_hwcmd_play_sound() -> None:
    """PiClient._on_message parses real HwCmd play_sound payload."""
    pi = PiClient(sim=True)
    pi.client = MQTTClient(CallbackAPIVersion.VERSION2)

    hwcmd = HwCmd(
        id="sound-chime-123456",
        target="pi",
        action="play_sound",
        args={"name": "chime"},
    )
    payload_bytes = hwcmd.model_dump_json().encode("utf-8")

    msg = MagicMock(spec=MQTTMessage)
    msg.topic = HW_CMD
    msg.payload = payload_bytes

    pi._on_message(pi.client, None, msg)

    assert "chime" in pi.played_sounds


def test_pi_client_on_message_handles_real_hwcmd_move_servo() -> None:
    """PiClient._on_message parses real HwCmd move_servo payload."""
    pi = PiClient(sim=True)
    pi.client = MQTTClient(CallbackAPIVersion.VERSION2)

    hwcmd = HwCmd(
        id="servo-pan-123456",
        target="esp32",
        action="move_servo",
        args={"id": "pan", "angle": 45.0},
    )
    payload_bytes = hwcmd.model_dump_json().encode("utf-8")

    msg = MagicMock(spec=MQTTMessage)
    msg.topic = HW_CMD
    msg.payload = payload_bytes

    pi._on_message(pi.client, None, msg)

    assert len(pi.serial_log) == 1
    cmd, reply = pi.serial_log[0]
    assert "MOVE pan 45.0" in cmd
    assert "OK pan 45.0" in reply


def test_pi_client_on_message_handles_brain_say() -> None:
    """PiClient._on_message parses real BrainSay payload."""
    pi = PiClient(sim=True)
    pi.client = MQTTClient(CallbackAPIVersion.VERSION2)

    from bmo_contracts.models import BrainSay

    say = BrainSay(utt_id="utt-1", text="Hello!", emotion=Emotion.HAPPY)
    payload_bytes = say.model_dump_json().encode("utf-8")

    msg = MagicMock(spec=MQTTMessage)
    msg.topic = BRAIN_SAY
    msg.payload = payload_bytes

    pi._on_message(pi.client, None, msg)

    assert pi.current_emotion == Emotion.HAPPY
    assert len(pi.spoken_utterances) == 1
    assert pi.spoken_utterances[0].text == "Hello!"
    assert pi.last_frame["mouth_open"] is True


def test_pi_client_on_message_handles_brain_status() -> None:
    """PiClient._on_message parses real BrainStatus payload."""
    pi = PiClient(sim=True)
    pi.client = MQTTClient(CallbackAPIVersion.VERSION2)

    from bmo_contracts.models import BrainStatus

    status = BrainStatus(online=True, state="listening")
    payload_bytes = status.model_dump_json().encode("utf-8")

    msg = MagicMock(spec=MQTTMessage)
    msg.topic = BRAIN_STATUS
    msg.payload = payload_bytes

    pi._on_message(pi.client, None, msg)

    assert pi.brain_status is not None
    assert pi.brain_status.state == "listening"
    assert pi.brain_status.online is True


def test_pi_client_on_message_handles_vision_status() -> None:
    """PiClient._on_message parses real VisionStatus payload."""
    pi = PiClient(sim=True)
    pi.client = MQTTClient(CallbackAPIVersion.VERSION2)

    status = VisionStatus(online=True)
    payload_bytes = status.model_dump_json().encode("utf-8")

    msg = MagicMock(spec=MQTTMessage)
    msg.topic = VISION_STATUS
    msg.payload = payload_bytes

    pi._on_message(pi.client, None, msg)

    assert pi.vision_status is not None
    assert pi.vision_status.online is True


def test_full_integration_mqttbridge_to_piclient_via_broker() -> None:
    """End-to-end: MQTTBridge publishes -> broker -> PiClient receives and parses."""
    port = _get_free_port()
    with _running_broker(port, require_auth=False):
        # Start PiClient (will send credentials but broker ignores them)
        pi = PiClient(host="127.0.0.1", port=port, sim=True)
        pi.start()

        # Start MQTTBridge (as CV would) - no auth support yet
        bridge = MQTTBridge(host="127.0.0.1", port=port)
        bridge.start()

        # Wait for connections
        time.sleep(0.5)
        assert _wait_until(lambda: pi.vision_status is not None and pi.vision_status.online)
        assert _wait_until(lambda: bridge.connected)

        # 1. Test Scene publish via bridge
        scene = Scene(
            v=1,
            ts=time.time(),
            people=[ScenePerson(track_id=1, name="Test", pos="center", proximity="near", since_s=0.0)],
            objects=["test_obj"],
            recent=[],
        )
        bridge.publish_scene(scene.model_dump())

        # 2. Test VisionCmdResult publish
        bridge.publish_cmd_result("cmd-e2e-1", True, "ack:test")

        # 3. Test HwCmd from brain -> PiClient (simulate brain publishing)
        brain_client = MQTTClient(CallbackAPIVersion.VERSION2)
        brain_client.username_pw_set("brain", "brain_insecure_dev_password")
        brain_client.connect("127.0.0.1", port, keepalive=60)
        brain_client.loop_start()

        # set_expression
        hwcmd_expr = HwCmd(
            id="expr-e2e-1",
            target="pi",
            action="set_expression",
            args={"emotion": "surprised"},
        )
        brain_client.publish(HW_CMD, hwcmd_expr.model_dump_json(), qos=1)
        assert _wait_until(lambda: pi.current_emotion == Emotion.SURPRISED)

        # play_sound
        hwcmd_sound = HwCmd(
            id="sound-e2e-1",
            target="pi",
            action="play_sound",
            args={"name": "beep"},
        )
        brain_client.publish(HW_CMD, hwcmd_sound.model_dump_json(), qos=1)
        assert _wait_until(lambda: "beep" in pi.played_sounds)

        # move_servo
        hwcmd_servo = HwCmd(
            id="servo-e2e-1",
            target="esp32",
            action="move_servo",
            args={"id": "tilt", "angle": 30.0},
        )
        brain_client.publish(HW_CMD, hwcmd_servo.model_dump_json(), qos=1)
        assert _wait_until(lambda: len(pi.serial_log) > 0)

        # Cleanup
        brain_client.loop_stop()
        brain_client.disconnect()
        bridge.stop()
        pi.stop()


def test_event_parsing_via_event_adapter() -> None:
    """Vision events parse correctly through EVENT_ADAPTER."""
    from bmo_contracts.models import (
        FaceIdentified,
        FaceData,
        PersonDetected,
        PersonData,
        ObjectSeen,
        ObjectData,
    )

    # FaceIdentified
    face = FaceIdentified(
        id="evt-1",
        ts=time.time(),
        data=FaceData(track_id=1, name="Finn", conf=0.95),
    )
    parsed = EVENT_ADAPTER.validate_python(face.model_dump())
    assert isinstance(parsed, FaceIdentified)
    assert parsed.data.name == "Finn"

    # PersonDetected
    person = PersonDetected(
        id="evt-2",
        ts=time.time(),
        data=PersonData(track_id=2, pos="left", proximity="medium"),
    )
    parsed = EVENT_ADAPTER.validate_python(person.model_dump())
    assert isinstance(parsed, PersonDetected)
    assert parsed.data.pos == "left"

    # ObjectSeen
    obj = ObjectSeen(
        id="evt-4",
        ts=time.time(),
        data=ObjectData(label="cup", conf=0.85),
    )
    parsed = EVENT_ADAPTER.validate_python(obj.model_dump())
    assert isinstance(parsed, ObjectSeen)
    assert parsed.data.label == "cup"

    # Parse from JSON bytes
    json_bytes = face.model_dump_json().encode()
    parsed_json = parse_event_json(json_bytes)
    assert isinstance(parsed_json, FaceIdentified)
    assert parsed_json.data.name == "Finn"


def test_hwcmd_args_validation_coerces_correctly() -> None:
    """HwCmd model validator coerces args correctly for each action type."""
    # set_expression
    cmd_expr = HwCmd(
        id="1",
        target="pi",
        action="set_expression",
        args={"emotion": "happy"},
    )
    assert cmd_expr.args["emotion"] == "happy"

    # move_servo
    cmd_servo = HwCmd(
        id="2",
        target="esp32",
        action="move_servo",
        args={"id": "pan", "angle": 45.0},
    )
    assert cmd_servo.args["id"] == "pan"
    assert cmd_servo.args["angle"] == 45.0

    # play_sound
    cmd_sound = HwCmd(
        id="3",
        target="pi",
        action="play_sound",
        args={"name": "chime"},
    )
    assert cmd_sound.args["name"] == "chime"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])