from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt
import pytest

from bmo_contracts.models import Emotion, Scene, ScenePerson
from bmo_contracts.topics import (
    HW_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_STATUS,
)
from brain.llm import ChatChunk, FakeLLMBackend, LLMClient, ToolCall
from brain.prompt import PromptBuilder
from brain.scene import SceneCache
from brain.settings import load_settings

try:
    from brain.tools.actuator_tools import (
        ActuatorController,
        register_actuator_tools,
    )
except ImportError:
    from brain.tools.actuator_tools import (  # type: ignore[attr-defined]
        ActuatorTools as ActuatorController,
        register_actuator_tools,
    )

from brain.tools.identity_tools import register_identity_tools
from brain.tools.registry import ToolRegistry
from brain.tools.scene_tools import register_scene_tools
from brain.vision_client import VisionClient
from tools.mock_cv import MockCV
from tools.mock_pi import MockPi
from tools.simple_broker import SimpleMQTTBroker, parse_passwords_file


def _wait_for_port(port: int, timeout: float = 3.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


@pytest.fixture
def broker():
    b = SimpleMQTTBroker(
        host="127.0.0.1",
        port=18896,
        require_auth=True,
        users=parse_passwords_file("infra/passwords.example"),
        acl_path="infra/acl",
    )
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    asyncio.run_coroutine_threadsafe(b.start(), loop).result()
    assert _wait_for_port(18896)
    yield b
    asyncio.run_coroutine_threadsafe(b.stop(), loop).result()
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1.0)


class _MQTTBus:
    def __init__(self, client: mqtt.Client) -> None:
        self.client = client

    def publish(self, topic: str, msg: Any) -> None:
        payload = msg.model_dump_json() if hasattr(msg, "model_dump_json") else str(msg)
        self.client.publish(topic, payload)


def test_t26_integration_checkpoint_a(broker: Any) -> None:
    _ = broker
    settings = load_settings()
    mock_cv = MockCV(host="127.0.0.1", port=18896)
    mock_cv.start()

    mock_pi = MockPi(host="127.0.0.1", port=18896)
    mock_pi.start()

    scene_cache = SceneCache(settings.scene)

    try:
        brain_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="brain")
    except (AttributeError, TypeError):
        brain_client = mqtt.Client(client_id="brain")

    brain_client.username_pw_set("brain", "brain_insecure_dev_password")

    def publish_fn(topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        brain_client.publish(topic, payload, qos=qos, retain=retain)

    vision_client = VisionClient(publish_fn=publish_fn)
    bus = _MQTTBus(brain_client)
    _ = ActuatorController

    def on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        payload = msg.payload
        for handler in [
            scene_cache.handle_mqtt_message,
            vision_client.handle_mqtt_message,
        ]:
            try:
                handler(msg.topic, payload)
            except TypeError:
                handler(msg.topic, payload.decode("utf-8"))

    brain_client.on_message = on_message
    brain_client.connect("127.0.0.1", 18896)
    brain_client.subscribe([(VISION_STATUS, 0), (VISION_CONTEXT, 0), (VISION_CMD_RESULT, 0)])
    brain_client.loop_start()

    try:
        tool_registry = ToolRegistry()
        register_scene_tools(tool_registry, scene_cache)
        register_identity_tools(tool_registry, vision_client)
        register_actuator_tools(tool_registry, bus=bus)

        prompt_builder = PromptBuilder(scene_cache=scene_cache)

        chunk1 = ChatChunk(tool_call=ToolCall(name="list_people", arguments={}))
        chunk2 = ChatChunk(text="Jud is here.")
        backend = FakeLLMBackend(chunks=[chunk1])
        llm_client = LLMClient(settings.llm, backend=backend)

        scene = Scene(
            ts=100.0,
            people=[
                ScenePerson(
                    track_id=1,
                    name="Jud",
                    pos="center",
                    proximity="near",
                    since_s=15.0,
                )
            ],
            objects=["keys", "cup"],
            last_sign=None,
            recent=["Jud arrived"],
        )

        mock_cv.publish_scene(scene)

        deadline = time.time() + 2.0
        while time.time() < deadline:
            sc = scene_cache.current
            if sc is not None and getattr(sc, "people", None):
                break
            time.sleep(0.05)

        # 1. "Who is here?" -> ToolCall(name="list_people", arguments={})
        messages = prompt_builder.build_messages("Who is here?")

        first_call = list(llm_client.chat(messages))
        assert len(first_call) >= 1
        assert first_call[0].tool_call is not None
        assert first_call[0].tool_call.name == "list_people"

        people_res = tool_registry.dispatch(first_call[0].tool_call)
        assert "Jud" in str(people_res)

        backend.chunks = [chunk2]
        second_call = list(
            llm_client.chat(
                [
                    {"role": "user", "content": "Who is here?"},
                    {"role": "tool", "content": str(people_res)},
                ]
            )
        )
        assert len(second_call) >= 1
        reply_text = getattr(second_call[0], "text", "")
        assert "Jud is here." in reply_text

        # 2. "Do you see my keys?" -> ToolCall(name="find_object", arguments={"name": "keys"})
        keys_call = ToolCall(name="find_object", arguments={"name": "keys"})
        keys_res = tool_registry.dispatch(keys_call)
        assert isinstance(keys_res, dict)
        assert keys_res.get("ok") is True
        assert keys_res.get("found") is True

        # 3. "Remember me as Jud" -> ToolCall(name="remember_person", arguments={"name": "Jud"})
        enroll_call = ToolCall(name="remember_person", arguments={"name": "Jud"})
        enroll_res = tool_registry.dispatch(enroll_call)
        assert isinstance(enroll_res, dict)
        assert enroll_res.get("ok") is True
        assert any(getattr(c, "cmd", None) == "enroll" for c in mock_cv.received_cmds)

        # 4. "Forget Bob" -> ToolCall(name="forget_person", arguments={"name": "Bob"})
        forget_call = ToolCall(name="forget_person", arguments={"name": "Bob"})
        forget_unconfirmed = tool_registry.dispatch(forget_call)
        assert isinstance(forget_unconfirmed, dict)
        assert forget_unconfirmed.get("needs_confirmation") is True
        cmds_before = [c for c in mock_cv.received_cmds if getattr(c, "cmd", None) == "forget"]
        assert len(cmds_before) == 0

        forget_confirmed = tool_registry.dispatch(forget_call, confirmed=True)
        assert isinstance(forget_confirmed, dict)
        assert forget_confirmed.get("ok") is True
        cmds_after = [c for c in mock_cv.received_cmds if getattr(c, "cmd", None) == "forget"]
        assert len(cmds_after) == 1
        assert getattr(cmds_after[0], "args", {}).get("name") == "Bob"

        # 5. "Look left and smile" -> move_servo(id="pan", angle=-30) and set_expression(emotion="happy")
        res_servo = tool_registry.dispatch(ToolCall(name="move_servo", arguments={"id": "pan", "angle": -30}))
        assert res_servo.get("ok", True) is not False

        res_expr = tool_registry.dispatch(ToolCall(name="set_expression", arguments={"emotion": Emotion.HAPPY.value}))
        assert res_expr.get("ok", True) is not False

        deadline = time.time() + 2.0
        while time.time() < deadline and len(mock_pi.hw_commands) < 2:
            time.sleep(0.05)

        assert len(mock_pi.hw_commands) >= 2
        cmds_str = [str(c) for c in mock_pi.hw_commands]
        assert any("pan" in s and "-30" in s for s in cmds_str)
        assert any(getattr(c, "emotion", None) == Emotion.HAPPY or "happy" in str(c) for c in mock_pi.hw_commands)
        assert any(getattr(c, "topic", None) == HW_CMD or "pan" in str(c) for c in mock_pi.hw_commands)

    finally:
        brain_client.loop_stop()
        brain_client.disconnect()
        mock_cv.stop()
        mock_pi.stop()
