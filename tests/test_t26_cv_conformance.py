from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt
import pytest

from assistant_cv.memory import ShortTermMemory
from assistant_cv.mqtt_bridge import MQTTBridge
from bmo_contracts.models import (
    EVENT_ADAPTER,
    Scene,
    VisionCmdResult,
    VisionStatus,
)
from bmo_contracts.topics import (
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
    VISION_STATUS,
)
from tools.simple_broker import SimpleMQTTBroker


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
    b = SimpleMQTTBroker(host="127.0.0.1", port=18897, require_auth=False)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    asyncio.run_coroutine_threadsafe(b.start(), loop).result()
    assert _wait_for_port(18897)
    yield b
    asyncio.run_coroutine_threadsafe(b.stop(), loop).result()
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1.0)


def test_t26_cv_conformance(broker: Any) -> None:
    _ = broker
    received: dict[str, list[bytes]] = {
        VISION_STATUS: [],
        VISION_EVENTS: [],
        VISION_CONTEXT: [],
        VISION_CMD_RESULT: [],
    }

    try:
        verifier = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="verifier")
    except (AttributeError, TypeError):
        verifier = mqtt.Client(client_id="verifier")

    def on_verifier_message(_client: Any, _userdata: Any, msg: Any) -> None:
        if msg.topic in received:
            received[msg.topic].append(msg.payload)

    verifier.on_message = on_verifier_message
    verifier.connect("127.0.0.1", 18897)
    verifier.subscribe(
        [
            (VISION_STATUS, 0),
            (VISION_EVENTS, 0),
            (VISION_CONTEXT, 0),
            (VISION_CMD_RESULT, 0),
        ]
    )
    verifier.loop_start()

    bridge = MQTTBridge(host="127.0.0.1", port=18897)
    bridge.start()

    deadline_conn = time.time() + 2.0
    while time.time() < deadline_conn and not bridge.connected:
        time.sleep(0.05)

    mem = ShortTermMemory(window_sec=60.0)
    mem.update(
        objects=[
            {
                "label": "person",
                "identity": "Jud",
                "id": 1,
                "bbox": (250, 60, 390, 420),
            },
            {
                "label": "cup",
                "bbox": (100, 100, 200, 200),
            },
        ]
    )

    try:
        # Publish scene contract
        bridge.publish_scene(mem.get_scene_contract())

        # Publish 4 events
        bridge.publish_event(
            "person_detected",
            {"track_id": 1, "pos": "center", "proximity": "near"},
        )
        bridge.publish_event(
            "face_identified",
            {"track_id": 1, "name": "Jud", "conf": 0.95},
        )
        bridge.publish_event(
            "sign_recognized",
            {"sign": "wave", "conf": 0.9},
        )
        bridge.publish_event(
            "object_seen",
            {"name": "cup", "conf": 0.85},
        )

        # Publish cmd result
        bridge.publish_cmd_result("c-1", True, "ok")

        deadline = time.time() + 3.0
        while time.time() < deadline:
            if (
                len(received[VISION_STATUS]) > 0
                and len(received[VISION_EVENTS]) >= 4
                and len(received[VISION_CONTEXT]) > 0
                and len(received[VISION_CMD_RESULT]) > 0
            ):
                break
            time.sleep(0.05)

        # Assert VISION_STATUS validates
        assert len(received[VISION_STATUS]) > 0, "No status message received"
        for payload in received[VISION_STATUS]:
            st = VisionStatus.model_validate_json(payload)
            assert isinstance(st, VisionStatus)

        # Assert VISION_EVENTS validates cleanly against EVENT_ADAPTER
        assert len(received[VISION_EVENTS]) >= 4, f"Expected >= 4 events, got {len(received[VISION_EVENTS])}"
        for payload in received[VISION_EVENTS]:
            evt = EVENT_ADAPTER.validate_json(payload)
            assert evt is not None

        # Assert VISION_CONTEXT validates cleanly against Scene
        assert len(received[VISION_CONTEXT]) > 0, "No context message received"
        for payload in received[VISION_CONTEXT]:
            sc = Scene.model_validate_json(payload)
            assert isinstance(sc, Scene)
            assert any(p.name == "Jud" for p in sc.people)
            assert "cup" in sc.objects

        # Assert VISION_CMD_RESULT validates cleanly against VisionCmdResult
        assert len(received[VISION_CMD_RESULT]) > 0, "No cmd result message received"
        for payload in received[VISION_CMD_RESULT]:
            res = VisionCmdResult.model_validate_json(payload)
            assert isinstance(res, VisionCmdResult)
            assert res.id == "c-1"
            assert res.ok is True

    finally:
        verifier.loop_stop()
        verifier.disconnect()
        bridge.stop()
