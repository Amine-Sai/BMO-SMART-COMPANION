from __future__ import annotations

import asyncio
import inspect
import logging
import socket
import threading
import time

from bmo_contracts.models import VisionCmd, VisionCmdResult
from bmo_contracts.topics import TOPICS, VISION_CMD_RESULT
from brain.vision_client import VisionClient
import paho.mqtt.client as mqtt
import pytest
from tools.mock_cv import MockCV
from tools.simple_broker import SimpleMQTTBroker


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def broker():
    port = _find_free_port()
    sig = inspect.signature(SimpleMQTTBroker.__init__)
    kwargs = {}
    for name in sig.parameters:
        if name in ("port", "listen_port"):
            kwargs[name] = port
        elif name in ("host", "bind_address"):
            kwargs[name] = "127.0.0.1"
    b = SimpleMQTTBroker(**kwargs)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    asyncio.run_coroutine_threadsafe(b.start(), loop).result()
    actual_port = getattr(b, "port", None) or getattr(b, "listen_port", None) or port
    yield b, actual_port
    asyncio.run_coroutine_threadsafe(b.stop(), loop).result()
    loop.call_soon_threadsafe(loop.stop())
    t.join(timeout=2.0)


@pytest.fixture
def mock_cv_instance(broker):
    b, port = broker
    sig = inspect.signature(MockCV.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if name in ("port", "broker_port"):
            kwargs[name] = port
        elif name in ("host", "broker_host"):
            kwargs[name] = "127.0.0.1"
        elif name == "broker":
            kwargs[name] = b
        elif name in ("broker_url", "url"):
            kwargs[name] = f"mqtt://127.0.0.1:{port}"
        elif param.default is inspect.Parameter.empty:
            if "user" in name:
                kwargs[name] = "bmo"
            elif "pass" in name:
                kwargs[name] = "bmo"
    cv = MockCV(**kwargs)
    if hasattr(cv, "start"):
        cv.start()
    time.sleep(0.1)
    yield cv
    if hasattr(cv, "stop"):
        cv.stop()


@pytest.fixture
def mqtt_client(broker):
    _, port = broker
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test-vision-client")
    except (AttributeError, TypeError):
        c = mqtt.Client(client_id="test-vision-client")
    c.connect("127.0.0.1", port, keepalive=30)
    c.loop_start()
    time.sleep(0.05)
    yield c
    c.loop_stop()
    c.disconnect()


def test_vision_client_against_mock_cv(broker, mock_cv_instance, mqtt_client):
    _ = (broker, mock_cv_instance)
    client = mqtt_client
    vision = VisionClient(
        publish_fn=lambda topic, payload, qos, retain: client.publish(topic, payload, qos=qos, retain=retain)
    )
    client.on_message = lambda _c, _u, msg: vision.handle_mqtt_message(msg.topic, msg.payload)
    client.subscribe(VISION_CMD_RESULT, qos=TOPICS[VISION_CMD_RESULT].qos)
    time.sleep(0.1)

    res_enroll = vision.enroll("alice", shots=8, timeout_s=5.0)
    assert res_enroll.ok is True
    assert res_enroll.id.startswith("vcmd-")

    res_forget = vision.forget("alice", timeout_s=5.0)
    assert res_forget.ok is True

    res_reload = vision.reload_gallery(timeout_s=5.0)
    assert res_reload.ok is True

    res_modules = vision.set_modules(face=True, sign=False, objects=True, timeout_s=5.0)
    assert res_modules.ok is True


def test_failure_path():
    client_under_test: VisionClient | None = None

    def fake_publish(_topic: str, payload: bytes, _qos: int, _retain: bool) -> None:
        req = VisionCmd.model_validate_json(payload)
        err = VisionCmdResult(id=req.id, ok=False, detail="not_found")

        def reply() -> None:
            if client_under_test is not None:
                client_under_test.handle_mqtt_message(VISION_CMD_RESULT, err.model_dump_json().encode())

        threading.Thread(target=reply).start()

    client_under_test = VisionClient(publish_fn=fake_publish)
    res = client_under_test.forget("unknown_person", timeout_s=1.0)
    assert res.ok is False
    assert res.detail == "not_found"


def test_timeout_path():
    def fake_publish(_topic: str, _payload: bytes, _qos: int, _retain: bool) -> None:
        pass

    client_under_test = VisionClient(publish_fn=fake_publish)
    res = client_under_test.send_command("reload_gallery", timeout_s=0.15)
    assert res.ok is False
    assert res.detail == "timeout"
    assert len(client_under_test._pending) == 0


def test_mismatched_id_ignored():
    client_under_test: VisionClient | None = None

    def fake_publish(_topic: str, payload: bytes, _qos: int, _retain: bool) -> None:
        req = VisionCmd.model_validate_json(payload)

        def reply() -> None:
            time.sleep(0.02)
            if client_under_test is not None:
                client_under_test.handle_cmd_result(VisionCmdResult(id="other-id-9999", ok=True, detail="other"))
                time.sleep(0.02)
                client_under_test.handle_cmd_result(VisionCmdResult(id=req.id, ok=True, detail="success"))

        threading.Thread(target=reply).start()

    client_under_test = VisionClient(publish_fn=fake_publish)
    res = client_under_test.send_command("reload_gallery", timeout_s=1.0)
    assert res.ok is True
    assert res.detail == "success"


def test_publish_raise_cleans_pending():
    def broken_publish(_topic: str, _payload: bytes, _qos: int, _retain: bool) -> None:
        raise RuntimeError("network failure during publish")

    client = VisionClient(publish_fn=broken_publish)
    with pytest.raises(RuntimeError, match="network failure during publish"):
        client.send_command("reload_gallery", timeout_s=1.0)
    assert len(client._pending) == 0


def test_bad_payload_ignored(caplog: pytest.LogCaptureFixture):
    client = VisionClient(publish_fn=lambda *args: None)
    with caplog.at_level(logging.WARNING):
        client.handle_cmd_result(b"not valid json")
        client.handle_cmd_result(b'{"wrong": "field"}')
        client.handle_cmd_result("invalid string json")
        client.handle_mqtt_message(VISION_CMD_RESULT, b"corrupt payload")

    assert len(client._pending) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 4
    assert all("Failed to validate VisionCmdResult payload" in r.message for r in warnings)


def test_late_result_after_timeout_ignored(caplog: pytest.LogCaptureFixture):
    client = VisionClient(publish_fn=lambda *args: None)
    res = client.send_command("reload_gallery", timeout_s=0.05)
    assert res.ok is False
    assert res.detail == "timeout"
    assert len(client._pending) == 0

    late_result = VisionCmdResult(id=res.id, ok=True, detail="late arrival")
    with caplog.at_level(logging.DEBUG):
        client.handle_cmd_result(late_result)

    assert len(client._pending) == 0
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "unknown or expired" in r.message]
    assert len(debugs) == 1
    assert res.id in debugs[0].message

