from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import threading
import time

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import pytest

from bmo_contracts.topics import BRAIN_SAY, VISION_EVENTS, VISION_STATUS
from tools.gen_acl import generate_acl_text
from tools.simple_broker import SimpleMQTTBroker, parse_passwords_file


@pytest.fixture(
    scope="module",
    params=["simple_broker", pytest.param("mosquitto", marks=pytest.mark.broker)],
)
def broker_port(request: pytest.FixtureRequest) -> int:
    if request.param == "mosquitto" and shutil.which("mosquitto") is None:
        pytest.skip("mosquitto not installed")
    loop = asyncio.new_event_loop()
    broker = SimpleMQTTBroker(
        host="127.0.0.1",
        port=18892,
        require_auth=True,
        users=parse_passwords_file("infra/passwords.example"),
        acl_path="infra/acl",
    )

    async def _runner() -> None:
        await broker.start()
        try:
            await broker.server.serve_forever()
        except asyncio.CancelledError:
            pass

    threading.Thread(target=lambda: loop.run_until_complete(_runner()), daemon=True).start()
    time.sleep(0.15)
    yield 18892
    if broker.server is not None:
        loop.call_soon_threadsafe(broker.server.close)


def _connect_fails(port: int, **auth: str) -> None:
    c = mqtt.Client(CallbackAPIVersion.VERSION2)
    if "u" in auth:
        c.username_pw_set(auth["u"], auth.get("p", ""))
    codes: list[int] = []
    c.on_connect = lambda _c, _u, _f, rc, _p=None: codes.append(getattr(rc, "value", rc))
    try:
        c.connect("127.0.0.1", port, 5)
        c.loop_start()
        time.sleep(0.2)
    except Exception:
        pass
    finally:
        c.loop_stop()
    assert not codes or codes[0] != 0


def _client(u: str, p: str, port: int) -> mqtt.Client:
    c = mqtt.Client(CallbackAPIVersion.VERSION2)
    c.username_pw_set(u, p)
    c.connect("127.0.0.1", port, 10)
    c.loop_start()
    time.sleep(0.15)
    return c


def test_acl_matches_generator() -> None:
    assert Path("infra/acl").read_text("utf-8") == generate_acl_text()


def test_unauthenticated_connect_fails(broker_port: int) -> None:
    _connect_fails(broker_port)


def test_wrong_password_connect_fails(broker_port: int) -> None:
    _connect_fails(broker_port, u="cv", p="wrong_password")


def test_cv_publish_brain_receives_and_retained(broker_port: int) -> None:
    brain = _client("brain", "brain_insecure_dev_password", broker_port)
    recv: list[tuple[str, str]] = []
    brain.on_message = lambda _c, _u, m: recv.append((m.topic, m.payload.decode()))
    brain.subscribe(VISION_EVENTS)
    time.sleep(0.15)

    cv = _client("cv", "cv_insecure_dev_password", broker_port)
    cv.publish(VISION_EVENTS, "event_1")
    cv.publish(VISION_STATUS, "ok_status", retain=True)
    time.sleep(0.2)
    assert (VISION_EVENTS, "event_1") in recv

    brain2 = _client("brain", "brain_insecure_dev_password", broker_port)
    retained: list[tuple[str, str]] = []
    brain2.on_message = lambda _c, _u, m: retained.append((m.topic, m.payload.decode()))
    brain2.subscribe(VISION_STATUS)
    time.sleep(0.2)
    assert (VISION_STATUS, "ok_status") in retained
    for cl in (cv, brain, brain2):
        cl.loop_stop()


def test_brain_publish_vision_events_blocked(broker_port: int) -> None:
    sub = _client("brain", "brain_insecure_dev_password", broker_port)
    recv: list[str] = []
    sub.on_message = lambda _c, _u, m: recv.append(m.payload.decode())
    sub.subscribe(VISION_EVENTS)
    time.sleep(0.15)

    pub = _client("brain", "brain_insecure_dev_password", broker_port)
    pub.publish(VISION_EVENTS, "blocked")
    time.sleep(0.2)
    assert not recv
    pub.loop_stop()
    sub.loop_stop()


def test_face_ui_publish_brain_say_blocked(broker_port: int) -> None:
    sub = _client("face_ui", "face_ui_insecure_dev_password", broker_port)
    recv: list[str] = []
    sub.on_message = lambda _c, _u, m: recv.append(m.payload.decode())
    sub.subscribe(BRAIN_SAY)
    time.sleep(0.15)

    pub = _client("face_ui", "face_ui_insecure_dev_password", broker_port)
    pub.publish(BRAIN_SAY, "blocked")
    time.sleep(0.2)
    assert not recv
    pub.loop_stop()
    sub.loop_stop()
