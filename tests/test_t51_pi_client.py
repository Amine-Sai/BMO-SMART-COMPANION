"""Tests for T5.1 Pi edge client."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
import socket
import threading
import time
from typing import Any
from unittest.mock import patch

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import pytest

from bmo_contracts.models import (
    BrainSay,
    BrainStatus,
    Emotion,
    HwCmd,
    HwEvent,
    HwStatus,
    VisionStatus,
)
from bmo_contracts.topics import (
    BRAIN_SAY,
    BRAIN_STATUS,
    HW_CMD,
    HW_EVENT,
    HW_STATUS,
    VISION_STATUS,
)
from edge.esp32_protocol import parse_esp32_response
from edge.pi_client import PiClient, main
from tools.simple_broker import SimpleMQTTBroker, parse_passwords_file


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until(predicate: Callable[[], bool], timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@contextmanager
def _running_broker(port: int) -> Generator[SimpleMQTTBroker, None, None]:
    users = parse_passwords_file("infra/passwords.example")
    broker = SimpleMQTTBroker(
        host="127.0.0.1",
        port=port,
        require_auth=True,
        users=users,
        acl_path="infra/acl",
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


def test_pi_client_integration() -> None:
    port = _get_free_port()
    with _running_broker(port):
        pi_client = PiClient(host="127.0.0.1", port=port, sim=True)
        brain_client = mqtt.Client(CallbackAPIVersion.VERSION2)
        brain_client.username_pw_set("brain", "brain_insecure_dev_password")

        received_hw_status: list[HwStatus] = []
        received_hw_events: list[HwEvent] = []

        def on_brain_message(
            _client: mqtt.Client,
            _userdata: Any,
            msg: mqtt.MQTTMessage,
        ) -> None:
            payload = msg.payload.decode("utf-8")
            if msg.topic == HW_STATUS:
                received_hw_status.append(HwStatus.model_validate_json(payload))
            elif msg.topic == HW_EVENT:
                received_hw_events.append(HwEvent.model_validate_json(payload))

        brain_client.on_message = on_brain_message

        try:
            brain_client.connect("127.0.0.1", port, keepalive=60)
            brain_client.loop_start()
            brain_client.subscribe([(HW_STATUS, 1), (HW_EVENT, 1)])

            pi_client.start()

            # 1. Verify PiClient publishes HwStatus(online=True, ...) on HW_STATUS
            assert _wait_until(lambda: len(received_hw_status) > 0)
            assert received_hw_status[-1].online is True
            assert "servos" in received_hw_status[-1].capabilities

            # 2. Publish BrainSay from "brain" to BRAIN_SAY
            say_cmd = BrainSay(utt_id="u1", text="Hello friend!", emotion=Emotion.HAPPY)
            brain_client.publish(BRAIN_SAY, say_cmd.model_dump_json(), qos=1)

            assert _wait_until(lambda: len(pi_client.spoken_utterances) > 0)
            assert pi_client.current_emotion == Emotion.HAPPY
            assert pi_client.last_frame["mouth_open"] is True
            assert pi_client.spoken_utterances[-1].text == "Hello friend!"

            # 3. Publish HwCmd move_servo from "brain" to HW_CMD
            move_cmd = HwCmd(
                id="c1",
                target="esp32",
                action="move_servo",
                args={"id": "pan", "angle": 45.0},
            )
            brain_client.publish(HW_CMD, move_cmd.model_dump_json(), qos=1)

            assert _wait_until(lambda: len(pi_client.serial_log) > 0)
            assert ("MOVE pan 45.0\n", "OK pan 45.0\n") in pi_client.serial_log
            assert pi_client.esp32 is not None
            assert pi_client.esp32.servos["pan"].target_deg == 45.0

            # Validate ESP32 protocol parser
            parsed = parse_esp32_response(pi_client.serial_log[0][1])
            assert parsed is not None

            # 4. Publish mute event from pi_client
            pi_client.publish_mute_event(True)
            assert _wait_until(lambda: len(received_hw_events) > 0)
            assert received_hw_events[-1].kind == "mute_switch"
            assert received_hw_events[-1].value is True

            # Additional command coverage: set_expression & play_sound
            expr_cmd = HwCmd(
                id="c2",
                target="pi",
                action="set_expression",
                args={"emotion": "surprised"},
            )
            brain_client.publish(HW_CMD, expr_cmd.model_dump_json(), qos=1)
            assert _wait_until(lambda: pi_client.current_emotion == Emotion.SURPRISED)

            sound_cmd = HwCmd(
                id="c3",
                target="pi",
                action="play_sound",
                args={"name": "ping.wav"},
            )
            brain_client.publish(HW_CMD, sound_cmd.model_dump_json(), qos=1)
            assert _wait_until(lambda: "ping.wav" in pi_client.played_sounds)

            # BrainStatus & VisionStatus coverage
            b_status = BrainStatus(online=True, state="idle")
            brain_client.publish(BRAIN_STATUS, b_status.model_dump_json(), qos=1)
            assert _wait_until(lambda: pi_client.brain_status is not None)

            cv_client = mqtt.Client(CallbackAPIVersion.VERSION2)
            cv_client.username_pw_set("cv", "cv_insecure_dev_password")
            cv_client.connect("127.0.0.1", port, keepalive=60)
            cv_client.loop_start()
            try:
                v_status = VisionStatus(online=True)
                cv_client.publish(VISION_STATUS, v_status.model_dump_json(), qos=1)
                assert _wait_until(lambda: pi_client.vision_status is not None)
            finally:
                cv_client.loop_stop()
                cv_client.disconnect()

            # Attribute validation for vulture hygiene
            assert pi_client.host == "127.0.0.1"
            assert pi_client.port == port
            assert pi_client.sim is True
            assert pi_client.client is not None
            assert callable(pi_client._clock)

        finally:
            pi_client.stop()
            brain_client.loop_stop()
            brain_client.disconnect()


def test_cli_main_help() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_cli_main_execution() -> None:
    port = _get_free_port()
    with _running_broker(port):
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            main(["--host", "127.0.0.1", "--port", str(port)])
