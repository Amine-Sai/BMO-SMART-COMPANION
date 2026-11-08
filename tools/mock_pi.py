"""Mock Pi hardware peer for testing and local simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Callable

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paho.mqtt.client import CallbackAPIVersion, Client, MQTTMessage

from bmo_contracts.models import (
    BrainSay,
    BrainStatus,
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
from tools.mock_cv import make_sample


class MockPi:
    """Mock Pi peer implementing hardware/audio MQTT interface."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        username: str = "pi",
        password: str = "pi_insecure_dev_password",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.clock = clock
        self.client = Client(CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.say_messages: list[BrainSay] = []
        self.hw_commands: list[HwCmd] = []
        self.brain_statuses: list[BrainStatus] = []
        self.vision_statuses: list[VisionStatus] = []

    def _on_connect(
        self,
        client: Client,
        _userdata: object,
        _flags: object,
        _rc: object,
        _properties: object = None,
    ) -> None:
        client.subscribe(
            [
                (BRAIN_SAY, 0),
                (HW_CMD, 0),
                (BRAIN_STATUS, 0),
                (VISION_STATUS, 0),
            ]
        )
        status = HwStatus(
            online=True,
            ts=self.clock(),
            capabilities=["lcd", "speaker", "mic", "servos"],
        )
        client.publish(HW_STATUS, status.model_dump_json(), retain=True)

    def _on_message(
        self,
        _client: Client,
        _userdata: object,
        msg: MQTTMessage,
    ) -> None:
        if msg.topic == BRAIN_SAY:
            self.say_messages.append(BrainSay.model_validate_json(msg.payload))
        elif msg.topic == HW_CMD:
            self.hw_commands.append(HwCmd.model_validate_json(msg.payload))
        elif msg.topic == BRAIN_STATUS:
            self.brain_statuses.append(BrainStatus.model_validate_json(msg.payload))
        elif msg.topic == VISION_STATUS:
            self.vision_statuses.append(VisionStatus.model_validate_json(msg.payload))

    def start(self) -> None:
        self.client.connect(self.host, self.port)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def publish_hw_event(self, event: HwEvent) -> None:
        payload = event.model_dump_json() if hasattr(event, "model_dump_json") else str(event)
        self.client.publish(HW_EVENT, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Pi peer for BMO")
    parser.add_argument("--host", default="127.0.0.1", help="Broker host")
    parser.add_argument("--port", type=int, default=1883, help="Broker port")
    parser.add_argument(
        "--scenario",
        choices=["mute_toggle", "heartbeat"],
        default="mute_toggle",
        help="Scenario to run",
    )
    args = parser.parse_args()

    mock = MockPi(host=args.host, port=args.port)
    mock.start()
    try:
        time.sleep(0.5)
        if args.scenario == "mute_toggle":
            sample_event = make_sample(HwEvent)
            if sample_event is not None:
                mock.publish_hw_event(sample_event)
        elif args.scenario == "heartbeat":
            time.sleep(1.0)
    finally:
        mock.stop()
