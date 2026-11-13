"""Mock CV peer for testing and local simulation."""

from __future__ import annotations

import argparse
from enum import Enum
import time
from typing import Any, Literal, Union, get_args, get_origin

from paho.mqtt.client import CallbackAPIVersion, Client, MQTTMessage
from pydantic import BaseModel

from bmo_contracts.models import (
    FaceData,
    FaceIdentified,
    Scene,
    VisionCmd,
    VisionCmdResult,
    VisionStatus,
)
from bmo_contracts.topics import (
    VISION_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
    VISION_STATUS,
)


def make_sample(cls: Any) -> Any:
    """Generate a valid sample instance of a Pydantic model for tests and CLI."""
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        return None
    kwargs: dict[str, Any] = {}
    for name, field in cls.model_fields.items():
        if not field.is_required():
            continue
        kwargs[name] = _sample_value(name, field.annotation)
    return cls(**kwargs)


def _sample_value(name: str, ann: Any) -> Any:
    origin = get_origin(ann)
    if origin is Literal:
        return get_args(ann)[0]
    if origin is Union:
        args = [a for a in get_args(ann) if a is not type(None)]
        if args:
            return _sample_value(name, args[0])
        return None
    if origin is list:
        elem_cls = get_args(ann)[0]
        if isinstance(elem_cls, type) and issubclass(elem_cls, BaseModel):
            return [make_sample(elem_cls)]
        if elem_cls is str:
            if name == "capabilities":
                return ["lcd", "speaker", "mic", "servos"]
            return ["item1"]
        if elem_cls is int:
            return [0, 0, 10, 10]
        if elem_cls is float:
            return [0.0, 0.0, 1.0, 1.0]
        return []
    if origin is dict:
        return {}
    if isinstance(ann, type) and issubclass(ann, Enum):
        return list(ann)[0]
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return make_sample(ann)
    if ann is str:
        if name == "state":
            return "idle"
        if name == "cmd":
            return "ping"
        if name == "id":
            return "test-id-1"
        if name == "text":
            return "hello world"
        if name == "detail":
            return "ack:ping"
        if name == "name":
            return "Finn"
        if name == "label":
            return "cup"
        if name == "person_id":
            return "p-1"
        if name == "event":
            return "button_press"
        return "test"
    if ann is float:
        if name == "confidence":
            return 0.95
        if name == "ts":
            return 1700000000.0
        return 1.0
    if ann is int:
        return 1
    if ann is bool:
        return True
    return "test"


class MockCV:
    """Mock CV peer implementing camera/vision MQTT interface."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        username: str = "cv",
        password: str = "cv_insecure_dev_password",
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = Client(CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.received_cmds: list[VisionCmd] = []

    def _on_connect(
        self,
        client: Client,
        _userdata: object,
        _flags: object,
        _rc: object,
        _properties: object = None,
    ) -> None:
        client.subscribe(VISION_CMD)
        status = VisionStatus(online=True)
        client.publish(VISION_STATUS, status.model_dump_json(), retain=True)

    def _on_message(
        self,
        _client: Client,
        _userdata: object,
        msg: MQTTMessage,
    ) -> None:
        if msg.topic == VISION_CMD:
            cmd = VisionCmd.model_validate_json(msg.payload)
            self.received_cmds.append(cmd)
            result = VisionCmdResult(id=cmd.id, ok=True, detail=f"ack:{cmd.cmd}")
            self.client.publish(VISION_CMD_RESULT, result.model_dump_json())

    def start(self) -> None:
        self.client.connect(self.host, self.port)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def publish_event(self, event: Any) -> None:
        payload = event.model_dump_json() if hasattr(event, "model_dump_json") else str(event)
        self.client.publish(VISION_EVENTS, payload)

    def publish_scene(self, scene: Scene) -> None:
        self.client.publish(VISION_CONTEXT, scene.model_dump_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock CV peer for BMO")
    parser.add_argument("--host", default="127.0.0.1", help="Broker host")
    parser.add_argument("--port", type=int, default=1883, help="Broker port")
    parser.add_argument(
        "--scenario",
        choices=["greeting", "cmd_echo"],
        default="greeting",
        help="Scenario to run",
    )
    args = parser.parse_args()

    mock = MockCV(host=args.host, port=args.port)
    mock.start()
    try:
        time.sleep(0.5)
        if args.scenario == "greeting":
            sample_face = make_sample(FaceIdentified)
            if sample_face is None:
                sample_face = FaceIdentified(data=FaceData(name="Finn"))
            mock.publish_event(sample_face)
        elif args.scenario == "cmd_echo":
            time.sleep(1.0)
    finally:
        mock.stop()
