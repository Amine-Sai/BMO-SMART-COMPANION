"""BMO Pi edge client: MQTT communication, emotion rendering, and ESP32 bridge."""

import argparse
from collections.abc import Callable
import time
from typing import Any

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

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
    TOPICS,
    VISION_STATUS,
)
from edge.emotions import render_face_frame
from edge.esp32_protocol import ESP32ProtocolSimulator, format_move_cmd


class PiClient:
    """Pi edge client orchestrating MQTT, LCD rendering, audio, and ESP32 bridge."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        username: str = "pi",
        password: str = "pi_insecure_dev_password",
        sim: bool = True,
        esp32: ESP32ProtocolSimulator | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.sim = sim
        self._clock = clock
        self.esp32 = esp32 if esp32 is not None else (ESP32ProtocolSimulator() if sim else None)

        self.current_emotion: Emotion = Emotion.NEUTRAL
        self.last_frame: dict[str, Any] = render_face_frame(self.current_emotion, speaking=False)
        self.spoken_utterances: list[BrainSay] = []
        self.played_sounds: list[str] = []
        self.serial_log: list[tuple[str, str]] = []
        self.brain_status: BrainStatus | None = None
        self.vision_status: VisionStatus | None = None

        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(self.username, self.password)
        lwt_payload = HwStatus(online=False, ts=0.0, capabilities=[]).model_dump_json()
        self.client.will_set(HW_STATUS, lwt_payload, qos=1, retain=True)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(
        self,
        client: mqtt.Client,
        _userdata: Any,
        _flags: Any,
        _reason_code: Any,
        _properties: Any = None,
    ) -> None:
        for topic in (BRAIN_SAY, HW_CMD, BRAIN_STATUS, VISION_STATUS):
            client.subscribe(topic, qos=TOPICS[topic].qos)
        self.publish_heartbeat()

    def _on_message(
        self,
        _client: mqtt.Client,
        _userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        payload = msg.payload if isinstance(msg.payload, str) else msg.payload.decode("utf-8")
        if msg.topic == BRAIN_SAY:
            say = BrainSay.model_validate_json(payload)
            self.current_emotion = say.emotion
            self.last_frame = render_face_frame(say.emotion, speaking=True, step=0)
            self.spoken_utterances.append(say)
        elif msg.topic == HW_CMD:
            cmd = HwCmd.model_validate_json(payload)
            if cmd.action == "set_expression":
                raw_emotion = cmd.args["emotion"]
                self.current_emotion = raw_emotion if isinstance(raw_emotion, Emotion) else Emotion(str(raw_emotion))
                self.last_frame = render_face_frame(self.current_emotion, speaking=False)
            elif cmd.action == "play_sound":
                self.played_sounds.append(str(cmd.args["name"]))
            elif cmd.action == "move_servo":
                raw_cmd = format_move_cmd(str(cmd.args["id"]), float(cmd.args["angle"]))
                reply = ""
                if self.sim and self.esp32 is not None:
                    reply = self.esp32.handle_line(raw_cmd)
                self.serial_log.append((raw_cmd, reply))
        elif msg.topic == BRAIN_STATUS:
            self.brain_status = BrainStatus.model_validate_json(payload)
        elif msg.topic == VISION_STATUS:
            self.vision_status = VisionStatus.model_validate_json(payload)

    def publish_heartbeat(self) -> None:
        status = HwStatus(
            online=True,
            ts=self._clock(),
            capabilities=["lcd", "speaker", "mic", "servos"],
        )
        self.client.publish(
            HW_STATUS,
            status.model_dump_json(),
            qos=TOPICS[HW_STATUS].qos,
            retain=TOPICS[HW_STATUS].retain,
        )

    def publish_mute_event(self, muted: bool) -> None:
        event = HwEvent(ts=self._clock(), kind="mute_switch", value=muted)
        self.client.publish(
            HW_EVENT,
            event.model_dump_json(),
            qos=TOPICS[HW_EVENT].qos,
            retain=TOPICS[HW_EVENT].retain,
        )

    def connect(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)

    def start(self) -> None:
        self.connect()
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BMO Pi Client")
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--sim", action="store_true", default=True, help="Enable simulator")
    args = parser.parse_args(argv)

    client = PiClient(host=args.host, port=args.port, sim=args.sim)
    client.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()


if __name__ == "__main__":
    main()
