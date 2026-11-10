"""High-speed real-time MQTT JSON bridge for BMO Computer Vision and LLM Agent."""

from __future__ import annotations

import queue
import socket
import sys
import time
from typing import Any, Callable

try:
    import orjson

    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)

    def _loads(data: bytes | str) -> Any:
        return orjson.loads(data)
except ImportError:
    import json

    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def _loads(data: bytes | str) -> Any:
        return json.loads(data)


try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    mqtt = None
    CallbackAPIVersion = None

from bmo_contracts.models import EVENT_ADAPTER, Scene, VisionCmdResult, VisionStatus
from bmo_contracts.topics import (
    VISION_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
    VISION_STATUS,
)

try:
    from .config import (
        MQTT_HOST,
        MQTT_PORT,
        MQTT_TOPIC_CMD,
        MQTT_TOPIC_CTX,
        MQTT_TOPIC_EVENT,
        MQTT_TOPIC_RESP,
    )
except ImportError:
    from config import (
        MQTT_HOST,
        MQTT_PORT,
        MQTT_TOPIC_CMD,
        MQTT_TOPIC_CTX,
        MQTT_TOPIC_EVENT,
        MQTT_TOPIC_RESP,
    )


class MQTTBridge:
    """Non-blocking low-latency MQTT bridge with TCP_NODELAY and orjson serialization."""

    def __init__(
        self,
        host: str = MQTT_HOST,
        port: int = MQTT_PORT,
        cmd_topic: str = MQTT_TOPIC_CMD,
        event_topic: str = MQTT_TOPIC_EVENT,
        ctx_topic: str = MQTT_TOPIC_CTX,
        resp_topic: str = MQTT_TOPIC_RESP,
        on_command_callback: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.cmd_topic = cmd_topic or VISION_CMD
        self.event_topic = event_topic or VISION_EVENTS
        self.ctx_topic = ctx_topic or VISION_CONTEXT
        self.resp_topic = resp_topic
        self.cmd_result_topic = VISION_CMD_RESULT
        self.on_command_callback = on_command_callback
        self.clock = clock

        self.cmd_queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self.connected: bool = False
        self.client: Any = None
        self._event_counter: int = 0

        if mqtt is not None:
            if hasattr(CallbackAPIVersion, "VERSION2"):
                self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id="bmo_vision_pipeline")
            else:
                self.client = mqtt.Client(client_id="bmo_vision_pipeline")

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_socket_open = self._on_socket_open

    def _on_socket_open(self, _client: Any, _userdata: Any, sock: socket.socket) -> None:
        """Disable Nagle's algorithm for sub-millisecond local network packet delivery."""
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, rc: Any, _properties: Any = None) -> None:
        rc_code = getattr(rc, "value", rc)
        if rc_code == 0:
            self.connected = True
            try:
                online_bytes = _dumps(VisionStatus(online=True).model_dump())
                client.publish(VISION_STATUS, online_bytes, qos=1, retain=True)
            except Exception:
                pass

            client.subscribe(self.cmd_topic, qos=1)
            if self.cmd_topic != VISION_CMD:
                client.subscribe(VISION_CMD, qos=1)
            print(f"[MQTTBridge] Connected to {self.host}:{self.port}, subscribed to '{self.cmd_topic}'", flush=True)
        else:
            print(f"[MQTTBridge] Connect failed with code {rc_code}", file=sys.stderr, flush=True)

    def _on_disconnect(
        self, _client: Any, _userdata: Any, _flags_or_rc: Any, _rc: Any = None, _properties: Any = None
    ) -> None:
        self.connected = False

    def _on_message(self, _client: Any, _userdata: Any, msg: Any) -> None:
        """Parse incoming JSON command and push to thread-safe queue."""
        try:
            payload = _loads(msg.payload)
            if isinstance(payload, dict):
                if "command" not in payload:
                    if "cmd" in payload:
                        payload["command"] = payload["cmd"]
                    elif "action" in payload:
                        payload["command"] = payload["action"]

                self.cmd_queue.put(payload)
                if self.on_command_callback:
                    self.on_command_callback(payload)
        except Exception as exc:
            print(f"[MQTTBridge] Malformed JSON message on '{msg.topic}': {exc}", file=sys.stderr, flush=True)

    def start(self) -> bool:
        """Connect asynchronously to MQTT broker and start background network thread."""
        if self.client is None:
            print("[MQTTBridge] paho-mqtt not available; MQTT bridge disabled.", file=sys.stderr)
            return False

        try:
            lwt_payload = _dumps(VisionStatus(online=False).model_dump())
            self.client.will_set(VISION_STATUS, lwt_payload, qos=1, retain=True)
            self.client.connect_async(self.host, self.port, keepalive=30)
            self.client.loop_start()
            return True
        except Exception as exc:
            print(f"[MQTTBridge] Could not initiate connection to {self.host}:{self.port}: {exc}", file=sys.stderr)
            return False

    def stop(self) -> None:
        """Disconnect and cleanly stop network loop thread."""
        if self.client is not None:
            if self.connected:
                try:
                    offline_payload = _dumps(VisionStatus(online=False).model_dump())
                    self.client.publish(VISION_STATUS, offline_payload, qos=1, retain=True)
                except Exception:
                    pass
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            self.connected = False

    def poll_commands(self) -> list[dict[str, Any]]:
        """Drain all queued incoming commands received since last poll."""
        cmds: list[dict[str, Any]] = []
        while not self.cmd_queue.empty():
            try:
                cmds.append(self.cmd_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def publish_json(self, topic: str, payload: dict[str, Any], qos: int = 0, retain: bool = False) -> None:
        """Publish JSON payload over MQTT without blocking."""
        if self.client is None or not self.connected:
            return
        try:
            data = _dumps(payload)
            if retain:
                self.client.publish(topic, data, qos=qos, retain=True)
            else:
                self.client.publish(topic, data, qos=qos)
        except Exception as exc:
            print(f"[MQTTBridge] Publish error on '{topic}': {exc}", file=sys.stderr, flush=True)

    def publish_scene(self, scene_dict: dict[str, Any]) -> None:
        """Validate scene against Scene model and publish retained context."""
        if self.client is None or not self.connected:
            return
        try:
            scene = Scene.model_validate(scene_dict)
            payload_bytes = _dumps(scene.model_dump())
            self.client.publish(self.ctx_topic, payload_bytes, qos=0, retain=True)
        except Exception as exc:
            print(f"[MQTTBridge] Scene publish error on '{self.ctx_topic}': {exc}", file=sys.stderr, flush=True)

    def publish_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish real-time vision event conforming to contract models when applicable."""
        if self.client is None or not self.connected:
            return

        norm_type = "object_seen" if event_type == "object_detected" else event_type
        contract_types = {
            "face_identified",
            "person_detected",
            "person_left",
            "sign_recognized",
            "object_seen",
        }

        if norm_type in contract_types:
            self._event_counter += 1
            evt_id = f"evt-{self._event_counter:06d}"
            now_ts = round(self.clock(), 3)
            norm_data: dict[str, Any] = {}

            if norm_type == "face_identified":
                norm_data = {
                    "track_id": data.get("track_id", data.get("id")),
                    "name": str(data.get("name", "unknown")),
                    "conf": float(data.get("conf", data.get("score", 1.0))),
                }
            elif norm_type == "person_detected":
                prox = str(data.get("proximity", "medium"))
                if prox == "close":
                    prox = "near"
                elif prox not in ("near", "medium", "far"):
                    prox = "medium"

                pos = str(data.get("pos", data.get("position", "center")))
                if pos not in ("left", "center", "right"):
                    pos = "center"

                norm_data = {
                    "track_id": data.get("track_id", data.get("id")),
                    "pos": pos,
                    "proximity": prox,
                }
            elif norm_type == "person_left":
                norm_data = {
                    "track_id": data.get("track_id", data.get("id")),
                }
            elif norm_type == "sign_recognized":
                sign_txt = str(data.get("text") or data.get("sign") or "").lower()
                norm_data = {
                    "text": sign_txt,
                    "conf": float(data.get("conf", 1.0)),
                }
            elif norm_type == "object_seen":
                label_txt = str(data.get("label") or data.get("name") or "")
                norm_data = {
                    "label": label_txt,
                    "conf": float(data.get("conf", data.get("score", 1.0))),
                }

            raw_payload = {
                "v": 1,
                "id": evt_id,
                "ts": now_ts,
                "type": norm_type,
                "data": norm_data,
            }
            try:
                validated_event = EVENT_ADAPTER.validate_python(raw_payload)
                payload_bytes = _dumps(validated_event.model_dump())
                self.client.publish(self.event_topic, payload_bytes, qos=1)
                return
            except Exception as exc:
                print(f"[MQTTBridge] Event validation error: {exc}", file=sys.stderr, flush=True)

        payload = {
            "type": "event",
            "event": event_type,
            "timestamp": self.clock(),
            "data": data,
        }
        self.publish_json(self.event_topic, payload, qos=0)

    def publish_cmd_result(self, cmd_id: str, ok: bool, detail: str) -> None:
        """Publish contract-validated VisionCmdResult on VISION_CMD_RESULT."""
        if self.client is None or not self.connected:
            return
        try:
            res = VisionCmdResult(id=str(cmd_id), ok=bool(ok), detail=str(detail))
            self.client.publish(self.cmd_result_topic, _dumps(res.model_dump()), qos=1)
        except Exception as exc:
            print(f"[MQTTBridge] Cmd result publish error: {exc}", file=sys.stderr, flush=True)

    def publish_response(
        self,
        command: str,
        status: str,
        data: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Publish command execution status and response payload."""
        payload = {
            "type": "command_response",
            "command": command,
            "status": status,
            "request_id": request_id,
            "timestamp": self.clock(),
            "data": data or {},
        }
        self.publish_json(self.resp_topic, payload)
