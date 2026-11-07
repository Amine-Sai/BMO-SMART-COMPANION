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
    ) -> None:
        self.host = host
        self.port = int(port)
        self.cmd_topic = cmd_topic
        self.event_topic = event_topic
        self.ctx_topic = ctx_topic
        self.resp_topic = resp_topic
        self.on_command_callback = on_command_callback

        self.cmd_queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self.connected: bool = False
        self.client: Any = None

        if mqtt is not None:
            if hasattr(CallbackAPIVersion, "VERSION2"):
                self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id="bmo_vision_pipeline")
            else:
                self.client = mqtt.Client(client_id="bmo_vision_pipeline")

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_socket_open = self._on_socket_open

    def _on_socket_open(self, client: Any, userdata: Any, sock: socket.socket) -> None:
        """Disable Nagle's algorithm for sub-millisecond local network packet delivery."""
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        rc_code = getattr(rc, "value", rc)
        if rc_code == 0:
            self.connected = True
            client.subscribe(self.cmd_topic, qos=0)
            print(f"[MQTTBridge] Connected to {self.host}:{self.port}, subscribed to '{self.cmd_topic}'", flush=True)
        else:
            print(f"[MQTTBridge] Connect failed with code {rc_code}", file=sys.stderr, flush=True)

    def _on_disconnect(self, client: Any, userdata: Any, flags_or_rc: Any, rc: Any = None, properties: Any = None) -> None:
        self.connected = False

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Parse incoming JSON command and push to thread-safe queue."""
        try:
            payload = _loads(msg.payload)
            if isinstance(payload, dict):
                # Standardize command field (allow aliases 'cmd' or 'action')
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
            self.client.connect_async(self.host, self.port, keepalive=30)
            self.client.loop_start()
            return True
        except Exception as exc:
            print(f"[MQTTBridge] Could not initiate connection to {self.host}:{self.port}: {exc}", file=sys.stderr)
            return False

    def stop(self) -> None:
        """Disconnect and cleanly stop network loop thread."""
        if self.client is not None:
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

    def publish_json(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish JSON payload over MQTT without blocking."""
        if self.client is None or not self.connected:
            return
        try:
            data = _dumps(payload)
            self.client.publish(topic, data, qos=0)
        except Exception as exc:
            print(f"[MQTTBridge] Publish error on '{topic}': {exc}", file=sys.stderr, flush=True)

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
            "timestamp": time.time(),
            "data": data or {},
        }
        self.publish_json(self.resp_topic, payload)

    def publish_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish real-time vision event."""
        payload = {
            "type": "event",
            "event": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        self.publish_json(self.event_topic, payload)

    def publish_context(self, context: dict[str, Any], request_id: str | None = None) -> None:
        """Publish short-term memory visual context snapshot."""
        payload = {
            "type": "context",
            "request_id": request_id,
            "timestamp": time.time(),
            "context": context,
        }
        self.publish_json(self.ctx_topic, payload)
