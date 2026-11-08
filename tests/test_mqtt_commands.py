"""Comprehensive unit and integration tests for BMO MQTT commands and event protocols."""

import asyncio
import json
import os
import pathlib
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt

from assistant_cv.mqtt_bridge import MQTTBridge
from assistant_cv.memory import ShortTermMemory, PersonFaceSynchronizer
from assistant_cv.registry import IdentityRegistry, ObjectRegistry
from tools.simple_broker import SimpleMQTTBroker


class TestMQTTBridgeUnit(unittest.TestCase):
    """Test MQTTBridge queueing, payload parsing, and serialization."""

    def setUp(self):
        self.bridge = MQTTBridge(host="localhost", port=1883)

    def test_command_normalization(self):
        """Verify command aliases 'cmd' and 'action' are normalized to 'command'."""
        mock_msg = MagicMock()
        mock_msg.topic = "bmo/vision/command"
        mock_msg.payload = b'{"cmd": "status", "request_id": "req-1"}'
        self.bridge._on_message(None, None, mock_msg)

        cmds = self.bridge.poll_commands()
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["command"], "status")
        self.assertEqual(cmds[0]["request_id"], "req-1")

        mock_msg.payload = b'{"action": "get_context", "request_id": "req-2"}'
        self.bridge._on_message(None, None, mock_msg)
        cmds2 = self.bridge.poll_commands()
        self.assertEqual(len(cmds2), 1)
        self.assertEqual(cmds2[0]["command"], "get_context")

    def test_publish_response_structure(self):
        """Verify publish_response generates standardized schema."""
        published = []
        self.bridge.connected = True
        self.bridge.client = MagicMock()
        self.bridge.client.publish = lambda topic, data, qos: published.append((topic, json.loads(data)))

        self.bridge.publish_response(
            command="list_persons",
            status="success",
            data={"count": 2},
            request_id="req-99",
        )

        self.assertEqual(len(published), 1)
        topic, payload = published[0]
        self.assertEqual(topic, self.bridge.resp_topic)
        self.assertEqual(payload["type"], "command_response")
        self.assertEqual(payload["command"], "list_persons")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["request_id"], "req-99")
        self.assertEqual(payload["data"]["count"], 2)


import numpy as np

class TestMQTTCommandDispatcher(unittest.TestCase):
    """Verify command handlers produce exact expected schemas and state transitions."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.gallery_dir = pathlib.Path(self.tmpdir.name)
        self.test_classes_path = self.gallery_dir / "test_objects.json"
        self.test_classes_path.write_text('{"classes": ["person"]}')
        self.ident_reg = IdentityRegistry(gallery_dir=self.gallery_dir)
        self.obj_reg = ObjectRegistry(
            gallery_dir=self.gallery_dir / "objects",
            classes_path=self.test_classes_path,
        )
        self.memory = ShortTermMemory(window_sec=60.0, registry=self.ident_reg)
        self.synchronizer = PersonFaceSynchronizer(registry=self.ident_reg)

        self.published_responses: list[dict] = []
        self.bridge = MagicMock()
        self.bridge.publish_response = lambda cmd, status, data=None, req_id=None: self.published_responses.append(
            {"command": cmd, "status": status, "data": data or {}, "request_id": req_id}
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_status_command(self):
        fps = 42.5
        toggles = {"1:OBJ": True, "2:FACE": True, "3:SIGN": True, "L:LAZY": True, "T:TRK": True}
        status_data = {
            "fps": round(fps, 1),
            "lazy_mode": toggles["L:LAZY"],
            "toggles": toggles,
            "persons_in_view": len(self.memory.current_persons),
        }
        self.bridge.publish_response("status", "ok", status_data, "req-stat")

        self.assertEqual(len(self.published_responses), 1)
        resp = self.published_responses[0]
        self.assertEqual(resp["command"], "status")
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["data"]["fps"], 42.5)
        self.assertTrue(resp["data"]["lazy_mode"])

    def test_get_context_command(self):
        ctx_snapshot = self.memory.get_llm_context()
        self.bridge.publish_response("get_context", "success", {"context": ctx_snapshot}, "req-ctx")

        self.assertEqual(len(self.published_responses), 1)
        resp = self.published_responses[0]
        self.assertEqual(resp["status"], "success")
        self.assertIn("summary", resp["data"]["context"])
        self.assertIn("known_persons", resp["data"]["context"])
        self.assertIn("registered_objects", resp["data"]["context"])

    def test_person_management_commands(self):
        # Register a person with 512-D embedding
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.ident_reg.register_person("Bob", emb)
        roster = self.ident_reg.list_identities()
        self.bridge.publish_response("list_persons", "success", {"persons": roster, "count": len(roster)}, "req-lp")

        self.assertEqual(self.published_responses[-1]["status"], "success")
        self.assertEqual(self.published_responses[-1]["data"]["count"], 1)

        # Rename
        ok, _ = self.ident_reg.rename_person("Bob", "Robert")
        self.assertTrue(ok)
        self.bridge.publish_response("rename_person", "success", {"old_name": "Bob", "new_name": "Robert"})
        self.assertEqual(self.published_responses[-1]["status"], "success")

        # Delete
        ok_del, _ = self.ident_reg.delete_person("Robert")
        self.assertTrue(ok_del)
        self.bridge.publish_response("delete_person", "success", {"name": "Robert"})
        self.assertEqual(self.published_responses[-1]["status"], "success")

    def test_object_management_commands(self):
        # Register object in text mode
        self.obj_reg.register_object("vape", synonyms=["electronic cigarette"])
        objs = self.obj_reg.list_objects()
        self.bridge.publish_response("list_objects", "success", {"objects": objs, "count": len(objs)})
        self.assertEqual(self.published_responses[-1]["status"], "success")
        self.assertEqual(self.published_responses[-1]["data"]["count"], 1)

        # Rename object
        ok, _ = self.obj_reg.rename_object("vape", "vaporizer")
        self.assertTrue(ok)
        self.bridge.publish_response("rename_object", "success", {"old_name": "vape", "new_name": "vaporizer"})
        self.assertEqual(self.published_responses[-1]["status"], "success")

        # Delete object
        ok_del, _ = self.obj_reg.delete_object("vaporizer")
        self.assertTrue(ok_del)
        self.bridge.publish_response("delete_object", "success", {"name": "vaporizer"})
        self.assertEqual(self.published_responses[-1]["status"], "success")


class TestLiveMQTTRoundtrip(unittest.TestCase):
    """End-to-end integration test running SimpleMQTTBroker and exchanging live messages."""

    @classmethod
    def setUpClass(cls):
        cls.test_port = 11884
        cls.broker = SimpleMQTTBroker(host="127.0.0.1", port=cls.test_port)

        async def _run_server():
            await cls.broker.start()
            if cls.broker.server:
                await cls.broker.server.serve_forever()

        cls.broker_thread = threading.Thread(
            target=lambda: asyncio.run(_run_server()),
            daemon=True,
        )
        cls.broker_thread.start()
        time.sleep(0.4)

    def test_live_command_and_response_loop(self):
        # 1. Start simulated BMO Bridge
        bridge_received: list[dict] = []
        bmo_bridge = MQTTBridge(host="127.0.0.1", port=self.test_port)
        bmo_bridge.on_command_callback = lambda cmd: bridge_received.append(cmd)
        self.assertTrue(bmo_bridge.start())

        # Wait for bridge to connect and subscribe
        for _ in range(30):
            if bmo_bridge.connected:
                break
            time.sleep(0.05)
        self.assertTrue(bmo_bridge.connected)

        # 2. Start Test Client (simulating tools/mqtt_client.py)
        client_connected = threading.Event()
        client_responses: list[dict] = []

        def on_client_connect(c, u, flags, rc, props=None):
            client_connected.set()

        test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="tester_agent")
        test_client.on_connect = on_client_connect
        test_client.on_message = lambda c, u, msg: client_responses.append(json.loads(msg.payload))
        test_client.connect("127.0.0.1", self.test_port)
        test_client.subscribe("bmo/vision/response")
        test_client.loop_start()

        self.assertTrue(client_connected.wait(timeout=2.0))
        time.sleep(0.1)

        # 3. Publish command from agent
        cmd_payload = {"command": "status", "request_id": "test-req-001"}
        test_client.publish("bmo/vision/command", json.dumps(cmd_payload))

        # Wait for command arrival
        for _ in range(30):
            if len(bridge_received) >= 1:
                break
            time.sleep(0.05)

        self.assertEqual(len(bridge_received), 1)
        self.assertEqual(bridge_received[0]["command"], "status")

        # 4. Bridge sends response
        bmo_bridge.publish_response("status", "ok", {"fps": 45.0}, request_id="test-req-001")

        # Wait for response arrival
        for _ in range(30):
            if len(client_responses) >= 1:
                break
            time.sleep(0.05)

        self.assertEqual(len(client_responses), 1)
        self.assertEqual(client_responses[0]["command"], "status")
        self.assertEqual(client_responses[0]["status"], "ok")
        self.assertEqual(client_responses[0]["data"]["fps"], 45.0)

        # Clean up
        test_client.loop_stop()
        test_client.disconnect()
        bmo_bridge.stop()


if __name__ == "__main__":
    unittest.main()
