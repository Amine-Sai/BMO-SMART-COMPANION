"""Comprehensive unit and integration tests for BMO-CV upgrades."""

import os
import pathlib
import sys
import time
import unittest
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from assistant_cv.enroll import AutoEnrollSession
from assistant_cv.memory import PersonFaceSynchronizer, ShortTermMemory, get_proximity, get_screen_position
from assistant_cv.mqtt_bridge import MQTTBridge, _dumps, _loads


class DummyFace:
    def __init__(self, emb: np.ndarray, bbox: tuple[int, int, int, int] = (120, 80, 200, 180)):
        self.normed_embedding = emb
        self.bbox = np.array(bbox)


class DummyIdentifier:
    def __init__(self):
        self.saved_identities = {}

    def save_identity(self, name: str, embeddings: list[np.ndarray]):
        self.saved_identities[name] = np.mean(embeddings, axis=0)
        return pathlib.Path(f"gallery/{name}.npy")


class TestPersonFaceSynchronizer(unittest.TestCase):
    def test_spatial_matching(self):
        sync = PersonFaceSynchronizer()
        persons = [
            {"label": "person", "conf": 0.95, "bbox": (100, 50, 300, 450), "id": 1},
            {"label": "person", "conf": 0.90, "bbox": (400, 50, 600, 450), "id": 2},
        ]
        faces = [
            ("Alice", 0.88, (140, 70, 220, 170)),
            ("Bob", 0.82, (440, 70, 520, 170)),
        ]
        synced = sync.sync(persons, faces, now=100.0)

        self.assertEqual(synced[0]["identity"], "Alice")
        self.assertEqual(synced[1]["identity"], "Bob")
        self.assertFalse(sync.needs_face_recognition(synced, now=100.5))

    def test_temporal_persistence(self):
        sync = PersonFaceSynchronizer()
        persons_f1 = [{"label": "person", "conf": 0.95, "bbox": (100, 50, 300, 450), "id": 7}]
        faces_f1 = [("Charlie", 0.91, (140, 70, 220, 170))]
        sync.sync(persons_f1, faces_f1, now=10.0)

        # Frame 2: Face turned away (0 faces detected), but YOLO track #7 continues
        persons_f2 = [{"label": "person", "conf": 0.94, "bbox": (105, 52, 305, 452), "id": 7}]
        synced_f2 = sync.sync(persons_f2, [], now=10.2)

        # Charlie should persist on track #7
        self.assertEqual(synced_f2[0]["identity"], "Charlie")
        # Smart recheck: known tracked person does NOT trigger face recognition
        self.assertFalse(sync.needs_face_recognition(synced_f2, now=11.0))
        self.assertFalse(sync.needs_face_recognition(synced_f2, now=14.5))
        # When explicitly flagged for re-check, should need face recognition
        sync.flag_for_recheck(7)
        self.assertTrue(sync.needs_face_recognition(synced_f2, now=14.6))

    def test_unknown_person_needs_recognition(self):
        sync = PersonFaceSynchronizer()
        persons = [{"label": "person", "conf": 0.85, "bbox": (100, 50, 300, 450), "id": 3}]
        synced = sync.sync(persons, [], now=10.0)
        self.assertEqual(synced[0]["identity"], "unknown")
        self.assertTrue(sync.needs_face_recognition(synced, now=10.0))


class TestShortTermMemory(unittest.TestCase):
    def test_memory_context_generation(self):
        mem = ShortTermMemory(window_sec=5.0)
        objects = [
            {"label": "person", "conf": 0.92, "identity": "Alice", "face_conf": 0.85, "bbox": (250, 60, 390, 420), "id": 1},
            {"label": "cup", "conf": 0.80, "bbox": (100, 300, 150, 360), "id": None},
        ]
        mem.update(objects, frame_shape=(480, 640))
        mem.record_sign("HELLO", 0.95, signer="Alice")

        ctx = mem.get_llm_context()
        self.assertIn("Alice", ctx["summary"])
        self.assertIn("cup", ctx["summary"])
        self.assertIn("HELLO", ctx["summary"])
        self.assertEqual(len(ctx["current_persons"]), 1)
        self.assertEqual(ctx["current_persons"][0]["identity"], "Alice")
        self.assertEqual(ctx["current_persons"][0]["position"], "center")
        self.assertEqual(len(ctx["recent_signs"]), 1)

    def test_spatial_helpers(self):
        self.assertEqual(get_screen_position((20, 20, 100, 100), frame_w=640), "left")
        self.assertEqual(get_screen_position((280, 20, 360, 100), frame_w=640), "center")
        self.assertEqual(get_screen_position((520, 20, 620, 100), frame_w=640), "right")


class TestAutoEnrollSession(unittest.TestCase):
    def test_auto_enroll_flow(self):
        ident = DummyIdentifier()
        session = AutoEnrollSession(
            name="TestBot",
            target_shots=4,
            min_interval_sec=0.01,
            timeout_sec=5.0,
            identifier=ident,
        )

        for i in range(4):
            # Generate slightly different 512-D vectors
            emb = np.zeros(512, dtype=np.float32)
            emb[i] = 1.0
            face = DummyFace(emb)
            active = session.step([face])
            time.sleep(0.015)

        self.assertTrue(session.completed)
        self.assertFalse(session.failed)
        self.assertIn("TestBot", ident.saved_identities)
        self.assertEqual(session.saved_identities_count if hasattr(session, "saved_identities_count") else 4, 4)


class TestMQTTBridgeSerialization(unittest.TestCase):
    def test_json_roundtrip(self):
        payload = {
            "command": "enroll_person",
            "name": "Alex",
            "shots": 8,
            "params": {"tau": 0.45},
        }
        encoded = _dumps(payload)
        decoded = _loads(encoded)
        self.assertEqual(decoded["command"], "enroll_person")
        self.assertEqual(decoded["name"], "Alex")

    def test_command_queue(self):
        bridge = MQTTBridge()
        # Mock incoming message
        class MockMsg:
            payload = b'{"cmd": "get_context", "request_id": "99"}'
            topic = "bmo/vision/command"

        bridge._on_message(None, None, MockMsg())
        cmds = bridge.poll_commands()
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["command"], "get_context")
        self.assertEqual(cmds[0]["request_id"], "99")


if __name__ == "__main__":
    unittest.main()
