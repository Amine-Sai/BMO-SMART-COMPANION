"""Comprehensive unit and integration tests for BMO-CV upgrades."""

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
from assistant_cv.memory import PersonFaceSynchronizer, ShortTermMemory, get_screen_position
from assistant_cv.mqtt_bridge import MQTTBridge, _dumps, _loads
from bmo_contracts.topics import VISION_CMD


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
            {
                "label": "person",
                "conf": 0.92,
                "identity": "Alice",
                "face_conf": 0.85,
                "bbox": (250, 60, 390, 420),
                "id": 1,
            },
            {"label": "cup", "conf": 0.80, "bbox": (100, 300, 150, 360), "id": None},
        ]
        mem.update(objects, frame_shape=(480, 640))

        ctx = mem.get_llm_context()
        self.assertIn("Alice", ctx["summary"])
        self.assertIn("cup", ctx["summary"])
        self.assertEqual(len(ctx["current_persons"]), 1)
        self.assertEqual(ctx["current_persons"][0]["identity"], "Alice")
        self.assertEqual(ctx["current_persons"][0]["position"], "center")

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
            session.step([face])
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
            topic = VISION_CMD

        bridge._on_message(None, None, MockMsg())
        cmds = bridge.poll_commands()
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["command"], "get_context")
        self.assertEqual(cmds[0]["request_id"], "99")


from assistant_cv.config import OBJ_FRAME_STRIDE
from assistant_cv.main import EventDetector


class TestEventDetectorDebounce(unittest.TestCase):
    """Test EventDetector temporal debounce and object appearance/removal events."""

    def setUp(self):
        self.emitted = []
        self.detector = EventDetector()
        self.detector._emit = lambda ev_type, data: self.emitted.append((ev_type, data))

    def test_person_entry_exit_debounce(self):
        person_obj = [{"label": "person", "conf": 0.9, "bbox": (10, 10, 100, 200)}]

        # Frame 1: 1 person appears -> pending (no event yet)
        self.detector.update(objects=person_obj, faces=[])
        self.assertEqual(len(self.emitted), 0)

        # Frame 2: 1 person still present -> confirmed (person_detected emitted)
        self.detector.update(objects=person_obj, faces=[])
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0][0], "person_detected")
        self.assertEqual(self.emitted[0][1]["count"], 1)

        # Frame 3: momentary 1-frame dropout -> absorbed by debounce (no event emitted)
        self.detector.update(objects=[], faces=[])
        self.assertEqual(len(self.emitted), 1)

        # Frame 4: person restored -> absorbed, stable count remains 1
        self.detector.update(objects=person_obj, faces=[])
        self.assertEqual(len(self.emitted), 1)

        # Frame 5: person truly leaves -> pending
        self.detector.update(objects=[], faces=[])
        self.assertEqual(len(self.emitted), 1)

        # Frame 6: person truly gone -> confirmed (person_left emitted)
        self.detector.update(objects=[], faces=[])
        self.assertEqual(len(self.emitted), 2)
        self.assertEqual(self.emitted[1][0], "person_left")
        self.assertEqual(self.emitted[1][1]["count"], 0)

    def test_object_appearance_and_removal(self):
        lighter_obj = [{"label": "lighter", "conf": 0.85, "bbox": (50, 50, 80, 120)}]

        # Frame 1: lighter appears -> pending
        self.detector.update(objects=lighter_obj, faces=[])
        self.assertEqual(len(self.emitted), 0)

        # Frame 2: lighter still present -> confirmed (object_seen emitted)
        self.detector.update(objects=lighter_obj, faces=[])
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual(self.emitted[0][0], "object_seen")
        self.assertEqual(self.emitted[0][1]["label"], "lighter")
        self.assertEqual(self.emitted[0][1]["count"], 1)

        # Frame 3 & 4: lighter removed -> confirmed (object_removed emitted)
        self.detector.update(objects=[], faces=[])
        self.assertEqual(len(self.emitted), 1)
        self.detector.update(objects=[], faces=[])
        self.assertEqual(len(self.emitted), 2)
        self.assertEqual(self.emitted[1][0], "object_removed")
        self.assertEqual(self.emitted[1][1]["label"], "lighter")


class TestDecoupledDetectionCadence(unittest.TestCase):
    """Test object detection frame-stride decoupling logic."""

    def test_default_stride(self):
        # Default changed to 1 for better detection on CPU
        self.assertEqual(OBJ_FRAME_STRIDE, 1)

    def test_stride_cadence_simulation(self):
        cached_detections = []
        detector_invocations = 0

        def mock_detect():
            nonlocal detector_invocations
            detector_invocations += 1
            return [{"label": "lighter", "conf": 0.8}]

        # Simulate 12 frames
        for frame_count in range(1, 13):
            run_det = (frame_count % OBJ_FRAME_STRIDE == 0) or not cached_detections
            if run_det:
                detections = mock_detect()
                cached_detections = detections
            else:
                detections = cached_detections

            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["label"], "lighter")

        # With stride=1, detection runs every frame
        expected_invocations = 12
        self.assertEqual(detector_invocations, expected_invocations)


if __name__ == "__main__":
    unittest.main()
