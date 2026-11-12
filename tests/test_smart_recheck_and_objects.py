"""Unit and integration tests for smart re-checking, persistent registry, and expanded object detection."""

import os
import pathlib
import sys
import tempfile
import time
import unittest
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from assistant_cv.enroll import AutoEnrollSession
from assistant_cv.memory import PersonFaceSynchronizer, ShortTermMemory
from assistant_cv.objects import ObjectDetector
from assistant_cv.registry import IdentityRegistry


class DummyFace:
    def __init__(self, emb: np.ndarray, bbox: tuple[int, int, int, int] = (120, 80, 200, 180)):
        self.normed_embedding = emb
        self.bbox = np.array(bbox)


class MockIdentifier:
    def __init__(self, registered: dict[str, np.ndarray] | None = None):
        self.registered = registered or {}

    def check_already_registered(self, embedding: np.ndarray, tau: float = 0.40):
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        for name, saved_emb in self.registered.items():
            saved_norm = saved_emb / (np.linalg.norm(saved_emb) + 1e-8)
            sim = float(emb @ saved_norm)
            if sim >= tau:
                return name, sim
        return None, 0.0

    def save_identity(self, name: str, embeddings: list[np.ndarray]):
        self.registered[name] = np.mean(embeddings, axis=0)
        return pathlib.Path(f"{name}.npy")


class TestIdentityRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gallery_dir = pathlib.Path(self.temp_dir.name)
        self.registry = IdentityRegistry(gallery_dir=self.gallery_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_register_and_persist(self):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        ok, msg, path = self.registry.register_person("Judas", emb, notes="Primary user")
        self.assertTrue(ok)
        self.assertTrue(path.exists())
        self.assertTrue(self.registry.is_registered("Judas"))

        # Reload from disk in a fresh instance
        registry2 = IdentityRegistry(gallery_dir=self.gallery_dir)
        self.assertTrue(registry2.is_registered("Judas"))
        names = registry2.get_known_names()
        self.assertIn("Judas", names)
        entry = registry2.get_person("Judas")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["notes"], "Primary user")

    def test_rename_person(self):
        emb = np.random.randn(512).astype(np.float32)
        self.registry.register_person("OldName", emb)
        self.assertTrue((self.gallery_dir / "OldName.npy").exists())

        ok, msg = self.registry.rename_person("OldName", "NewName")
        self.assertTrue(ok)
        self.assertFalse(self.registry.is_registered("OldName"))
        self.assertTrue(self.registry.is_registered("NewName"))
        self.assertFalse((self.gallery_dir / "OldName.npy").exists())
        self.assertTrue((self.gallery_dir / "NewName.npy").exists())

    def test_delete_person(self):
        emb = np.random.randn(512).astype(np.float32)
        self.registry.register_person("DeleteMe", emb)
        self.assertTrue(self.registry.is_registered("DeleteMe"))

        ok, msg = self.registry.delete_person("DeleteMe")
        self.assertTrue(ok)
        self.assertFalse(self.registry.is_registered("DeleteMe"))
        self.assertFalse((self.gallery_dir / "DeleteMe.npy").exists())

    def test_permanent_retention(self):
        # Adding and modifying people never deletes other entries
        for name in ["UserA", "UserB", "UserC"]:
            emb = np.random.randn(512).astype(np.float32)
            self.registry.register_person(name, emb)

        self.assertEqual(len(self.registry.list_identities()), 3)
        self.registry.record_encounter("UserA")

        # Reload
        registry_reloaded = IdentityRegistry(gallery_dir=self.gallery_dir)
        self.assertEqual(len(registry_reloaded.list_identities()), 3)
        self.assertIn("UserA", registry_reloaded.get_known_names())
        self.assertIn("UserB", registry_reloaded.get_known_names())
        self.assertIn("UserC", registry_reloaded.get_known_names())


class TestSmartRecheckEnrollment(unittest.TestCase):
    def test_aborts_if_already_registered_under_different_name(self):
        fixed_emb = np.zeros(512, dtype=np.float32)
        fixed_emb[0] = 1.0

        mock_ident = MockIdentifier(registered={"Alice": fixed_emb})
        session = AutoEnrollSession(
            name="Bob",
            target_shots=4,
            min_interval_sec=0.01,
            timeout_sec=5.0,
            identifier=mock_ident,
        )

        face = DummyFace(fixed_emb.copy())
        active = session.step([face])

        self.assertFalse(active)
        self.assertTrue(session.failed)
        self.assertTrue(session.already_registered)
        self.assertEqual(session.already_registered_as, "Alice")
        self.assertIn("Alice", session.error_msg)

    def test_allows_re_enrolling_same_person(self):
        fixed_emb = np.zeros(512, dtype=np.float32)
        fixed_emb[0] = 1.0

        mock_ident = MockIdentifier(registered={"Alice": fixed_emb})
        session = AutoEnrollSession(
            name="Alice",
            target_shots=2,
            min_interval_sec=0.01,
            timeout_sec=5.0,
            identifier=mock_ident,
        )

        # Shot 1
        active1 = session.step([DummyFace(fixed_emb.copy())])
        self.assertTrue(active1)
        self.assertFalse(session.failed)

        # Shot 2 slightly perturbed
        time.sleep(0.015)
        emb2 = fixed_emb.copy()
        emb2[1] = 0.2
        emb2 = emb2 / np.linalg.norm(emb2)
        active2 = session.step([DummyFace(emb2)])

        self.assertFalse(active2)
        self.assertTrue(session.completed)
        self.assertFalse(session.failed)


class TestPersonFaceSmartTracking(unittest.TestCase):
    def test_known_person_no_recheck(self):
        sync = PersonFaceSynchronizer()
        persons = [{"label": "person", "conf": 0.95, "bbox": (100, 50, 300, 450), "id": 42}]
        faces = [("Judas", 0.92, (140, 70, 220, 170))]

        # Initial frame identifies Judas on track #42
        synced = sync.sync(persons, faces, now=10.0)
        self.assertEqual(synced[0]["identity"], "Judas")

        # Track #42 continues for 20 seconds without faces
        for t in [11.0, 15.0, 20.0, 25.0]:
            p_cont = [{"label": "person", "conf": 0.93, "bbox": (102, 51, 302, 451), "id": 42}]
            synced_cont = sync.sync(p_cont, [], now=t)
            self.assertEqual(synced_cont[0]["identity"], "Judas")
            # Smart re-check: must NOT request face recognition while tracked!
            self.assertFalse(sync.needs_face_recognition(synced_cont, now=t))

    def test_flag_for_recheck(self):
        sync = PersonFaceSynchronizer()
        persons = [{"label": "person", "conf": 0.95, "bbox": (100, 50, 300, 450), "id": 10}]
        faces = [("Elena", 0.89, (140, 70, 220, 170))]
        sync.sync(persons, faces, now=10.0)

        p_cont = [{"label": "person", "conf": 0.92, "bbox": (100, 50, 300, 450), "id": 10}]
        self.assertFalse(sync.needs_face_recognition(p_cont, now=15.0))

        # Explicitly flag track #10
        sync.flag_for_recheck(10)
        self.assertTrue(sync.needs_face_recognition(p_cont, now=15.1))


class TestShortTermMemoryRoster(unittest.TestCase):
    def test_known_persons_roster_in_context(self):
        temp_dir = tempfile.TemporaryDirectory()
        try:
            reg = IdentityRegistry(gallery_dir=pathlib.Path(temp_dir.name))
            reg.register_person("Judas", np.random.randn(512).astype(np.float32))
            reg.register_person("BMO_Friend", np.random.randn(512).astype(np.float32))

            mem = ShortTermMemory(window_sec=30.0, registry=reg)
            ctx = mem.get_llm_context()

            self.assertIn("known_persons", ctx)
            self.assertIn("Judas", ctx["known_persons"])
            self.assertIn("BMO_Friend", ctx["known_persons"])
        finally:
            temp_dir.cleanup()


class TestExpandedObjectDetector(unittest.TestCase):
    def test_yolo_world_classes(self):
        detector = ObjectDetector(model_type="world")
        self.assertTrue(detector.is_world_model)
        self.assertGreaterEqual(len(detector.class_names), 40)
        self.assertIn("pen", detector.class_names)
        self.assertIn("coffee mug", detector.class_names)
        self.assertIn("glasses", detector.class_names)
        self.assertIn("keys", detector.class_names)

        # Test set_classes dynamically
        custom_classes = ["smart watch", "water bottle", "person"]
        detector.set_classes(custom_classes)
        self.assertEqual(detector.class_names, custom_classes)

        # Test inference returns valid structure
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        results = detector.detect(dummy_img, track=False)
        self.assertIsInstance(results, list)


class TestConfigAndClasses(unittest.TestCase):
    def test_config_constants(self):
        from assistant_cv.config import MAX_FPS, OBJ_CONF, load_object_classes
        self.assertEqual(MAX_FPS, 60)
        # OBJ_CONF default raised to 0.45 to reduce false positives on CPU
        self.assertGreaterEqual(OBJ_CONF, 0.40)
        self.assertLessEqual(OBJ_CONF, 0.55)

        # Test load_object_classes with dict format
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"classes": ["pen", "mug", "person"]}')
            f_path = f.name
        try:
            loaded = load_object_classes(f_path)
            self.assertEqual(loaded, ["pen", "mug", "person"])
        finally:
            os.remove(f_path)

        # Test load_object_classes with list format
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('["laptop", "keys"]')
            f_path = f.name
        try:
            loaded = load_object_classes(f_path)
            self.assertEqual(loaded, ["laptop", "keys"])
        finally:
            os.remove(f_path)

        # Test non-existent file fallback
        fallback = load_object_classes("non_existent_file.json")
        self.assertEqual(fallback, ["person"])


if __name__ == "__main__":
    unittest.main()
