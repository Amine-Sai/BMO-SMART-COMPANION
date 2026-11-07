"""Unit and integration tests for ObjectRegistry, AutoObjectEnrollSession, and dynamic object registration."""

import json
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

from assistant_cv.enroll import AutoObjectEnrollSession
from assistant_cv.objects import ObjectDetector
from assistant_cv.registry import ObjectRegistry


class DummyDetector:
    def __init__(self):
        self.class_names = ["person", "soda can"]
        self.registered_classes = []
        self.counter = 0

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        vec = np.zeros(512, dtype=np.float32)
        vec[self.counter % 512] = 1.0
        self.counter += 1
        return vec

    def register_class(self, name: str, synonyms: list[str] | None = None):
        self.registered_classes.append(name)
        if name not in self.class_names:
            self.class_names.append(name)

    def reload_classes(self):
        pass


class TestObjectRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gallery_dir = pathlib.Path(self.temp_dir.name)
        self.classes_file = self.gallery_dir / "test_classes.json"
        with open(self.classes_file, "w") as f:
            json.dump({"classes": ["person", "soda can"]}, f)
        self.registry = ObjectRegistry(
            gallery_dir=self.gallery_dir,
            classes_path=self.classes_file,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_register_and_persist(self):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        ok, msg, path = self.registry.register_object(
            "lighter",
            synonyms=["cigarette lighter", "bic lighter"],
            embedding=emb,
        )
        self.assertTrue(ok)
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        self.assertTrue(self.registry.is_registered("lighter"))
        self.assertTrue(self.registry.is_registered("cigarette lighter"))

        # Verify synced into classes file
        with open(self.classes_file) as f:
            cdata = json.load(f)
        self.assertIn("lighter", cdata["classes"])
        self.assertIn("cigarette lighter", cdata["classes"])

        # Reload in a new instance from disk
        reg2 = ObjectRegistry(
            gallery_dir=self.gallery_dir,
            classes_path=self.classes_file,
        )
        self.assertTrue(reg2.is_registered("lighter"))
        self.assertEqual(len(reg2._gallery_names), 1)
        self.assertEqual(reg2._gallery_names[0], "lighter")

    def test_visual_matching(self):
        emb = np.zeros(512, dtype=np.float32)
        emb[42] = 1.0

        self.registry.register_object("Special_Mug", embedding=emb)

        # Match identical vector
        matched_name, sim = self.registry.match_visual(emb, tau=0.65)
        self.assertEqual(matched_name, "Special_Mug")
        self.assertGreater(sim, 0.99)

        # Non-matching vector
        diff_emb = np.zeros(512, dtype=np.float32)
        diff_emb[0] = 1.0
        diff_name, diff_sim = self.registry.match_visual(diff_emb, tau=0.65)
        self.assertIsNone(diff_name)
        self.assertLess(diff_sim, 0.65)

    def test_rename_and_delete(self):
        emb = np.random.randn(512).astype(np.float32)
        self.registry.register_object("old_tool", embedding=emb)
        self.assertTrue(self.registry.is_registered("old_tool"))

        ok, _ = self.registry.rename_object("old_tool", "new_tool")
        self.assertTrue(ok)
        self.assertFalse(self.registry.is_registered("old_tool"))
        self.assertTrue(self.registry.is_registered("new_tool"))
        self.assertTrue((self.gallery_dir / "new_tool.npy").exists())

        ok, _ = self.registry.delete_object("new_tool")
        self.assertTrue(ok)
        self.assertFalse(self.registry.is_registered("new_tool"))
        self.assertFalse((self.gallery_dir / "new_tool.npy").exists())


class TestAutoObjectEnrollSession(unittest.TestCase):
    def test_object_enroll_flow(self):
        temp_dir = tempfile.TemporaryDirectory()
        try:
            reg = ObjectRegistry(
                gallery_dir=pathlib.Path(temp_dir.name),
                classes_path=pathlib.Path(temp_dir.name) / "classes.json",
            )
            detector = DummyDetector()
            session = AutoObjectEnrollSession(
                name="lighter",
                target_shots=3,
                min_interval_sec=0.01,
                timeout_sec=5.0,
                synonyms=["bic lighter"],
                detector=detector,
                registry=reg,
            )

            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = [{"label": "lighter", "conf": 0.85, "bbox": (200, 150, 260, 280), "id": 1}]

            for i in range(3):
                # Create slightly different frame patches to ensure visual diversity
                dummy_frame[150:280, 200:260] = (i + 1) * 40
                active = session.step(dummy_frame, detections)
                time.sleep(0.015)

            self.assertTrue(session.completed)
            self.assertFalse(session.failed)
            self.assertTrue(reg.is_registered("lighter"))
            self.assertIn("lighter", detector.registered_classes)
            self.assertTrue((reg.gallery_dir / "lighter.npy").exists())
            self.assertTrue((reg.gallery_dir / "lighter.jpg").exists())
        finally:
            temp_dir.cleanup()


class TestVocabularyAugmentation(unittest.TestCase):
    def test_lighter_and_contrasting_items_present(self):
        classes_path = BASE_DIR / "assistant_cv" / "everyday_objects.json"
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", [])

        # Verify lighter and common everyday EDC items are in base vocabulary
        self.assertIn("lighter", classes)
        self.assertIn("cigarette lighter", classes)
        self.assertIn("vape", classes)
        self.assertIn("flashlight", classes)
        self.assertIn("battery", classes)
        self.assertIn("soda can", classes)
        self.assertIn("coffee mug", classes)


if __name__ == "__main__":
    unittest.main()
