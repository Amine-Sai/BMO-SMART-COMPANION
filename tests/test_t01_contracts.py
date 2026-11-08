"""Tests for BMO Contract Models, Serialization, and CV Conformance."""

import json
import pathlib
import re
import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from assistant_cv.memory import ShortTermMemory
from assistant_cv.mqtt_bridge import MQTTBridge
from bmo_contracts.models import (
    EVENT_ADAPTER,
    BrainSay,
    BrainStatus,
    Emotion,
    FaceIdentified,
    HwCmd,
    HwEvent,
    HwStatus,
    ObjectSeen,
    PersonDetected,
    PersonLeft,
    Scene,
    SignRecognized,
    VisionCmd,
    VisionCmdResult,
    VisionStatus,
    parse_event_json,
    validate_event,
)
from bmo_contracts.topics import TOPICS, VISION_CONTEXT, VISION_EVENTS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestSection43Examples(unittest.TestCase):
    """Verify that all verbatim JSON examples from docs/contracts.md pass Pydantic validation."""

    def setUp(self):
        contracts_doc = REPO_ROOT / "docs" / "contracts.md"
        self.assertTrue(contracts_doc.is_file(), "docs/contracts.md must exist")
        content = contracts_doc.read_text(encoding="utf-8")

        # Extract JSON blocks under Section 4.3
        sec_match = re.search(r"## 4\.3 Verbatim JSON Payload Examples.*", content, re.DOTALL)
        self.assertIsNotNone(sec_match, "Section 4.3 missing in docs/contracts.md")
        sec_text = sec_match.group(0)

        self.json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", sec_text, re.DOTALL)
        self.assertGreaterEqual(len(self.json_blocks), 14, "Expected at least 14 JSON examples in Section 4.3")

    def test_all_verbatim_examples_validate(self):
        """Validate each Section 4.3 verbatim JSON example against its corresponding model."""
        # 1. VisionStatus
        vs = VisionStatus.model_validate_json(self.json_blocks[0])
        self.assertTrue(vs.online)

        # 2. BrainStatus
        bs = BrainStatus.model_validate_json(self.json_blocks[1])
        self.assertTrue(bs.online)
        self.assertEqual(bs.state, "listening")

        # 3. HwStatus
        hs = HwStatus.model_validate_json(self.json_blocks[2])
        self.assertTrue(hs.online)
        self.assertIn("display", hs.capabilities)

        # 4. HwEvent
        he = HwEvent.model_validate_json(self.json_blocks[3])
        self.assertEqual(he.kind, "mute_switch")
        self.assertTrue(he.value)

        # 5. FaceIdentified Event
        ev_face = parse_event_json(self.json_blocks[4])
        self.assertIsInstance(ev_face, FaceIdentified)
        self.assertEqual(ev_face.data.name, "Alice")
        ev_val = validate_event(json.loads(self.json_blocks[4]))
        self.assertIsInstance(ev_val, FaceIdentified)

        # 6. PersonDetected Event
        ev_p = parse_event_json(self.json_blocks[5])
        self.assertIsInstance(ev_p, PersonDetected)
        self.assertEqual(ev_p.data.proximity, "near")

        # 7. PersonLeft Event
        ev_pl = parse_event_json(self.json_blocks[6])
        self.assertIsInstance(ev_pl, PersonLeft)
        self.assertEqual(ev_pl.data.track_id, 1)

        # 8. SignRecognized Event
        ev_sr = parse_event_json(self.json_blocks[7])
        self.assertIsInstance(ev_sr, SignRecognized)
        self.assertEqual(ev_sr.data.text, "hello")

        # 9. ObjectSeen Event
        ev_os = parse_event_json(self.json_blocks[8])
        self.assertIsInstance(ev_os, ObjectSeen)
        self.assertEqual(ev_os.data.label, "cup")

        # 10. Scene Context
        scene = Scene.model_validate_json(self.json_blocks[9])
        self.assertEqual(len(scene.people), 1)
        self.assertEqual(scene.people[0].name, "Alice")
        self.assertIn("cup", scene.objects)

        # 11. VisionCmd
        vc = VisionCmd.model_validate_json(self.json_blocks[10])
        self.assertEqual(vc.cmd, "reload_gallery")

        # 12. VisionCmdResult
        vcr = VisionCmdResult.model_validate_json(self.json_blocks[11])
        self.assertTrue(vcr.ok)

        # 13. BrainSay
        bsay = BrainSay.model_validate_json(self.json_blocks[12])
        self.assertEqual(bsay.emotion, Emotion.HAPPY)

        # 14. HwCmd
        hwcmd = HwCmd.model_validate_json(self.json_blocks[13])
        self.assertEqual(hwcmd.target, "pi")
        self.assertEqual(hwcmd.action, "set_expression")


class TestContractValidationStrictness(unittest.TestCase):
    """Verify that extra-field and malformed payloads strictly raise ValidationError."""

    def test_extra_fields_rejected(self):
        # Top-level extra field on Model
        with self.assertRaises(ValidationError):
            VisionStatus.model_validate_json('{"online": true, "bogus_extra": 123}')

        with self.assertRaises(ValidationError):
            Scene.model_validate_json(
                '{"ts": 1.0, "people": [], "objects": [], "recent": [], "extra_field": "disallowed"}'
            )

        # Event extra fields
        with self.assertRaises(ValidationError):
            EVENT_ADAPTER.validate_json(
                '{"v": 1, "id": "evt-01", "ts": 1.0, "type": "person_left", "data": {"track_id": 1}, "extra": true}'
            )

        # Event data extra fields
        with self.assertRaises(ValidationError):
            EVENT_ADAPTER.validate_json(
                '{"v": 1, "id": "evt-01", "ts": 1.0, "type": "person_left", "data": {"track_id": 1, "extra": true}}'
            )

    def test_malformed_values_rejected(self):
        with self.assertRaises(ValidationError):
            VisionStatus.model_validate_json('{"online": "invalid_boolean"}')

        with self.assertRaises(ValidationError):
            BrainStatus.model_validate_json('{"online": true, "state": "dancing"}')

        with self.assertRaises(ValidationError):
            EVENT_ADAPTER.validate_json(
                '{"v": 1, "id": "evt-01", "ts": 1.0, "type": "nonexistent_event_type", "data": {}}'
            )


class TestShortTermMemorySceneContract(unittest.TestCase):
    """Verify ShortTermMemory.get_scene_contract() conforms to Scene schema and <= 300 token budget."""

    def test_scene_contract_generation(self):
        sim_time = 1000.0
        mem = ShortTermMemory(window_sec=60.0, clock=lambda: sim_time)

        # Add initial detections
        detections = [
            {"label": "person", "conf": 0.95, "id": 10, "identity": "Alice", "bbox": (100, 100, 400, 400)},
            {"label": "cup", "conf": 0.88, "bbox": (50, 50, 80, 80)},
        ]
        mem.update(detections, frame_shape=(480, 640))
        mem.record_sign("HELLO", conf=0.94)

        sim_time = 1005.0
        scene_dict = mem.get_scene_contract()

        # Validate against Scene schema
        scene = Scene.model_validate(scene_dict)
        self.assertEqual(scene.ts, 1005.0)
        self.assertEqual(len(scene.people), 1)
        self.assertEqual(scene.people[0].name, "Alice")
        self.assertEqual(scene.people[0].proximity, "near")
        self.assertEqual(scene.people[0].since_s, 5.0)
        self.assertIn("cup", scene.objects)
        self.assertIsNotNone(scene.last_sign)
        self.assertEqual(scene.last_sign.text, "hello")
        self.assertEqual(scene.last_sign.age_s, 5.0)

        # Token budget test: serialized JSON must be <= 300 tokens
        # Conservative token estimation: ~3.5 chars per token for English JSON
        raw_json = json.dumps(scene_dict)
        char_count = len(raw_json)
        estimated_tokens = char_count / 3.0
        self.assertLess(estimated_tokens, 300, f"Scene JSON tokens exceed budget: ~{estimated_tokens:.0f} tokens")


class TestMQTTBridgeContractConformance(unittest.TestCase):
    """Verify MQTTBridge emits strictly valid Event and Scene payloads."""

    def setUp(self):
        self.bridge = MQTTBridge(clock=lambda: 1700000000.0)
        self.bridge.connected = True
        self.bridge.client = MagicMock()
        self.published_messages: list[tuple[str, bytes, int, bool]] = []

        def mock_pub(topic, payload, qos=0, retain=False):
            self.published_messages.append((topic, payload, qos, retain))

        self.bridge.client.publish = mock_pub

    def test_publish_scene_contract(self):
        scene_data = {
            "ts": 1700000000.0,
            "people": [{"track_id": 1, "name": "Bob", "pos": "center", "proximity": "near", "since_s": 2.5}],
            "objects": ["bottle"],
            "last_sign": None,
            "recent": ["person detected (track: 1)"],
        }
        self.bridge.publish_scene(scene_data)
        self.assertEqual(len(self.published_messages), 1)
        topic, payload, qos, retain = self.published_messages[0]
        self.assertEqual(topic, VISION_CONTEXT)
        self.assertEqual(qos, 0)
        self.assertTrue(retain)

        validated_scene = Scene.model_validate_json(payload)
        self.assertEqual(validated_scene.people[0].name, "Bob")

    def test_publish_event_contracts(self):
        # 1. person_detected with close proximity normalized to near
        self.bridge.publish_event("person_detected", {"track_id": 5, "pos": "left", "proximity": "close"})
        self.assertEqual(len(self.published_messages), 1)
        topic, payload, qos, retain = self.published_messages[0]
        self.assertEqual(topic, VISION_EVENTS)
        self.assertEqual(qos, 1)
        self.assertFalse(retain)

        ev1 = parse_event_json(payload)
        self.assertIsInstance(ev1, PersonDetected)
        self.assertEqual(ev1.data.proximity, "near")
        self.assertEqual(ev1.data.track_id, 5)

        # 2. face_identified
        self.bridge.publish_event("face_identified", {"track_id": 5, "name": "Alice", "conf": 0.95})
        self.assertEqual(len(self.published_messages), 2)
        ev2 = parse_event_json(self.published_messages[1][1])
        self.assertIsInstance(ev2, FaceIdentified)
        self.assertEqual(ev2.data.name, "Alice")

        # 3. sign_recognized
        self.bridge.publish_event("sign_recognized", {"sign": "THANKS", "conf": 0.91})
        self.assertEqual(len(self.published_messages), 3)
        ev3 = parse_event_json(self.published_messages[2][1])
        self.assertIsInstance(ev3, SignRecognized)
        self.assertEqual(ev3.data.text, "thanks")

        # 4. object_seen
        self.bridge.publish_event("object_seen", {"label": "pen", "conf": 0.85})
        self.assertEqual(len(self.published_messages), 4)
        ev4 = parse_event_json(self.published_messages[3][1])
        self.assertIsInstance(ev4, ObjectSeen)
        self.assertEqual(ev4.data.label, "pen")

    def test_topic_registry_coverage(self):
        """All 10 canonical topics exist in TOPICS registry with valid TopicSpec."""
        self.assertEqual(len(TOPICS), 10)
        for topic_name, spec in TOPICS.items():
            self.assertEqual(spec.name, topic_name)
            self.assertIn(spec.qos, (0, 1))
            self.assertIsInstance(spec.retain, bool)
            self.assertGreater(len(spec.pub_roles), 0)
            self.assertGreater(len(spec.sub_roles), 0)


if __name__ == "__main__":
    unittest.main()
