"""Tests for BMO Contract Models, Serialization, and CV Conformance."""

import json
import pathlib
import re
import unittest
from unittest.mock import MagicMock

from pydantic import TypeAdapter, ValidationError

from assistant_cv.memory import ShortTermMemory
from assistant_cv.mqtt_bridge import MQTTBridge
from bmo_contracts.models import (
    EVENT_ADAPTER,
    BrainSay,
    BrainStatus,
    Emotion,
    EnrollArgs,
    FaceIdentified,
    ForgetArgs,
    HwCmd,
    HwEvent,
    HwStatus,
    MoveServoArgs,
    ObjectSeen,
    PersonDetected,
    PersonLeft,
    PersonName,
    PlaySoundArgs,
    Scene,
    SetExpressionArgs,
    VisionCmd,
    VisionCmdResult,
    VisionStatus,
    parse_event_json,
    validate_event,
)
from bmo_contracts.topics import (
    BROKER_HOST,
    BROKER_PORT,
    TOPICS,
    VISION_CONTEXT,
    VISION_EVENTS,
)

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
        self.assertGreaterEqual(len(self.json_blocks), 13, "Expected at least 13 JSON examples in Section 4.3")

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

        # 8. ObjectSeen Event
        ev_os = parse_event_json(self.json_blocks[7])
        self.assertIsInstance(ev_os, ObjectSeen)
        self.assertEqual(ev_os.data.label, "cup")

        # 9. Scene Context
        scene = Scene.model_validate_json(self.json_blocks[8])
        self.assertEqual(len(scene.people), 1)
        self.assertEqual(scene.people[0].name, "Alice")
        self.assertIn("cup", scene.objects)

        # 10. VisionCmd
        vc = VisionCmd.model_validate_json(self.json_blocks[9])
        self.assertEqual(vc.cmd, "reload_gallery")

        # 11. VisionCmdResult
        vcr = VisionCmdResult.model_validate_json(self.json_blocks[10])
        self.assertTrue(vcr.ok)

        # 12. BrainSay
        bsay = BrainSay.model_validate_json(self.json_blocks[11])
        self.assertEqual(bsay.emotion, Emotion.HAPPY)

        # 13. HwCmd
        hwcmd = HwCmd.model_validate_json(self.json_blocks[12])
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

        # 3. object_seen
        self.bridge.publish_event("object_seen", {"label": "pen", "conf": 0.85})
        self.assertEqual(len(self.published_messages), 3)
        ev3 = parse_event_json(self.published_messages[2][1])
        self.assertIsInstance(ev3, ObjectSeen)
        self.assertEqual(ev3.data.label, "pen")

    def test_topic_registry_coverage(self):
        """All 10 canonical topics exist in TOPICS registry with valid TopicSpec."""
        self.assertEqual(len(TOPICS), 10)
        for topic_name, spec in TOPICS.items():
            self.assertEqual(spec.name, topic_name)
            self.assertIn(spec.qos, (0, 1))
            self.assertIsInstance(spec.retain, bool)
            self.assertGreater(len(spec.pub_roles), 0)
            self.assertGreater(len(spec.sub_roles), 0)


class TestT1SchemaEnhancements(unittest.TestCase):
    """Test enhancements added in T1: versioning, typed args, PersonName, broker defaults."""

    def test_broker_defaults(self):
        from bmo_contracts.topics import BROKER_HOST as TOPICS_BROKER_HOST
        from bmo_contracts.topics import BROKER_PORT as TOPICS_BROKER_PORT

        self.assertEqual(BROKER_HOST, "127.0.0.1")
        self.assertEqual(BROKER_PORT, 1883)
        self.assertEqual(TOPICS_BROKER_HOST, "127.0.0.1")
        self.assertEqual(TOPICS_BROKER_PORT, 1883)

    def test_envelope_version_defaults_to_one(self):
        vs = VisionStatus(online=True)
        self.assertEqual(vs.v, 1)

        bs = BrainStatus(online=True, state="idle")
        self.assertEqual(bs.v, 1)

        hs = HwStatus(online=True, ts=1.0, capabilities=["lcd"])
        self.assertEqual(hs.v, 1)

        he = HwEvent(ts=1.0, kind="mute_switch", value=False)
        self.assertEqual(he.v, 1)

        sc = Scene(ts=1.0, people=[], objects=[], recent=[])
        self.assertEqual(sc.v, 1)

        vc = VisionCmd(id="cmd-1", cmd="reload_gallery")
        self.assertEqual(vc.v, 1)

        vcr = VisionCmdResult(id="cmd-1", ok=True, detail="ok")
        self.assertEqual(vcr.v, 1)

        bsay = BrainSay(utt_id="utt-1", text="hi", emotion=Emotion.NEUTRAL)
        self.assertEqual(bsay.v, 1)

        hwcmd = HwCmd(id="hw-1", target="pi", action="set_expression", args={"emotion": "neutral"})
        self.assertEqual(hwcmd.v, 1)

    def test_invalid_envelope_version_rejected(self):
        with self.assertRaises(ValidationError):
            VisionStatus.model_validate_json('{"v": 2, "online": true}')

        with self.assertRaises(ValidationError):
            BrainStatus.model_validate_json('{"v": 2, "online": true, "state": "idle"}')

        with self.assertRaises(ValidationError):
            HwStatus.model_validate_json('{"v": 99, "online": true, "ts": 1.0, "capabilities": []}')

        with self.assertRaises(ValidationError):
            HwEvent.model_validate_json('{"v": 0, "ts": 1.0, "kind": "mute_switch", "value": false}')

        with self.assertRaises(ValidationError):
            Scene.model_validate_json('{"v": 2, "ts": 1.0, "people": [], "objects": [], "recent": []}')

        with self.assertRaises(ValidationError):
            VisionCmd.model_validate_json('{"v": 2, "id": "1", "cmd": "reload_gallery"}')

        with self.assertRaises(ValidationError):
            VisionCmdResult.model_validate_json('{"v": 2, "id": "1", "ok": true, "detail": "ok"}')

        with self.assertRaises(ValidationError):
            BrainSay.model_validate_json('{"v": 2, "utt_id": "1", "text": "hi", "emotion": "neutral"}')

        with self.assertRaises(ValidationError):
            HwCmd.model_validate_json(
                '{"v": 2, "id": "1", "target": "pi", "action": "set_expression", "args": {"emotion": "happy"}}'
            )

    def test_vision_cmd_result_data(self):
        # Default is None
        res_default = VisionCmdResult(id="c1", ok=True, detail="ok")
        self.assertIsNone(res_default.data)

        # Accepts custom dict
        payload = {"gallery_size": 5, "identities": ["Alice", "Bob"]}
        res = VisionCmdResult(id="c1", ok=True, detail="ok", data=payload)
        self.assertEqual(res.data, payload)

        # Roundtrip JSON
        dumped = res.model_dump_json()
        loaded = VisionCmdResult.model_validate_json(dumped)
        self.assertEqual(loaded.data, payload)

        # Extra top-level fields still forbidden
        with self.assertRaises(ValidationError):
            VisionCmdResult.model_validate_json('{"id": "1", "ok": true, "detail": "ok", "extra_bogus": 123}')

    def test_person_name_validation(self):
        adapter = TypeAdapter(PersonName)

        # Valid names
        self.assertEqual(adapter.validate_python("Alice"), "Alice")
        self.assertEqual(adapter.validate_python("Bob-12_3"), "Bob-12_3")
        self.assertEqual(adapter.validate_python("Jane Doe"), "Jane Doe")
        self.assertEqual(adapter.validate_python("A" * 64), "A" * 64)

        # Invalid names
        with self.assertRaises(ValidationError):
            adapter.validate_python("../traversal")

        with self.assertRaises(ValidationError):
            adapter.validate_python("")

        with self.assertRaises(ValidationError):
            adapter.validate_python("A" * 65)

        with self.assertRaises(ValidationError):
            adapter.validate_python("User@Domain")

        with self.assertRaises(ValidationError):
            adapter.validate_python("/etc/passwd")

    def test_vision_cmd_person_name_enforcement(self):
        # Valid enroll & forget
        cmd_enroll = VisionCmd(id="1", cmd="enroll", args={"name": "Alice", "shots": 5})
        self.assertEqual(cmd_enroll.args["name"], "Alice")

        cmd_forget = VisionCmd(id="2", cmd="forget", args={"name": "Bob"})
        self.assertEqual(cmd_forget.args["name"], "Bob")

        # Typed models also supported
        enroll_typed = VisionCmd(id="1t", cmd="enroll", args=EnrollArgs(name="Charlie", shots=4))
        self.assertEqual(enroll_typed.args["name"], "Charlie")

        forget_typed = VisionCmd(id="2t", cmd="forget", args=ForgetArgs(name="David"))
        self.assertEqual(forget_typed.args["name"], "David")

        # Invalid names rejected in VisionCmd
        with self.assertRaises(ValidationError):
            VisionCmd(id="3", cmd="enroll", args={"name": "../traversal"})

        with self.assertRaises(ValidationError):
            VisionCmd(id="4", cmd="forget", args={"name": "../../bad"})

        with self.assertRaises(ValidationError):
            VisionCmd(id="5", cmd="enroll", args={"name": ""})

    def test_hw_cmd_set_expression_args_validation(self):
        # Valid emotion
        cmd = HwCmd(id="1", target="pi", action="set_expression", args={"emotion": "happy"})
        self.assertEqual(cmd.args["emotion"], Emotion.HAPPY)

        # Accepts "expression" key and normalizes to emotion
        cmd_expr = HwCmd(id="2", target="pi", action="set_expression", args={"expression": "happy"})
        self.assertEqual(cmd_expr.args["emotion"], Emotion.HAPPY)

        # Typed SetExpressionArgs
        typed_args = SetExpressionArgs(emotion=Emotion.SAD)
        cmd_typed = HwCmd(id="3", target="pi", action="set_expression", args=typed_args)
        self.assertEqual(cmd_typed.args["emotion"], Emotion.SAD)
        self.assertEqual(typed_args["emotion"], Emotion.SAD)
        self.assertEqual(typed_args.get_expression(), "sad")

        # Invalid emotion rejected
        with self.assertRaises(ValidationError):
            HwCmd(id="4", target="pi", action="set_expression", args={"emotion": "bogus"})

        # Extra fields forbidden
        with self.assertRaises(ValidationError):
            HwCmd(id="5", target="pi", action="set_expression", args={"emotion": "happy", "extra": True})

    def test_hw_cmd_move_servo_args_validation(self):
        # Valid servo
        cmd = HwCmd(id="1", target="esp32", action="move_servo", args={"id": "pan", "angle": 45.0})
        self.assertEqual(cmd.args["id"], "pan")
        self.assertEqual(cmd.args["angle"], 45.0)

        # Valid arms and tilt
        for sid in ("pan", "tilt", "left_arm", "right_arm"):
            cmd_arm = HwCmd(id="2", target="esp32", action="move_servo", args={"id": sid, "angle": 0})
            self.assertEqual(cmd_arm.args["id"], sid)

        # Typed MoveServoArgs
        typed_args = MoveServoArgs(id="tilt", angle=15.0)
        cmd_typed = HwCmd(id="3", target="esp32", action="move_servo", args=typed_args)
        self.assertEqual(cmd_typed.args["id"], "tilt")
        self.assertEqual(cmd_typed.args["angle"], 15.0)
        self.assertEqual(typed_args["id"], "tilt")

        # Invalid servo id rejected
        with self.assertRaises(ValidationError):
            HwCmd(id="4", target="esp32", action="move_servo", args={"id": "head", "angle": 0.0})

        # Extra fields forbidden
        with self.assertRaises(ValidationError):
            HwCmd(id="5", target="esp32", action="move_servo", args={"id": "pan", "angle": 0.0, "speed": 100})

    def test_hw_cmd_play_sound_args_validation(self):
        # Valid name
        cmd = HwCmd(id="1", target="pi", action="play_sound", args={"name": "chime"})
        self.assertEqual(cmd.args["name"], "chime")

        # Sound file name with dot (e.g. ping.wav)
        cmd_wav = HwCmd(id="2", target="pi", action="play_sound", args={"name": "ping.wav"})
        self.assertEqual(cmd_wav.args["name"], "ping.wav")

        # Accepts "sound" key and normalizes
        cmd_sound = HwCmd(id="3", target="pi", action="play_sound", args={"sound": "chime"})
        self.assertEqual(cmd_sound.args["name"], "chime")

        # Typed PlaySoundArgs
        typed_args = PlaySoundArgs(name="confirm")
        cmd_typed = HwCmd(id="4", target="pi", action="play_sound", args=typed_args)
        self.assertEqual(cmd_typed.args["name"], "confirm")
        self.assertEqual(typed_args["name"], "confirm")
        self.assertEqual(typed_args.get_sound(), "confirm")

        # Path traversal rejected
        with self.assertRaises(ValidationError):
            HwCmd(id="5", target="pi", action="play_sound", args={"name": "../traversal"})

        with self.assertRaises(ValidationError):
            HwCmd(id="6", target="pi", action="play_sound", args={"name": "dir/sound.wav"})

        # Extra fields forbidden
        with self.assertRaises(ValidationError):
            HwCmd(id="7", target="pi", action="play_sound", args={"name": "chime", "volume": 10})


if __name__ == "__main__":
    unittest.main()
