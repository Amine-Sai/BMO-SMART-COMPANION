import hashlib
import json
from pathlib import Path

import numpy as np

from brain.logging_jsonl import JSONLLogger
from brain.privacy import MAX_NUMERIC_ARRAY_LEN, sanitize_payload


def test_privacy_redaction_and_jsonl_logging(tmp_path: Path) -> None:
    assert MAX_NUMERIC_ARRAY_LEN == 32

    log_path = tmp_path / "test.jsonl"
    logger = JSONLLogger(log_path=log_path)

    rec = logger.log(
        "face_check",
        embedding=[0.123456] * 512,
        audio=b"\x00\x01" * 320,
        frame=np.zeros((320,), dtype=np.int16),
        safe_meta={"name": "Jud", "bbox": [10, 20, 30, 40]},
        unregistered_field="should_be_dropped",
    )
    logger.close()

    jud_hash = hashlib.sha256(b"Jud").hexdigest()[:12]
    assert rec["embedding"] == "<redacted:numeric_array:len=512>"
    assert rec["audio"] == "<redacted:bytes:len=640>"
    assert "<redacted:ndarray:shape=(320,):dtype=int16>" == rec["frame"]
    assert rec["safe_meta"] == {"name": jud_hash, "bbox": [10, 20, 30, 40]}
    assert "unregistered_field" not in rec

    content = log_path.read_text(encoding="utf-8")
    assert "<redacted:numeric_array:len=512>" in content
    assert "<redacted:bytes:len=640>" in content
    assert "<redacted:ndarray:" in content
    assert "0.123456" not in content
    assert f'"name":"{jud_hash}"' in content
    assert "Jud" not in content
    assert "[10,20,30,40]" in content
    assert "should_be_dropped" not in content


def test_transcript_redaction(tmp_path: Path) -> None:
    log_path = tmp_path / "transcript.jsonl"
    logger = JSONLLogger(log_path=log_path)

    text = "Turn on living room light"
    rec = logger.log(
        "transcript",
        text=text,
        utt_id="utt-42",
        conf=0.98,
        unallowed_leak="secret_token",
    )
    logger.close()

    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    assert rec["text"] == expected_hash
    assert rec["utt_id"] == "utt-42"
    assert rec["conf"] == 0.98
    assert "unallowed_leak" not in rec

    content = log_path.read_text(encoding="utf-8")
    assert expected_hash in content
    assert text not in content
    assert "secret_token" not in content


def test_tool_actions_and_identity_redaction(tmp_path: Path) -> None:
    log_path = tmp_path / "tools.jsonl"
    logger = JSONLLogger(log_path=log_path)

    # 1. Direct remember_person event
    rec_rem = logger.log("remember_person", name="Alice", ok=True)
    alice_hash = hashlib.sha256(b"Alice").hexdigest()[:12]
    assert rec_rem["name"] == alice_hash
    assert rec_rem["ok"] is True

    # 2. Direct forget_person event
    rec_for = logger.log("forget_person", name="Bob", ok=True)
    bob_hash = hashlib.sha256(b"Bob").hexdigest()[:12]
    assert rec_for["name"] == bob_hash

    # 3. tool_action with remember_person
    rec_act = logger.log(
        "tool_action",
        tool="remember_person",
        arguments={"name": "Charlie"},
        ok=True,
    )
    charlie_hash = hashlib.sha256(b"Charlie").hexdigest()[:12]
    assert rec_act["arguments"]["name"] == charlie_hash
    assert rec_act["tool"] == "remember_person"

    # 4. tool_action with non-person tool (e.g. play_sound) -> name not hashed
    rec_snd = logger.log(
        "tool_action",
        tool="play_sound",
        arguments={"name": "chime"},
        ok=True,
    )
    assert rec_snd["arguments"]["name"] == "chime"

    # 5. tool_call with tool name -> name not hashed
    rec_call = logger.log(
        "tool_call",
        name="move_head",
        arguments={"pan": -15.0},
    )
    assert rec_call["name"] == "move_head"

    logger.close()

    content = log_path.read_text(encoding="utf-8")
    assert alice_hash in content
    assert "Alice" not in content
    assert bob_hash in content
    assert "Bob" not in content
    assert charlie_hash in content
    assert "Charlie" not in content
    assert '"name":"chime"' in content
    assert '"name":"move_head"' in content


def test_unknown_fields_and_events_dropped(tmp_path: Path) -> None:
    log_path = tmp_path / "unknown.jsonl"
    logger = JSONLLogger(log_path=log_path)

    # Unknown fields in known event
    rec_lat = logger.log(
        "turn_latency",
        total_ms=120.0,
        utt_id="turn-9",
        rogue_field="leak_val",
    )
    assert "total_ms" in rec_lat
    assert "utt_id" in rec_lat
    assert "rogue_field" not in rec_lat

    # Completely unknown event
    rec_unk = logger.log("unregistered_event", foo="bar", num=42)
    assert rec_unk["event"] == "unregistered_event"
    assert "foo" not in rec_unk
    assert "num" not in rec_unk
    assert "ts" in rec_unk

    logger.close()
    content = log_path.read_text(encoding="utf-8")
    assert "rogue_field" not in content
    assert "leak_val" not in content
    assert "foo" not in content
    assert "bar" not in content


def test_non_serializable_values_do_not_crash(tmp_path: Path) -> None:
    log_path = tmp_path / "nonserial.jsonl"
    logger = JSONLLogger(log_path=log_path)

    class CustomObj:
        pass

    class BadStrObj:
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    circular: dict[str, object] = {"index": 1}
    circular["self"] = circular

    # None of these calls should crash the logger
    logger.log("step", index=1, obj=CustomObj())
    logger.log("step", index=2, obj=BadStrObj())
    logger.log("step", index=3, obj=lambda x: x)
    logger.log("step", index=4, data=circular)
    logger.close()

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 4
    assert lines[0]["index"] == 1
    assert "CustomObj" in lines[0]["obj"]
    assert lines[1]["index"] == 2
    assert "<unserializable:BadStrObj>" == lines[1]["obj"]
    assert lines[2]["index"] == 3
    assert "<function" in lines[2]["obj"]
    assert lines[3]["index"] == 4
    assert lines[3]["data"]["self"] == "<circular_ref>"


def test_redact_pii_flag_disabled(tmp_path: Path) -> None:
    log_path = tmp_path / "raw.jsonl"
    logger = JSONLLogger(log_path=log_path, redact_pii=False)

    rec = logger.log("face_check", safe_meta={"name": "Jud"})
    rec_t = logger.log("transcript", text="Raw speech text")
    logger.close()

    assert rec["safe_meta"]["name"] == "Jud"
    assert rec_t["text"] == "Raw speech text"

    content = log_path.read_text(encoding="utf-8")
    assert '"name":"Jud"' in content
    assert "Raw speech text" in content


def test_file_rotation(tmp_path: Path) -> None:
    log_path = tmp_path / "test_rot.jsonl"
    logger = JSONLLogger(log_path=log_path, max_bytes=256, backup_count=3)

    for i in range(10):
        logger.log("step", index=i, filler="x" * 60)
    logger.close()

    backup_file = tmp_path / "test_rot.jsonl.1"
    assert backup_file.exists()
    assert backup_file.stat().st_size > 0


def test_sanitize_payload_cases() -> None:
    assert sanitize_payload("hello") == "hello"
    assert sanitize_payload(123) == 123
    assert sanitize_payload([True] * 40) == [True] * 40
    assert sanitize_payload([1, 2, 3]) == [1, 2, 3]
