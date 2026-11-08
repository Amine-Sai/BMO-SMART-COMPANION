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
    )
    logger.close()

    assert rec["embedding"] == "<redacted:numeric_array:len=512>"
    assert rec["audio"] == "<redacted:bytes:len=640>"
    assert "<redacted:ndarray:shape=(320,):dtype=int16>" == rec["frame"]
    assert rec["safe_meta"] == {"name": "Jud", "bbox": [10, 20, 30, 40]}

    content = log_path.read_text(encoding="utf-8")
    assert "<redacted:numeric_array:len=512>" in content
    assert "<redacted:bytes:len=640>" in content
    assert "<redacted:ndarray:" in content
    assert "0.123456" not in content
    assert '"name":"Jud"' in content
    assert "[10,20,30,40]" in content


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
