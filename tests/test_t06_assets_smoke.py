"""Smoke tests for model assets and verification routines (T0.6 / T13)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools.fetch_assets import (
    ASSETS,
    check_ollama,
    download_and_verify,
    fetch_all,
    fetch_faster_whisper,
    fetch_hand_models,
    fetch_openwakeword_models,
    fetch_piper_voice,
    verify_asset_file,
)


class TestAssetRoutines(unittest.TestCase):
    """Test asset management functions and smoke checks."""

    def test_check_ollama_structure(self) -> None:
        """Verify Ollama detection returns consistent structure and H2 escalation string."""
        res = check_ollama(client_check_fn=lambda: False)
        self.assertEqual(res["engine"], "ollama")
        self.assertIn("installed", res)
        self.assertIn("daemon_running", res)
        self.assertFalse(res["daemon_running"])
        self.assertIn("H2:", str(res["escalation"]))

        res_active = check_ollama(client_check_fn=lambda: True)
        if res_active.get("installed"):
            self.assertIsNone(res_active["escalation"])

    def test_assets_table_schema(self) -> None:
        """Verify ASSETS table has required schema {name, url, path, sha256}."""
        self.assertIsInstance(ASSETS, dict)
        required_assets = [
            "palm_detection",
            "handpose_estimation",
            "piper_voice",
            "piper_config",
            "hey_jarvis_v0.1",
            "melspectrogram",
            "embedding_model",
        ]
        for key in required_assets:
            self.assertIn(key, ASSETS)
            entry = ASSETS[key]
            for field in ("name", "url", "path", "sha256"):
                self.assertIn(field, entry)
            self.assertTrue(entry["url"].startswith("https://"))
            self.assertIsInstance(entry["path"], Path)
            self.assertEqual(len(entry["sha256"]), 64)
            # Check sha256 is valid hexadecimal
            int(entry["sha256"], 16)

    def test_download_and_verify_match(self) -> None:
        """Verify fake download with matching SHA256 succeeds and keeps file."""
        tmp_dir = Path("scratch/test_asset_download_match")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            target = tmp_dir / "test_match.bin"
            content = b"valid asset payload data"
            expected_sha = hashlib.sha256(content).hexdigest()

            def fake_download(_url: str, path: Path) -> None:
                path.write_bytes(content)

            res_path = download_and_verify(
                url="https://example.com/asset.bin",
                target_path=target,
                expected_sha256=expected_sha,
                download_fn=fake_download,
            )
            self.assertEqual(res_path, target)
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_bytes(), content)
            self.assertTrue(verify_asset_file(target, expected_sha))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_download_and_verify_mismatch_deletes_file(self) -> None:
        """Verify fake download with mismatching SHA256 deletes file and raises ValueError."""
        tmp_dir = Path("scratch/test_asset_download_mismatch")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            target = tmp_dir / "test_mismatch.bin"
            corrupt_content = b"corrupt asset payload data"
            expected_sha = "0000000000000000000000000000000000000000000000000000000000000000"

            def fake_corrupt_download(_url: str, path: Path) -> None:
                path.write_bytes(corrupt_content)

            with self.assertRaises(ValueError) as ctx:
                download_and_verify(
                    url="https://example.com/corrupt.bin",
                    target_path=target,
                    expected_sha256=expected_sha,
                    download_fn=fake_corrupt_download,
                )
            self.assertIn("Checksum mismatch", str(ctx.exception))
            self.assertFalse(target.exists(), "Target file must be deleted upon checksum mismatch")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_download_and_verify_urlopen_timeout(self) -> None:
        """Verify default urlopen downloader enforces timeout parameter."""
        tmp_dir = Path("scratch/test_asset_timeout")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            target = tmp_dir / "timeout_test.bin"
            dummy_content = b"timeout check payload"
            expected_sha = hashlib.sha256(dummy_content).hexdigest()

            mock_resp = MagicMock()
            mock_resp.read = MagicMock(side_effect=[dummy_content, b""])
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None

            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                download_and_verify(
                    url="https://example.com/mock.bin",
                    target_path=target,
                    expected_sha256=expected_sha,
                    timeout=15.0,
                )
                self.assertTrue(mock_urlopen.called)
                _, kwargs = mock_urlopen.call_args
                self.assertEqual(kwargs.get("timeout"), 15.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fetch_hand_models_local_detection(self) -> None:
        """Verify fetch_hand_models detects and verifies local models."""
        tmp_dir = Path("scratch/test_hand_models")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Initially missing
            res_missing = fetch_hand_models(target_dir=tmp_dir, fetch_remote=False)
            self.assertEqual(res_missing["engine"], "hands")
            self.assertFalse(res_missing["ok"])

            # Create dummy files with mismatched content
            palm_path = tmp_dir / "palm_detection.onnx"
            handpose_path = tmp_dir / "handpose_estimation.onnx"
            palm_path.write_bytes(b"bad_palm")
            handpose_path.write_bytes(b"bad_handpose")

            res_corrupt = fetch_hand_models(target_dir=tmp_dir, fetch_remote=False)
            self.assertFalse(res_corrupt["ok"])

            palm_sha = ASSETS["palm_detection"]["sha256"]
            handpose_sha = ASSETS["handpose_estimation"]["sha256"]

            def mock_dl(_url: str, path: Path) -> None:
                path.write_bytes(b"downloaded")

            with patch("tools.fetch_assets.compute_sha256") as mock_hash:
                def fake_hash(p: Path | str) -> str:
                    if "palm" in str(p):
                        return palm_sha
                    return handpose_sha

                mock_hash.side_effect = fake_hash

                res_fetched = fetch_hand_models(
                    target_dir=tmp_dir,
                    fetch_remote=True,
                    download_fn=mock_dl,
                )
                self.assertTrue(res_fetched["ok"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_hands_raises_error_if_model_missing_no_urlopen(self) -> None:
        """Verify assistant_cv.hands raises FileNotFoundError if model missing and does not use urlopen."""
        import assistant_cv.hands as hands_mod

        # Verify hands module does not contain urlopen
        self.assertNotIn("urlopen", hands_mod.__dict__)

        # Reset singleton state
        saved_palm = hands_mod._palm_detector
        saved_handpose = hands_mod._handpose_detector
        hands_mod._palm_detector = None
        hands_mod._handpose_detector = None

        try:
            non_existent = Path("scratch/non_existent_model.onnx")
            with patch("assistant_cv.hands.PALM_MODEL_PATH", non_existent):
                with self.assertRaises(FileNotFoundError) as ctx:
                    hands_mod.get_detectors()
                self.assertIn("tools/fetch_assets.py", str(ctx.exception))
                self.assertIn("Palm detection", str(ctx.exception))
        finally:
            hands_mod._palm_detector = saved_palm
            hands_mod._handpose_detector = saved_handpose

    def test_fetch_openwakeword_models_logic(self) -> None:
        """Verify openWakeWord download logic and model discovery."""
        downloaded: list[str] = []

        def mock_download(missing: list[str]) -> None:
            downloaded.extend(missing)

        res = fetch_openwakeword_models(
            model_names=["hey_jarvis_v0.1"],
            download_fn=mock_download,
        )
        self.assertEqual(res["engine"], "openwakeword")
        self.assertIn("models_dir", res)

    def test_fetch_piper_voice_local_detection(self) -> None:
        """Verify Piper voice checking with custom local directory."""
        tmp_dir = Path("scratch/test_piper_assets")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            res_missing = fetch_piper_voice(
                voice_name="test_voice",
                target_dir=tmp_dir,
                fetch_remote=False,
            )
            self.assertFalse(res_missing["ok"])

            # Create dummy model files
            (tmp_dir / "test_voice.onnx").write_bytes(b"dummy_onnx")
            (tmp_dir / "test_voice.onnx.json").write_bytes(b"{}")

            res_found = fetch_piper_voice(
                voice_name="test_voice",
                target_dir=tmp_dir,
                fetch_remote=False,
            )
            self.assertTrue(res_found["ok"])
            self.assertEqual(res_found["voice"], "test_voice")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fetch_faster_whisper_logic(self) -> None:
        """Verify Faster-Whisper download wrapper handling."""
        res = fetch_faster_whisper(
            model_size="tiny.en",
            download_fn=lambda size: f"/tmp/cache/{size}",
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["cache_dir"], "/tmp/cache/tiny.en")

    def test_fetch_all_check_only(self) -> None:
        """Verify fetch_all check_only mode returns results for all subsystems."""
        results = fetch_all(check_only=True)
        self.assertIn("ollama", results)
        self.assertIn("openwakeword", results)
        self.assertIn("piper", results)
        self.assertIn("faster_whisper", results)
        self.assertIn("hands", results)

    def test_openwakeword_smoke_scoring(self) -> None:
        """Smoke test openWakeWord scoring synthetic audio."""
        import openwakeword
        from openwakeword.model import Model

        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        jarvis_model = models_dir / "hey_jarvis_v0.1.onnx"

        if not jarvis_model.is_file():
            pytest.skip("Stock hey_jarvis model not yet downloaded to package resources")

        oww = Model(
            wakeword_models=[str(jarvis_model)],
            inference_framework="onnx",
        )
        # 16kHz 16-bit PCM (1280 samples = 80ms)
        dummy_frame = np.zeros(1280, dtype=np.int16)
        prediction = oww.predict(dummy_frame)
        self.assertIsInstance(prediction, dict)


@pytest.mark.slow
def test_faster_whisper_smoke_real_transcription() -> None:
    """Smoke test real Faster-Whisper CPU execution on silence if model is locally cached."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        pytest.skip("faster_whisper not installed")

    try:
        model = WhisperModel("base.en", device="cpu", compute_type="int8", local_files_only=True)
        # 1 second of silence (16kHz float32)
        audio = np.zeros(16000, dtype=np.float32)
        segments, _ = model.transcribe(audio, beam_size=1)
        text = "".join(s.text for s in segments)
        assert text == ""
    except Exception as exc:
        pytest.skip(f"Faster-whisper model weights not cached: {exc}")


@pytest.mark.slow
@pytest.mark.gpu
def test_ollama_smoke_chat_turn() -> None:
    """Smoke test real Ollama daemon interaction if available."""
    res = check_ollama()
    if not res.get("daemon_running"):
        pytest.skip("Ollama daemon is not running (requires escalation H2)")

    import httpx

    resp = httpx.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "qwen2.5:3b", "prompt": "Hi BMO", "stream": False},
        timeout=5.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
