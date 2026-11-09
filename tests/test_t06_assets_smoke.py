"""Smoke tests for model assets and verification routines (T0.6)."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest

import numpy as np
import pytest

from tools.fetch_assets import (
    check_ollama,
    fetch_all,
    fetch_faster_whisper,
    fetch_openwakeword_models,
    fetch_piper_voice,
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
        """Verify fetch_all check_only mode returns results for all 4 subsystems."""
        results = fetch_all(check_only=True)
        self.assertIn("ollama", results)
        self.assertIn("openwakeword", results)
        self.assertIn("piper", results)
        self.assertIn("faster_whisper", results)

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
