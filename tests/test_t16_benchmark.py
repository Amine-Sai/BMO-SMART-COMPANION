"""Tests for LLM benchmarking, gate checks, and model selection (T1.6)."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest

import pytest
import yaml

from evals.run import grounded_baseline_responder
from tools.benchmark_llm import (
    ModelBenchmarkResult,
    check_benchmark_gates,
    benchmark_model,
    select_best_model,
    update_llm_config,
)


class TestLLMBenchmark(unittest.TestCase):
    """Test benchmarking routines and model selection logic."""

    def test_check_benchmark_gates(self) -> None:
        """Verify acceptance gate logic strictly enforces all 4 criteria."""
        # Passing case
        self.assertTrue(
            check_benchmark_gates(
                grounding_accuracy=0.92,
                warm_ttft_s=0.55,
                peak_vram_mib=2600.0,
                cv_fps=28.0,
            )
        )

        # Fails grounding < 0.90
        self.assertFalse(
            check_benchmark_gates(
                grounding_accuracy=0.88,
                warm_ttft_s=0.55,
                peak_vram_mib=2600.0,
                cv_fps=28.0,
            )
        )

        # Fails TTFT >= 1.0
        self.assertFalse(
            check_benchmark_gates(
                grounding_accuracy=0.95,
                warm_ttft_s=1.20,
                peak_vram_mib=2600.0,
                cv_fps=28.0,
            )
        )

        # Fails VRAM > 5300
        self.assertFalse(
            check_benchmark_gates(
                grounding_accuracy=0.95,
                warm_ttft_s=0.50,
                peak_vram_mib=5400.0,
                cv_fps=28.0,
            )
        )

        # Fails CV FPS < 20
        self.assertFalse(
            check_benchmark_gates(
                grounding_accuracy=0.95,
                warm_ttft_s=0.50,
                peak_vram_mib=2000.0,
                cv_fps=18.0,
            )
        )

    def test_benchmark_model_with_simulated_responder(self) -> None:
        """Verify benchmark_model runs smoothly with a grounded responder."""
        res = benchmark_model(
            model_name="test_model:3b",
            responder=grounded_baseline_responder,
            probe_vram=False,
            warmup_runs=1,
        )

        self.assertIsInstance(res, ModelBenchmarkResult)
        self.assertEqual(res.model_name, "test_model:3b")
        self.assertGreaterEqual(res.grounding_accuracy, 0.80)
        self.assertGreater(res.tokens_per_sec, 0.0)
        self.assertGreaterEqual(res.warm_ttft_s, 0.0)

    def test_select_best_model(self) -> None:
        """Verify model selector prefers gate-passing models with highest grounding score."""
        r1 = ModelBenchmarkResult("model_a:3b", 0.95, 0.45, 45.0, 2500.0, 30.0, True)
        r2 = ModelBenchmarkResult("model_b:3b", 0.91, 0.35, 50.0, 2200.0, 30.0, True)
        r3 = ModelBenchmarkResult("model_c:3b", 0.88, 0.30, 60.0, 2000.0, 30.0, False)

        winner = select_best_model([r1, r2, r3])
        self.assertIsNotNone(winner)
        self.assertEqual(winner.model_name, "model_a:3b")

    def test_update_llm_config(self) -> None:
        """Verify updating YAML configuration with the winning model."""
        tmp_dir = Path("scratch/test_t16")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / "llm.yaml"
        try:
            config_path.write_text(yaml.dump({"llm": {"model": "old_model:3b", "temperature": 0.2}}))
            update_llm_config("new_winner:3b", config_path=config_path)

            updated = yaml.safe_load(config_path.read_text())
            self.assertEqual(updated["llm"]["model"], "new_winner:3b")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.gpu
def test_live_ollama_benchmark_if_running() -> None:
    """Benchmark live Ollama model if daemon is running and model is loaded."""
    from tools.fetch_assets import check_ollama

    status = check_ollama()
    if not status.get("daemon_running"):
        pytest.skip("Ollama daemon is not active (requires escalation H2)")

    res = benchmark_model("qwen2.5:3b")
    assert res.grounding_accuracy >= 0.85
