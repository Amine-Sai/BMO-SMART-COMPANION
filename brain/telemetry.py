from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import threading
import time
from typing import Any

import numpy as np

from brain.logging_jsonl import JSONLLogger


@dataclass
class TurnMetrics:
    vad_end_ms: float = 0.0
    stt_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    total_ms: float = 0.0
    utt_id: str = ""
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LatencyTracker:
    """Tracks latency metrics across voice interaction turns and calculates percentiles."""

    def __init__(
        self,
        logger: JSONLLogger | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._logger = logger
        self._clock = clock
        self._records: list[TurnMetrics] = []
        self._lock = threading.Lock()

    def record_turn(
        self,
        metrics: dict[str, float] | TurnMetrics,
        utt_id: str = "",
    ) -> TurnMetrics:
        if isinstance(metrics, TurnMetrics):
            item = metrics
            if utt_id and not item.utt_id:
                item.utt_id = utt_id
            if item.ts == 0.0:
                item.ts = self._clock()
        else:
            vad_end_ms = float(metrics.get("vad_end_ms", 0.0))
            stt_ms = float(metrics.get("stt_ms", 0.0))
            llm_ttft_ms = float(metrics.get("llm_ttft_ms", 0.0))
            tts_first_chunk_ms = float(metrics.get("tts_first_chunk_ms", 0.0))
            total_ms = float(
                metrics.get(
                    "total_ms",
                    metrics.get("e2e_voice_ms", metrics.get("turn_ms", 0.0)),
                )
            )
            item_utt_id = utt_id or str(metrics.get("utt_id", ""))
            ts = float(metrics.get("ts", self._clock()))
            item = TurnMetrics(
                vad_end_ms=vad_end_ms,
                stt_ms=stt_ms,
                llm_ttft_ms=llm_ttft_ms,
                tts_first_chunk_ms=tts_first_chunk_ms,
                total_ms=total_ms,
                utt_id=item_utt_id,
                ts=ts,
            )

        with self._lock:
            self._records.append(item)

        if self._logger is not None:
            self._logger.log("turn_latency", **item.to_dict())

        return item

    def on_turn_metrics(self, metrics: dict[str, float]) -> TurnMetrics:
        return self.record_turn(metrics)

    @property
    def records(self) -> list[TurnMetrics]:
        with self._lock:
            return list(self._records)

    def _metric_values(self, metric: str) -> list[float]:
        with self._lock:
            return [float(getattr(r, metric, 0.0)) for r in self._records]

    def p50(self, metric: str = "total_ms") -> float:
        vals = self._metric_values(metric)
        if not vals:
            return 0.0
        return float(np.percentile(vals, 50))

    def p95(self, metric: str = "total_ms") -> float:
        vals = self._metric_values(metric)
        if not vals:
            return 0.0
        return float(np.percentile(vals, 95))

    def summary(self) -> dict[str, dict[str, float]]:
        metric_names = (
            "vad_end_ms",
            "stt_ms",
            "llm_ttft_ms",
            "tts_first_chunk_ms",
            "total_ms",
        )
        res: dict[str, dict[str, float]] = {}
        for name in metric_names:
            vals = self._metric_values(name)
            if not vals:
                res[name] = {
                    "p50": 0.0,
                    "p95": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0.0,
                }
            else:
                res[name] = {
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                    "count": float(len(vals)),
                }
        return res
