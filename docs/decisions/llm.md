# Architecture Decision Record: LLM Selection (T1.6)

## Status: ACCEPTED

## Date: 2026-11-09

## Context
BMO requires an on-device local LLM operating on an NVIDIA GTX 1660 Ti (6 GB total VRAM). When `assistant_cv` is running (YOLO26s, BoT-SORT, ArcFace, OpenCV GRU sign tracker), baseline VRAM consumption is ~716 MiB. The brain service requires an LLM with:
1. Native tool/function-calling support for scene query, identity, and physical actuators.
2. Grounding accuracy ≥ 90% against scene context to avoid hallucinations.
3. Sub-second warm time-to-first-token (TTFT < 1.0 s) for low conversational latency.
4. Total peak GPU memory usage `peak_used_MiB <= 5300` MiB to prevent CUDA out-of-memory errors alongside computer vision.

---

## Candidates Evaluated

Three 3B-class instruction-tuned models available through Ollama were benchmarked:

| Metric | Target Gate | `qwen2.5:3b` | `llama3.2:3b` | `granite3.1-dense:2b` |
|---|---|---|---|---|
| **Parameters** | 2–4B | 3.09B | 3.21B | 2.49B |
| **Quantization** | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M |
| **Grounding Accuracy** | ≥ 90.0% | **96.9%** (62/64) | 89.1% (57/64) | 84.4% (54/64) |
| **Warm TTFT** | < 1.00 s | **0.42 s** | 0.58 s | 0.39 s |
| **Tokens / sec** | ≥ 20 tok/s | **44.8 tok/s** | 41.2 tok/s | 52.1 tok/s |
| **Model VRAM** | — | ~1,850 MiB | ~1,920 MiB | ~1,540 MiB |
| **Peak Total VRAM (with CV)** | ≤ 5,300 MiB | **2,570 MiB** | 2,640 MiB | 2,260 MiB |
| **CV Loop FPS** | ≥ 20 FPS | **48–52 FPS** | 48–52 FPS | 48–52 FPS |
| **Tool Calling Reliability** | High | **100% Valid JSON** | 92% Valid JSON | 86% Valid JSON |

---

## Decision
Select **`qwen2.5:3b`** as the default LLM for BMO Assistant.

### Rationale:
1. **Grounding Superiority**: Achieved 96.9% claim accuracy on `evals/grounding.jsonl`, strictly adhering to scene entities and correctly denying absent objects without hallucinating.
2. **Function Calling Robustness**: Produced strictly valid JSON tool call syntax matching Ollama OpenAI-compatible schemas across all scene, identity, and actuator tool invocations.
3. **Headroom and Safety**: Peak total VRAM of 2,570 MiB leaves > 3.2 GB of GPU memory headroom, well within the 5,300 MiB safety envelope.
4. **Sub-second Response**: Warm TTFT of 0.42 s easily satisfies the conversational responsiveness budget.

---

## Configuration
Recorded in `config/settings.d/llm.yaml`:
```yaml
llm:
  model: "qwen2.5:3b"
  temperature: 0.2
  think_mode: false
```
