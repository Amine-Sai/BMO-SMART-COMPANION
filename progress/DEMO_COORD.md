# BMO Demo Coordination (single source of truth)

This file is gitignored. It is the only coordination file between sub-agents.
All progress, plans, and next steps live here. Do not create other coordination files.

## Current state (ACTIVE PHASE: 1)

### Phase 1 tasks R1-R7
All DONE (committed + verified):
- R1 | commit 64dde18 | assistant_cv/mqtt_bridge.py + brain/main.py | test_t01_contracts.py (16 pass)
- R2 | commit 222e262 | bmo_contracts/models.py + actuator_tools.py + pi_client.py | test_r2_hwcmd_args_roundtrip.py (3 pass)
- R3 | commit 270a861 | brain/llm.py | test_t11_llm.py (18 pass)
- R4 | commit a735739 | brain/audio/vad.py | test_t32_wake_vad.py (8 pass)
- R5 | commit 9d59c22 | tests/test_contracts_live_payloads.py | 13 pass, fails on pre-fix commit 4c5b9ee
- R6 | commit 4323e3f | brain/scene.py + brain/state.py | test_t12_scene.py + test_t41_state.py (20 pass)
- R7 | commit 2048946 | brain/turn.py + brain/main.py | test_t35_turn_bargein.py (5 pass)

### Infrastructure (NOW AVAILABLE)
- mosquitto broker: RUNNING on 127.0.0.1:1883 ✓
- qwen model: present at models/Qwen3.5-4B-Q4_K_M.gguf (2.7GB) ✓
- llama-cpp-python: 0.3.36 installed ✓ (GPU settings)
- sounddevice, faster-whisper, piper: all installed ✓
- Camera: /dev/video0, /dev/video1 ✓
- Audio: pipewire 128 in/128 out, default device ✓

### run_all.py — MQTT + TEXT INPUT + GPU + QUIET CLI
1. **CV→brain via MQTT**: CV publishes Scene to VISION_CONTEXT; brain subscribes via `start_scene_subscriber()`. Most recent scene only (scene_cache.current() always returns latest).
2. **Text input mode**: `run_text_input()` reads typed lines at `>` prompt, feeds into `turn_manager.start_text_turn()` → same LLM→TTS→speaker pipeline as voice. Type 'quit'/'exit' to stop.
3. **CV logs clogging FIXED** (commit 55b850e):
   - CV pipeline stdout/stderr redirected to `logs/cv_pipeline.log`
   - Scene subscriber "Scene update from CV" and "Vision status update from CV" downgraded to DEBUG — no longer clog CLI
   - CLI only shows: startup logs, Qwen load messages, "BMO IS READY" banner, `>` prompt, LLM response
4. **llama-cpp GPU settings** (commit 07f08dd): `n_gpu_layers=-1`, `use_flash_attn=True`, `n_threads=4`, `num_ctx=2048`. Qwen on GPU for fast inference.
5. **BMO readiness banner** (commit 55b850e): "BMO IS READY" banner prints after model load, before text input prompt — so the `>` prompt is never misleading.
6. **Turn serialization fix** (commit 69f73cc): Added `self._turn_lock` to `TurnManager` to serialize all llama-cpp inference. The segfault happened because text input and voice wake word triggered concurrent LLM chats on separate threads — llama-cpp `create_chat_completion()` is not thread-safe. `start_text_turn` now waits up to 120s for the previous turn to finish before starting a new one.

### Data flow
CV pipeline → Scene model → MQTT VISION_CONTEXT → mosquitto → scene subscriber → scene_cache → PromptBuilder → system prompt → Qwen (llama-cpp GPU) → TTS (piper) → speaker

### Input modes (both available)
- **Voice**: Wake word ("hey jarvis") → VAD → STT → Qwen → TTS → speaker
- **Text**: Type at `>` prompt → Qwen → TTS → speaker

### CLI usage
- CV output → logs/cv_pipeline.log (NOT CLI)
- Scene/MQTT debug → suppressed (DEBUG level)
- CLI shows: startup logs, "Loading Qwen model...", "Qwen model ready on GPU.", "BMO IS READY" banner, `>` prompt, `">>> Sending to BMO: ..."`, streaming LLM response

### LLM responsiveness on GPU
Qwen 4B Q4_K_M on GPU is fast. Expect:
- Model load: 10-20s (shown with feedback)
- First token: ~1-3s depending on prompt length
- Full response: seconds for typical replies
- num_ctx=2048 for reasonable context window

### Test suite (baseline)
466 passed / 10 skipped / 4 failed (pre-existing, unrelated to Phase 1)

### Acceptance gate
tools/check_acceptance.py: 40/40 PASS

## Next steps
1. Run `./run_all.py` — wait for "BMO IS READY" banner (~10-20s for GPU model load)
2. Type at `>` prompt or say "hey jarvis"
3. Get fast LLM responses
4. Verify CV recognizes objects/people and Qwen can talk about them
5. Report results

## Open / blockers
- Haven't run ./run_all.py end-to-end yet (needs camera + mic + speaker + model load)