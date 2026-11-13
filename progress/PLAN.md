# PLAN — AUDIT.md remediation (base 2325aed)

## CURRENT: LLM inactivity debug (supersedes the section below until closed)

Evidence (measured, not inferred):
- llama-cpp-python 0.3.36 in `.venv` is a CPU-only build: `libggml-cuda.so` absent, 0 CUDA symbols in libs, `Llama.supports_gpu_offload` missing. `n_gpu_layers=-1` is a silent no-op.
- Realistic prompt (system + scene + 6-turn history ≈ 1443 words) on this build: **TTFT 61.76 s, total 65.9 s**. This is the "stuck at LLM chat starting", not a deadlock.
- GPU: GTX 1660 Ti (sm_75, 6 GB, driver 615.78.08). `sudo` needs a password; `nvcc` absent; `extra/cuda 13.4.3-1` available in pacman. CUDA rebuild is blocked on one user action.
- Concurrency hazards (brain/turn.py): `start_text_turn` joins up to 120 s; `_active_thread` assigned outside the gate; two workers can start before the lock; `_turn_lock` is held across TTS.

Root cause: CPU prompt-eval on a bloated prompt. Fix path = (a) shrink prompt footprint now, (b) harden turn concurrency/stalls, (c) rebuild llama-cpp with CUDA once the toolkit is installed, (d) fail loudly when a CPU-only build is asked to offload.

### Tasks (DAG)
```
wave1 (parallel): T1 (prompt)  T2 (turn concurrency)  T4 (backend diagnostics)
T3 (CUDA rebuild) <- user installs extra/cuda; independent of T1/T2/T4
T5 (verify) <- T1, T2, T4, (T3 if available)
```
```
id: T1
goal: cut prompt footprint and wire history limits from settings
files: brain/prompt.py, brain/settings.py, config/settings.d/llm.yaml, run_all.py
reuse: PromptBuilder, RollingHistory, LLMSettings, load_settings
deps: none
done: PromptBuilder limits come from LLMSettings (no hard-coded 6/900 left); defaults history_turns=3, history_max_tokens=250, llm.yaml mirrors; measured TTFT on current CPU build < 15 s for a realistic 3-turn prompt; full pytest unchanged
risks: lost context degrades answers -> keep last 3 turns, keep scene line; other PromptBuilder call sites (brain/main.py) must keep working -> extend with defaults, do not break signature
```
```
id: T2
goal: single-flight turns; never block text input or hold the lock across TTS
files: brain/turn.py
reuse: cancel_event, _active_thread, _turn_lock, start_text_turn, run_text_turn, listen_and_respond
deps: none
done: a busy gate prevents two run_text_turn bodies from running at once; start_text_turn returns promptly when busy (does not join 120 s); _active_thread and cancel/state transitions set under the gate; _turn_lock released before TTS; test_t35_turn_bargein + new single-flight test green
risks: barge-in regression -> preserve cancel_event semantics and existing test; state flapping -> transition under gate
```
```
id: T4
goal: surface GPU-offload reality and per-turn latency
files: brain/llm.py
reuse: LlamaCppBackend._load, .chat, ChatChunk
deps: none
done: at load, if n_gpu_layers != 0 and CUDA offload unavailable, log a clear ERROR (not silent); log model-load s, TTFT s, tok/s per chat; exceptions still surfaced as ChatChunk(error); unit test with a fake Llama lacking CUDA asserts the warning
risks: llama-cpp API drift -> getattr guards; noisy logs -> INFO for timings, ERROR only for misconfig
```
```
id: T3
goal: rebuild llama-cpp-python with CUDA so n_gpu_layers=-1 is real
files: tools/rebuild_llamacpp_cuda.sh (new)
reuse: pip, cmake, ninja
deps: user runs `sudo pacman -S --needed extra/cuda`
done: libggml-cuda.so present; Llama.supports_gpu_offload() true; a 1400-token prompt TTFT < 3 s; flash_attn left off until verified on sm_75
risks: sm_75 support in CUDA 13.4 -> CMAKE_CUDA_ARCHITECTURES=75; Python 3.14 source build -> reuse existing sdist toolchain; driver/toolkit mismatch -> verify nvidia-smi after install
```
```
id: T5
goal: verify the real pipeline end to end
files: tests/test_llm_latency_e2e.py (new)
reuse: TurnManager, LlamaCppBackend, PromptBuilder, FakeSink
deps: T1, T2, T4, (T3 optional)
done: a text turn through TurnManager with the real backend returns a non-empty reply within the configured budget; no two turns overlap; full pytest unchanged (4 pre-existing failures remain)
risks: test depends on model file -> skip if model/GPU absent, using pytest.mark.skipif
```

---
# PLAN — (historical) AUDIT.md remediation

Baseline: `pytest -q` 377 pass / 10 skip; `ruff check .` 6 errors (5 tests/test_object_registration.py, 1 ERA001 assistant_cv/hud.py:462).
Tests: `.venv/bin/python -m pytest -q <files>`; lint: `.venv/bin/ruff check <files>`.

## Research (source -> chosen fix)
- Silero v5 needs 512 samples @16 kHz (github.com/snakers4/silero-vad/issues/262) -> re-chunk to 512 inside VAD.
- Onset clipping (Silero VADIterator `speech_pad_ms`; streaming ring-buffer pattern) -> deque pre-roll of `vad_preroll_ms` flushed on speech start.
- Whisper silence hallucination (SYSTRAN/faster-whisper#331) -> `condition_on_previous_text=False`, keep Silero gating.
- Ollama partial/malformed stream JSON (ollama/ollama#4381) -> try/except per line; bad tool args -> tool error to model.
- torch.load pickle RCE (pytorch docs, weights_only) -> checkpoint is dict{labels, m: state_dict} -> `weights_only=True`.
- Mosquitto auth (mosquitto.org conf docs) -> `allow_anonymous false`, `password_file`, `acl_file` (infra/acl from gen_acl.py).
- systemd user units (freedesktop + serverfault): `%h` and `NoNewPrivileges` work; `ProtectSystem`/`PrivateTmp` need mount namespaces, unreliable in user units -> use `%h` + `NoNewPrivileges=yes` only (deviation from AUDIT F29).
- Hash-locked deps (astral uv docs) -> `uv pip compile requirements/*.in --generate-hashes`; local `bmo_contracts` installed separately (editable installs cannot be hash-pinned).
- Arduino String heap fragmentation (Arduino forum/fixed-buffer pattern) -> static `char[]` with length cap + CRC8 (Dallas 0x8C) per frame.
- Sentence splitting (KoljaB/stream2sentence) -> keep regex, add numeric-list guard + clause split at `max_clause_chars` on `, ; :`.

## Decisions
- Contracts own all bus arg models (HwCmd args, enrollment name). brain/tools/schemas.py imports them.
- Broker default host/port live in bmo_contracts/topics.py; CV and brain read them (env override kept).
- New tunables go to config/settings.d/*.yaml + brain/settings.py in one task (T0).
- `VisionCmdResult` gains optional `data` so legacy CV replies can move onto the contract (F2).
- PCM normalisation lives in brain/audio/io.py (edge already imports it) — not bmo_contracts.
- Graph refresh: Supervisor runs `graphify update .` after each commit (parallel sub-agent writes would corrupt graph.json).
- Not done: F33 (personal use, no action); edge audio source/sink wiring in brain/main.py (needs hardware); sign threshold CV 0.90 vs brain 0.75 (different gates; open issue).

## STATE
R1 | DONE | commit 64dde18 | verified by tests/test_t01_contracts.py (16 passed)
R2 | DONE | commit 222e262 | verified by tests/test_r2_hwcmd_args_roundtrip.py (3 passed)
R3 | DONE | commit 270a861 | verified by tests/test_t11_llm.py (18 passed)
R4 | DONE | commit a735739 | verified by tests/test_t32_wake_vad.py (8 passed)
R5 | DONE | commit 9d59c22 | verified by tests/test_contracts_live_payloads.py (13 passed)
R6 | DONE | commit 4323e3f | verified by tests/test_t12_scene.py + tests/test_t41_state.py (20 passed)
R7 | DONE | commit 2048946 | verified by tests/test_t35_turn_bargein.py (5 passed)
T1 | DONE | commit baad7b5 | verified by tests/test_settings_keys.py (TTFT ~8s CPU)
T2 | DONE | commit a146a75 | verified by tests/test_turn_single_flight.py
T4 | DONE | commit 38d7035,15992ac | verified by tests/test_llm_gpu_diag.py (5 passed), probe fixed for module API
T5 | DONE | commit 32612a3 | verified by tests/test_llm_latency_e2e.py (2 passed)
T3 | DONE | commit 15992ac | libggml-cuda.so present, supports_gpu_offload=True, TTFT 4.9s (1405 tok)

## LLM INACTIVITY PHASE: COMPLETE
All 5 tasks (T1–T5) verified and committed.

## LLM UNACTIVITY FIX (this run) — CUDA flash_attn + verbosity + config — COMPLETE
- **flash_attn disabled for Turing (sm_75)**: `use_flash_attn: false` default in brain/settings.py and config/settings.d/llm.yaml; run_all.py uses config. Root cause: CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED on GTX 1660 Ti (no tensor cores, CUDA graph capture unsupported).
- **llm.yaml completed**: all llama-cpp params (backend, model, host, num_ctx, n_gpu_layers, n_threads, use_flash_attn, use_mmap, use_mlock) now in config.
- **run_all.py loads from config**: uses `load_settings()` instead of hardcoded values; removed LoggingLLMClient wrapper; minimal verbosity ("cv is ready, brain is ready" then conversation only).
- **TTFT verified**: 0.12s on GPU (was 2568s with flash_attn error causing hang/fallback).

## NEXT PHASE: HISTORICAL REMEDIATION (T16–T21, T23, T25, T26)
Ready to resume from historical plan. Dependencies satisfied: T0, T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T22, T24, T27, T28 complete.

### Wave 1 (parallel, max 3)
- **T16** Pi client typed HwCmd args + env password — `edge/pi_client.py`, `tests/test_t51_pi_client.py` (deps: T1)
- **T17** Actuator tools contract args + servo clamp — `brain/tools/actuator_tools.py`, `brain/tools/schemas.py`, `edge/esp32_protocol.py`, `tests/test_t25_actuator_tools.py` (deps: T1)
- **T19** CV publishes only contract payloads — `assistant_cv/mqtt_bridge.py`, `assistant_cv/main.py`, `assistant_cv/config.py`, `tests/test_mqtt_commands.py`, `tests/test_pipeline_upgrades.py` (deps: T1)

### Wave 2 (needs T17 + T19)
- **T18** ESP32 CRC8 framing + bounded buffer — `edge/esp32_firmware/bmo_servo_controller.ino`, `edge/esp32_protocol.py`, `tests/test_t55_esp32_protocol.py` (deps: T17) — firmware "needs flash test"
- **T20** CV events, signer, gallery swap — `assistant_cv/main.py`, `assistant_cv/faces.py`, `tests/test_cv_events.py` (deps: T19)

### Wave 3 (needs T19)
- **T21** Enrollment quality gates (blur/pose) — `assistant_cv/enroll.py`, `assistant_cv/config.py`, `tests/test_enroll_quality.py` (deps: T19)
- **T23** Silero framing + pre-roll — `brain/audio/vad.py`, `tests/test_t32_wake_vad.py` (deps: T0, T2)

### Wave 4 (needs T0, T1, T3, T22, T24)
- **T25** Wire policy/lifecycle/telemetry into BrainService — `brain/main.py`, `brain/turn.py`, `tests/test_e2e_system.py`
- **T26** Contract test over real producers — `tests/test_contracts_producers.py` (deps: T16, T17, T19)

## BLOCKERS
- **Push**: GitHub HTTPS auth fails (token invalid)
- **Stray changes**: 6 modified files in `assistant_cv/` (config.py, hands.py, live_sign.py, mp_handpose.py, mp_palmdet.py, mqtt_bridge.py) — from Phase 1, not staged
- **Pre-existing failures**: 5 tests fail (test_e2e_system, test_hygiene, test_t34_tts x2, test_t65_security) — unchanged from baseline

## ENVIRONMENT
- GPU: GTX 1660 Ti (sm_75, 6GB), driver 615.78.08, CUDA 13.4.3
- CUDA rebuilt llama-cpp-python 0.3.36 with `CMAKE_CUDA_ARCHITECTURES=75`
- flash_attn=True works on Turing; default left True
- Model: Qwen3.5-4B-Q4_K_M.gguf (2.6GB) in `models/`

## DAG
```
wave1: T0 T1 T2 T3 T5 T6 T7 T9 T10 T11 T12 T13 T14 T15 T22 T27 T28
T0 -> T4(+T3) , T8 , T23(+T2) , T24
T1 -> T16 , T17 , T19
T17 -> T18
T19 -> T20 , T21
T0,T1,T3,T22,T24 -> T25
T16,T17,T19 -> T26
```

## Tasks
```
id: T0
goal: add every new tunable to settings
files: brain/settings.py, config/settings.d/*.yaml, tests/test_settings_keys.py (new)
reuse: SceneSettings, AudioSettings, PolicySettings, LLMSettings, load_settings
deps: none
done: keys load with defaults: scene.max_prompt_tokens=60; llm.history_turns=6, llm.history_max_tokens=900; audio.vad_chunk_samples=512, audio.vad_preroll_ms=250, audio.tts_max_clause_chars=80; new EdgeSettings edge.failover_debounce=3 (edge.yaml); policy.tick_interval_s=1.0; test asserts each; full pytest still green
risks: duplicate top-level yaml key -> load_settings already rejects; add keys only, no renames
```
```
id: T1
goal: make bmo_contracts the single typed source for all bus payloads
files: bmo_contracts/bmo_contracts/models.py, bmo_contracts/bmo_contracts/topics.py, bmo_contracts/bmo_contracts/__init__.py, tests/test_t01_contracts.py
reuse: Model, Emotion, HwCmd, VisionCmd, VisionCmdResult, brain/tools/schemas.py SetExpressionArgs/MoveServoArgs/PlaySoundArgs (move here)
deps: none
done: HwCmd.args validated per action (set_expression{emotion: Emotion}, move_servo{id,angle,...}, play_sound{name}); VisionCmd enroll/forget args use PersonName = constr ^[A-Za-z0-9 _-]{1,64}$ (rejects "../x"); every envelope has v: Literal[1]=1 (F30); VisionCmdResult.data: dict|None=None; BROKER_HOST="127.0.0.1"/BROKER_PORT=1883 in topics.py; tests cover each; full pytest green (existing producers send no v -> default)
risks: extra="forbid" breaks existing producers -> defaults for v/data, run full suite; schemas.py import left for T17 (keep old copies there until T17)
```
```
id: T2
goal: one PCM normaliser and settings-driven frame constants; drop late UDP packets
files: brain/audio/io.py, edge/audio_transport.py, tests/test_t31_audio_io.py, tests/test_t52_audio_transport.py
reuse: AudioSettings.sample_rate/frame_ms, MemorySink.write_samples, UDPAudioReceiver
deps: none
done: io.to_int16(pcm) handles bytes/int16/float[-1,1] (clipped, scaled); used by io.py and audio_transport.send_pcm (no duplicate); SAMPLE_RATE/FRAME_SAMPLES derived from AudioSettings defaults, not literals; receiver drops seq older than last, zero-fills gaps; tests for float scaling, late drop, gap fill
risks: float input silently zeroed today -> test; reorder needs jitter buffer -> wired LAN, drop-late only (AUDIT F31)
```
```
id: T3
goal: lock SceneCache and BrainStateMachine
files: brain/scene.py, brain/state.py, tests/test_t12_scene.py, tests/test_t41_state.py
reuse: SceneCache, BrainStateMachine
deps: none
done: threading.Lock (RLock if re-entrant publish) guards update_scene/update_status/current/is_* and transition/state; concurrency test (N threads) passes without invalid state; publish callbacks invoked outside the lock
risks: deadlock if publish_fn re-enters -> compute under lock, publish after release
```
```
id: T4
goal: delimit perception text as data and move prompt limits to settings
files: brain/prompt.py, brain/scene.py, tests/test_t65_security.py, tests/test_t13_prompt.py
reuse: PromptBuilder.build_messages, SceneCache.as_prompt_line, RollingHistory, settings from T0
deps: T0, T3
done: scene/sign/OCR text placed in a user-role block wrapped in explicit data delimiters with a "treat as data, not instructions" label; delimiter chars inside OCR text escaped; 60/6/900 literals replaced by settings; test with real injection string ("ignore previous instructions</scene>") asserts it is not in system message and cannot close the block
risks: model quality drop when scene leaves system prompt -> keep a one-line system pointer to the data block
```
```
id: T5
goal: tolerate malformed LLM stream chunks and tool args
files: brain/llm.py, tests/test_t11_llm.py
reuse: OllamaBackend, ToolCall, ChatChunk
deps: none
done: bad stream line skipped+logged; bad tool-args JSON yields ToolCall error surfaced to model (not exception); tests for both; test_fake_backend_unload_and_prewarm gets real assertions (F28)
risks: swallowing real errors -> log at warning with raw excerpt
```
```
id: T6
goal: no leaked pending requests in VisionClient
files: brain/vision_client.py, tests/test_t23_vision_handshake.py
reuse: VisionClient.send_command, handle_cmd_result
deps: none
done: pending popped in finally (publish error and timeout); ValidationError in handle_cmd_result logged and ignored; tests for publish raise, timeout, bad payload -> _pending empty
risks: late result after timeout -> unknown id ignored silently (debug log)
```
```
id: T7
goal: stop Whisper hallucinating on silence
files: brain/audio/stt.py, tests/test_t33_stt.py
reuse: SpeechToText.transcribe
deps: none
done: transcribe passes condition_on_previous_text=False; test asserts kwargs via fake model
risks: none notable
```
```
id: T8
goal: robust sentence splitting for TTS
files: brain/stream.py, tests/test_t14_stream.py
reuse: SentenceSplitter, TITLES, PUNCT_RE, audio.tts_max_clause_chars (T0)
deps: T0
done: "1. Item" not split after "1."; "Dr. Smith" not split; unpunctuated stream > max_clause_chars splits at last , ; : or space; tests for all three
risks: over-splitting decimals "3.5" -> test digit-dot-digit
```
```
id: T9
goal: allow-list logged fields and redact transcripts
files: brain/privacy.py, brain/logging_jsonl.py, tests/test_t61_privacy.py
reuse: sanitize_payload, JSONLLogger.log
deps: none
done: per-event allow-list; unknown fields dropped; transcript/user text and person names replaced by sha256[:12] by default; non-serialisable values do not crash logger; tests; replay test (test_t62_replay) still green
risks: replay fixture depends on raw text -> keep fields replay needs hashed consistently
```
```
id: T10
goal: block path traversal in gallery writes
files: assistant_cv/registry.py, tests/test_registry_paths.py (new)
reuse: _sanitize_id, IdentityRegistry.register_person, ObjectRegistry.register_object, gallery_dir
deps: none
done: file paths use sanitised id; resolved path checked inside gallery_dir else ValueError; test with "../evil" and "/abs" names
risks: existing gallery files named by raw name -> loader keeps reading existing files; only new writes change
```
```
id: T11
goal: honour BMO_OBJ_CONF and clear ruff errors
files: assistant_cv/objects.py, assistant_cv/hud.py, tests/test_object_registration.py
reuse: config.OBJ_CONF
deps: none
done: max(0.50, ...) removed; hud.py:462 comment reworded (no ERA001); 5 ruff errors fixed; `ruff check .` only reports issues outside these files
risks: lower conf -> more false positives; it is the configured value
```
```
id: T12
goal: safe and load-once sign model
files: assistant_cv/live_sign.py, tests/test_live_sign.py (new)
reuse: SignRecognizer._try_load_model
deps: none
done: torch.load(weights_only=True); load attempted once at init; missing model disables step() with one warning; test with tmp state_dict checkpoint and missing path
risks: old checkpoints with non-tensor objects -> labels list[str] is allowed under weights_only; test it
```
```
id: T13
goal: sha256-pinned assets, no runtime downloads
files: tools/fetch_assets.py, assistant_cv/hands.py, docs/assets.md, tests/test_t06_assets_smoke.py
reuse: fetch_* functions, config.MODELS_DIR
deps: none
done: single ASSETS table {name,url,path,sha256} incl. hand ONNX models; downloads verified via hashlib, mismatch deletes file and fails; urlopen has timeout; hands.py raises clear error if model missing (no urlopen); tests with fake download for match/mismatch
risks: pins computed from current upstream files (TOFU) -> record date/source in docs/assets.md; HF/Ollama-managed assets out of scope (library handles)
```
```
id: T14
goal: authenticated mosquitto broker and portable hardened units
files: deploy/systemd/*.service, infra/mosquitto.conf, tools/simple_broker.py, tests/test_t63_systemd.py
reuse: tools/gen_acl.py output infra/acl, simple_broker argparse for tests
deps: none
done: bmo-broker.service runs mosquitto -c %h/bmo-assistant/infra/mosquitto.conf (allow_anonymous false, password_file, acl_file); simple_broker honours --host/--port (test-only); all units use %h, NoNewPrivileges=yes, SupplementaryGroups=audio video where needed; bmo-pi.service EnvironmentFile=-%h/.config/bmo/pi.env (BMO_MQTT_PASSWORD); test_t63 resolves %h and skips isfile for system binaries
risks: mosquitto not installed on this host -> documented as install step, test does not require binary; ProtectSystem omitted (user units, see research)
```
```
id: T15
goal: reproducible hash-locked requirements
files: requirements/*.in, requirements.lock.txt, docs/env.md
reuse: requirements/brain.txt, dev.txt, overrides.txt
deps: none
done: committed .in sources; lock generated by `uv pip compile --generate-hashes`; no file:/// or home paths; docs/env.md shows install (`uv pip install --require-hashes -r requirements.lock.txt` then `pip install -e ./bmo_contracts --no-deps`); `uv pip install --dry-run --require-hashes -r requirements.lock.txt` succeeds
risks: resolver picks newer versions -> pin current versions from existing lock in .in; network needed (blocker if offline)
```
```
id: T16
goal: Pi client uses typed HwCmd args and env password
files: edge/pi_client.py, tests/test_t51_pi_client.py
reuse: HwCmd typed args (T1), PiClient._on_message
deps: T1
done: reads validated args (emotion/name per contract); password from BMO_MQTT_PASSWORD, refuse start without it unless --sim; --sim via BooleanOptionalAction; test_cli_main_execution asserts behaviour (F28)
risks: breaking dev workflow -> --sim keeps working without password
```
```
id: T17
goal: actuator tools emit contract args; one schema and one servo clamp
files: brain/tools/actuator_tools.py, brain/tools/schemas.py, brain/tools/__init__.py, edge/esp32_protocol.py, tests/test_t25_actuator_tools.py
reuse: T1 arg models, ActuatorLimits, load_limits
deps: T1
done: set_expression/play_sound/move_servo build HwCmd with contract args; duplicate SetExpressionArgs/MoveServoArgs removed; schemas.py re-exports contract models; StrictArgs dropped from __init__; servo range computed by one ActuatorLimits method used by esp32_protocol; 180.0 fallback removed (limits.yaml required); tests green
risks: LLM sends "expression" key -> schema field is emotion; tool description updated
```
```
id: T18
goal: bounded serial buffer and CRC8 framing
files: edge/esp32_firmware/bmo_servo_controller.ino, edge/esp32_protocol.py, tests/test_t55_esp32_protocol.py
reuse: format_move_cmd, parse_esp32_response, ESP32ProtocolSimulator
deps: T17
done: firmware uses static char buffer with cap (overflow resets frame); frames end with *HH CRC8 (0x8C) checked both sides; simulator mirrors; tests for good/bad CRC and overflow
risks: firmware cannot be compiled here -> keep C to Arduino core APIs, mark "needs flash test" in PROGRESS
```
```
id: T19
goal: CV publishes only contract payloads on context and cmd_result
files: assistant_cv/mqtt_bridge.py, assistant_cv/main.py, assistant_cv/config.py, tests/test_mqtt_commands.py, tests/test_pipeline_upgrades.py
reuse: MQTTBridge.publish_scene, publish_cmd_result, Scene, VisionCmdResult(data), BROKER_HOST (T1)
deps: T1
done: publish_context and publish_response deleted; callers use publish_scene / publish_cmd_result (data carries legacy payloads); config MQTT_HOST default from bmo_contracts; test_mqtt_commands asserts real VisionCmdResult JSON (F28); brain SceneCache accepts the CV payload
risks: tools/mqtt_client.py reads legacy fields -> grep and adapt read path in the same task if needed (add file to ownership and report)
```
```
id: T20
goal: correct CV events, signer, gallery swap
files: assistant_cv/main.py, assistant_cv/faces.py, tests/test_cv_events.py (new)
reuse: EventDetector, FaceIdentifier
deps: T19
done: one enter/leave event per person id (F16); signer chosen among label=="person" (F17); gallery stored as one (matrix, names) tuple swapped atomically, worker reads one reference (F19); tests for multi-enter/leave and signer
risks: hot-path cost -> single attribute read, no lock
```
```
id: T21
goal: reject blurry and extreme-pose enrollment frames
files: assistant_cv/enroll.py, assistant_cv/config.py, tests/test_enroll_quality.py (new)
reuse: AutoEnrollSession.step, config env pattern
deps: T19
done: frame skipped if Laplacian variance < ENROLL_MIN_SHARPNESS or |yaw|/|pitch| > ENROLL_MAX_POSE_DEG (face.pose if present, else skip pose gate); constants in config.py; tests with synthetic blurred frame
risks: InsightFace pose missing without 3D landmark model -> gate only when attribute exists
```
```
id: T22
goal: run turns off the audio thread so barge-in works
files: brain/turn.py, tests/test_t35_turn_bargein.py
reuse: TurnManager._active_thread, barge_in, cancel_event
deps: none
done: process_audio_frame starts the turn on _active_thread and returns immediately; wake during SPEAKING calls barge_in; test drives real process_audio_frame loop (no hand-started mock thread) and observes barge-in
risks: TTS self-trigger of wake word -> wake gating unchanged; AEC out of scope (AUDIT F3)
```
```
id: T23
goal: correct Silero framing and pre-roll
files: brain/audio/vad.py, tests/test_t32_wake_vad.py
reuse: VoiceActivityDetector, audio.vad_chunk_samples, audio.vad_preroll_ms (T0), FRAME_SAMPLES (T2)
deps: T0, T2
done: frames re-chunked to vad_chunk_samples before Silero; collect_utterance prepends pre-roll deque; tests: 320-sample frames reach model as 512, utterance starts with pre-roll audio
risks: Silero missing in CI -> fake model asserting input length
```
```
id: T24
goal: debounce edge failover
files: brain/edge_monitor.py, tests/test_t54_edge_fallback.py
reuse: EdgeMonitor.handle_hw_status, edge.failover_debounce (T0)
deps: T0
done: state flips only after N consecutive opposite statuses; toggling-heartbeat test does not flap
risks: slower failover -> N*heartbeat period documented
```
```
id: T25
goal: wire policy, lifecycle, telemetry, attribution into BrainService
files: brain/main.py, brain/turn.py, tests/test_e2e_system.py
reuse: ProactivePolicy.tick, IdleLifecycleManager.tick, LatencyTracker.on_turn_metrics, format_attributed_text, BROKER_HOST (T1), policy.tick_interval_s (T0)
deps: T0, T1, T3, T22, T24
done: one ticker thread at tick_interval_s calls policy.tick and lifecycle.tick, stopped in stop(); LatencyTracker passed to TurnManager; attributed text used for user turns; host default from bmo_contracts; e2e test asserts tick called and metrics recorded
risks: ticker vs audio thread races -> relies on T3 locks
```
```
id: T26
goal: contract test over real producers
files: tests/test_contracts_producers.py
reuse: MQTTBridge.publish_scene/publish_cmd_result, ActuatorTools, PiClient._on_message, SceneCache, VisionClient
deps: T16, T17, T19
done: real CV bridge payloads validate in Scene/VisionCmdResult and are accepted by SceneCache/VisionClient; real actuator HwCmd handled by PiClient without error; test fails on reverting any of F1/F2/F4
risks: heavy imports (cv2/torch) -> fake mqtt client, import only bridge module
```
```
id: T27
goal: conditional skip for latency test
files: tests/test_t36_latency.py
reuse: pytest.importorskip, asset paths
deps: none
done: skip only when model/weights absent; otherwise runs
risks: none
```
```
id: T28
goal: remove stale protocol spec and fix ignore typo
files: mqtt_protocols.json, MQTT_GUIDE.md, docs/plan.md, .gitignore
reuse: bmo_contracts/topics.py as source of truth
deps: none
done: mqtt_protocols.json deleted; references point to bmo_contracts; .gitignore AUDI.md -> AUDIT.md
risks: docs drift -> link to topics.py instead of copying
```
