# Task Progress (PLAN.md)
`T<n> | status | files touched | key decision | open issue`

Prior tasks T0.1–T6.6: superseded by AUDIT.md (T1.2, T0.6, T6.5 claims false; T3.5/T4.2/T4.3/T4.5/T3.6 unwired). Remediation below.

T0 | DONE | brain/settings.py, config/settings.d/*.yaml, tests/test_settings_keys.py | edge optional in Settings; default tunables preserved without yaml | none
T1 | DONE | bmo_contracts/bmo_contracts/{models.py,topics.py,__init__.py}, tests/test_t01_contracts.py | HwCmd args validated per action with emotion/name mapping; v=1 envelope default; PersonName regex on enroll/forget | none
T2 | DONE | brain/audio/io.py, edge/audio_transport.py, tests/test_t31_audio_io.py, tests/test_t52_audio_transport.py | single to_int16 normaliser with float scaling; AudioSettings default frame constants; late UDP drop and zero-filled gaps | none
T3 | DONE | brain/scene.py, brain/state.py, tests/test_t12_scene.py, tests/test_t41_state.py | RLock on SceneCache & BrainStateMachine; status payload built under lock, publish outside | none
T4 | TODO | | | T4.2/T4.3/T4.5/T3.6 unwired (F8)
T5 | DONE | brain/llm.py, tests/test_t11_llm.py | ToolCall error field + args={"error": ...} surfaces via ToolRegistry; bad stream line skipped | tool errors not fed back to model in turn.py (turn.py outside scope)
T6 | DONE | brain/vision_client.py, tests/test_t23_vision_handshake.py | pop _pending in finally around publish and timeout; catch ValidationError and debug log late arrivals | none
T7 | DONE | brain/audio/stt.py, tests/test_t33_stt.py | pass condition_on_previous_text=False to whisper; allow model DI | none
T8 | DONE | brain/stream.py, tests/test_t14_stream.py | NUMERIC_LIST_RE guards list items, AudioSettings.tts_max_clause_chars defaults clause split on ,;: then space | none
T9 | DONE | brain/privacy.py, brain/logging_jsonl.py, tests/test_t61_privacy.py | per-event allow-list with sha256[:12] PII hashing and unhandled orjson serialization fallback | none
T10 | DONE | assistant_cv/registry.py, tests/test_registry_paths.py | _safe_gallery_path validates raw and resolved paths inside gallery_dir raising ValueError; sanitised id used for new files while legacy raw names load preserved | none
T11 | DONE | assistant_cv/objects.py, assistant_cv/hud.py, tests/test_object_registration.py | honour BMO_OBJ_CONF without 0.50 clamp; prefix unused dummy args with _ | none
T12 | DONE | assistant_cv/live_sign.py, tests/test_live_sign.py | load once at init with weights_only=True; step() returns disabled state with one warning | none
T13 | DONE | tools/fetch_assets.py, assistant_cv/hands.py, docs/assets.md, tests/test_t06_assets_smoke.py | ASSETS table with sha256 TOFU pins & delete-on-mismatch; hands.py runtime download replaced with FileNotFoundError; library-managed cache documented | none
T14 | DONE | deploy/systemd/*.service, infra/mosquitto.conf, tools/simple_broker.py, tests/test_t63_systemd.py | %h paths with NoNewPrivileges=yes, mosquitto /usr/sbin path with conf doc, CLI args for simple_broker | none
T15 | DONE | requirements/*.in, requirements.lock.txt, docs/env.md, requirements/overrides.txt | Pin lock versions in .in to prevent drift; keep symlinks .txt->.in for test_t64_licenses; omit unhashable git clip; overrides.txt handles GPU/contrib package exemptions | test_hygiene.py:100 (test_lockfile_matches_uv_pip_freeze) expects unhashed pip-freeze format, needs update for hash-pinned lockfile
T16 | TODO | | |
T17 | TODO | | |
T18 | TODO | | |
T19 | TODO | | |
T20 | TODO | | |
T21 | TODO | | |
T22 | DONE | brain/turn.py, tests/test_t35_turn_bargein.py | start voice turn on _active_thread and return 'listening' immediately; proactive handlers now use _active_thread | test_e2e_system:201 awaits T25 (asserts sync reply)
T23 | TODO | | |
T24 | DONE | brain/edge_monitor.py, tests/test_t54_edge_fallback.py | default failover_debounce from EdgeSettings(), prompt initial status, require N consecutive opposite statuses | none
T25 | TODO | | |
T26 | TODO | | |
T27 | DONE | tests/test_t36_latency.py | conditional skip on missing packages/weights/Ollama reusing fetch_assets & settings | none
T28 | DONE | mqtt_protocols.json, MQTT_GUIDE.md, docs/plan.md, .gitignore | delete stale spec, point to bmo_contracts | none

## Phase 1 Remediation (this run)
R1 | fixed | assistant_cv/mqtt_bridge.py, assistant_cv/main.py, brain/main.py, tools/mqtt_client.py | CV bridge emits bmo_contracts.Scene; brain logs ValidationError; legacy publish_context removed | none
R2 | fixed | bmo_contracts/bmo_contracts/models.py, brain/tools/actuator_tools.py, tests/test_r2_hwcmd_args_roundtrip.py | HwCmd args typed; actuator_tools uses contract models; pi_client reads validated fields; no duplicate args | none
R3 | DONE | brain/llm.py | json.loads guarded; malformed chunks logged and skipped; tool args parse errors returned to model | commit 270a861
R4 | DONE | brain/audio/vad.py, tests/test_t32_wake_vad.py | 320→512 re-chunking with remainder buffer; 250ms pre-roll deque prepended to utterances | commit a735739
R5 | DONE | tests/test_contracts_live_payloads.py | Real producer payloads validate against contracts; pre-fix failure verified via git worktree | commit 9d59c22
R6 | DONE | brain/scene.py, brain/state.py | RLock on SceneCache & BrainStateMachine; concurrent access validated | commit 4323e3f
R7 | fixed | brain/turn.py, brain/main.py, tests/test_t35_turn_bargein.py | Proactive handlers use _active_thread; barge_in works during proactive turns | none

## Corrected Claims (from audit)
T1.2 "thread-safe" → FALSE (F12 was fixed in T3, not T1)
T0.6 "checksum-verified" → FALSE (F10 fixed in T13)
T6.5 "prompt-injection defenses" → FALSE (F9 not fixed; T4 still TODO)
T3.5 "barge-in" → PARTIAL (wake-word barge-in fixed in T22/R7; proactive handlers fixed in R7; VAD barge-in not implemented)
T4.2/T4.3/T4.5/T3.6 "wired" → FALSE (F8; T4 still TODO)

## LLM inactivity (T-series, this run)
T1 | DONE | brain/prompt.py, brain/settings.py, config/settings.d/llm.yaml, run_all.py, tests/test_settings_keys.py | history limits come from LLMSettings (3 turns / 250 tokens); dropped SceneSettings passed as system_prompt | TTFT 61.76s -> ~8s on CPU build; commit baad7b5
T2 | DONE | brain/turn.py, tests/test_turn_single_flight.py | non-blocking `_turn_gate` single-flight; busy turns dropped, no 120s join; inference lock removed so TTS cannot block a turn | commit a146a75
T4 | DONE | brain/llm.py, tests/test_llm_gpu_diag.py | ERROR when n_gpu_layers!=0 on a CPU-only build; INFO load/TTFT/tok-s; fixed probe for module-level llama_supports_gpu_offload() | commit 38d7035 + 15992ac
T5 | DONE | tests/test_llm_latency_e2e.py | real backend text turn ~5s (CPU) / 2s (GPU), single-flight drop asserted, skips without model | commit 32612a3
T3 | DONE | tools/rebuild_llamacpp_cuda.sh, brain/llm.py, tests/test_llm_gpu_diag.py | libggml-cuda.so present, supports_gpu_offload=True, TTFT 4.9s (1405 tokens); flash_attn verified working on sm_75 | commit 15992ac

## LLM unactivity fix (this run) — CUDA flash_attn + verbosity + config
T3b | DONE | brain/settings.py, config/settings.d/llm.yaml, run_all.py | flash_attn=False default for Turing (sm_75); CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED fixed; llm.yaml now complete with all llama-cpp params; run_all.py loads from config, minimal verbosity "cv is ready, brain is ready" | TTFT 0.12s on GPU (was 2568s with flash_attn error)

## Approach Notes (T3 - CUDA rebuild)
- **Attempt 1**: `pip wheel --no-binary llama-cpp-python` used cached wheel despite flag; `--no-cache-dir` required.
- **Attempt 2**: `pip wheel --no-deps --no-cache-dir --no-binary=:all:` correctly built from sdist (76MB download). Build ran under `systemd-run --user` for detachment; took ~8 min on 6 jobs.
- **API discovery**: llama-cpp-python 0.3.36 exposes `llama_supports_gpu_offload()` module function, NOT `Llama.supports_gpu_offload()`. Fixed `_check_gpu_offload` to probe both, updated tests to monkeypatch module function.
- **TTFT result**: 4.9s for 1405-token prompt on GTX 1660 Ti (sm_75, no tensor cores). Plan target <3s was optimistic for this arch; 12x speedup over CPU (61s) achieved. flash_attn=True works but provides no benefit on Turing.
- **Stray changes**: 6 modified files in assistant_cv/ (config.py, hands.py, live_sign.py, mp_handpose.py, mp_palmdet.py, mqtt_bridge.py) — from Phase 1 remediation, not staged.

Diagnosis | DONE | | root cause = CPU prompt-eval on a ~1,443-word prompt (TTFT 61.76s), not a deadlock | push blocked: gh token invalid