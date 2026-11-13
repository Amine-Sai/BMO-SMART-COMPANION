# AUDIT — bmo-assistant (commit 2325aed, 2026-11-09)

## 1. Verdict
- Units are well tested (377 pass / 10 skip / 0 fail) but the system did not integrate: the CV→brain and brain→Pi payloads failed the shared contracts, so live messages were dropped or raised errors.
- **Phase 1 remediation complete (2026-11-10):** F1, F2, F3, F4, F7, F11, F12, F13 fixed. System now integrates minimally end-to-end.
- PROGRESS.md marks every task DONE. Three claims are false: T1.2 "thread-safe", T0.6 "checksum-verified" and T6.5 "prompt-injection defenses". T3.5 barge-in and the T4.2/T4.3/T4.5/T3.6 features are not wired into `brain/main.py`.
- The "40/40 acceptance" check verifies that docs exist, not that anything works. Integration tests use `tools/mock_cv.py`, not the real bridge, which is why none of this was caught.

## 2. Findings

```
id: F1
severity: critical
where: assistant_cv, assistant_cv/mqtt_bridge.py:335 -> brain/scene.py:114, brain/main.py:215
weakness: CV context payload is a legacy dict; brain Scene model rejects it (extra keys; missing ts/people/objects/recent) and drops it without logging
evidence: contract probe (in-memory validation) CONFIRMED
improvement: build the payload with bmo_contracts.Scene in mqtt_bridge; log ValidationError in brain
alternative: extra="ignore" on consumer | does not fix missing required fields; hides drift
effort: M
status: fixed — publish_scene() emits Scene model; brain logs ValidationError; test_t01_contracts validates round-trip
```
```
id: F2
severity: critical
where: assistant_cv, assistant_cv/mqtt_bridge.py:325 -> brain/vision_client.py:29
weakness: cmd_result uses request_id/command/status/data; brain expects VisionCmdResult id/ok/detail, so every vision tool call times out
evidence: contract probe CONFIRMED
improvement: emit VisionCmdResult from bmo_contracts in publish_response
alternative: adapter in brain | duplicates the contract; rejected
effort: M
status: fixed — publish_cmd_result() already emits VisionCmdResult; test_t01_contracts validates round-trip
```
```
id: F3
severity: critical
where: brain-runtime, brain/turn.py:55,68,110; brain/main.py:262
weakness: run_text_turn runs synchronously in the audio loop, so wake-word barge-in cannot fire while speaking; _active_thread is never started
evidence: probe Q6 CONFIRMED; tests/test_t35_turn_bargein.py:249-251 starts a mock thread outside the real loop
improvement: run the turn on the existing _active_thread so the audio loop keeps feeding wake/VAD
alternative: WebRTC AEC3 (needed later for VAD barge-in without self-trigger) | new dependency, 10 ms framing; not a substitute for the threading fix
effort: M
status: fixed — _active_thread started for wake-word turns; proactive handlers now use _active_thread; barge_in joins and cancels active turn; test_t35_turn_bargein validates
```
```
id: F4
severity: critical
where: cross-module, brain/tools/actuator_tools.py:179,207 vs edge/pi_client.py:97,101
weakness: producer sends {"expression"}/{"sound"} on HW_CMD; Pi reads args["emotion"]/args["name"], so it raises KeyError
evidence: contract probe CONFIRMED
improvement: define typed args in bmo_contracts and use them on both sides
alternative: rename keys on one side | quickest fix, but drift can happen again
effort: S
status: fixed — actuator_tools uses SetExpressionArgs/PlaySoundArgs from bmo_contracts; pi_client reads validated model fields; test_r2_hwcmd_args_roundtrip validates
```
```
id: F5
severity: high
where: tools, deploy/systemd/bmo-broker.service:10; tools/simple_broker.py:241-248
weakness: prod unit runs the fallback simple_broker with no auth/ACL by default; --host/--port CLI args are ignored
evidence: probe Q8 CONFIRMED
improvement: run mosquitto with the ACL that tools/gen_acl.py already generates; keep simple_broker for tests only
alternative: add auth to simple_broker | re-implements mosquitto; rejected
effort: S
status: verified
```
```
id: F6
severity: high
where: assistant_cv, assistant_cv/registry.py:132,428; bmo_contracts/models.py:151
weakness: enrollment name (from MQTT/LLM tool) is only .strip()'d before building a file path, so path traversal is possible
evidence: CV probe Q1 CONFIRMED; contract args is an unvalidated dict
improvement: constrain the name with a regex in the contract model and resolve/compare the path under gallery_dir
alternative: hash names to filenames | safe, but filenames become opaque
effort: S
status: verified
```
```
id: F7
severity: high
where: brain-runtime, brain/audio/io.py:14; brain/audio/vad.py:45-46,68-76
weakness: Silero VAD gets 320-sample frames (v5 needs 512 @16 kHz); no pre-speech ring buffer, so the first phonemes are clipped
evidence: probe Q4 CONFIRMED; https://github.com/snakers4/silero-vad/issues/262
improvement: re-chunk to 512 samples in the VAD; add a ~200-300 ms pre-roll buffer
alternative: TEN VAD / WebRTC VAD | new dependency, or more false triggers; keep Silero
effort: S
status: fixed — VoiceActivityDetector re-chunks 320→512 with remainder buffer; 250ms pre-roll deque prepended; test_t32_wake_vad validates
```
```
id: F8
severity: high
where: brain-runtime, brain/main.py
weakness: ProactivePolicy.tick, IdleLifecycleManager.tick, attribution and telemetry are never called or instantiated in prod (T4.2/T4.3/T4.5/T3.6 dead)
evidence: probe Q7 CONFIRMED (tick only in tests/test_t42_policy.py:110, test_t45_lifecycle.py:62)
improvement: wire them into BrainService (one ticker on the existing loop); otherwise delete them and mark the tasks undone
alternative: n/a
effort: M
status: verified
```
```
id: F9
severity: high
where: brain-core, brain/prompt.py:64; brain/scene.py:81,195
weakness: raw sign/OCR text goes into the system prompt with no delimiting; the T6.5 test only checks that the substring and roles are present
evidence: probe Q2 CONFIRMED (tests/test_t65_security.py:96,134,137)
improvement: move perception text into a delimited, data-labelled user-role block; add a test with a real injection string
alternative: guided JSON output (vLLM xgrammar) | stops malformed tool calls, not injection
effort: S
status: verified
```
```
id: F10
severity: high
where: tools/assistant_cv, tools/fetch_assets.py:65,106,136; assistant_cv/hands.py:38-40
weakness: no asset is sha256-verified (hashlib not imported) despite the T0.6 claim; hands.py downloads at runtime with urlopen, no checksum and no timeout
evidence: probe Q3 + CV probe Q7 CONFIRMED
improvement: add a pinned sha256 table to fetch_assets; remove the runtime download from hands.py (fail fast if missing)
alternative: Hugging Face hub revision pinning | adds a dependency; a sha table is enough
effort: S
status: verified
```
```
id: F11
severity: high
where: integrity, tools/check_acceptance.py:38-51; tests/test_t26_integration_a.py:39,87
weakness: acceptance only regex-matches "status: pass" in docs; the integration test uses mock_cv instead of the real MQTTBridge, so F1-F4 pass CI
evidence: integrity report; contract probe Q4 CONFIRMED
improvement: add one contract test that feeds real assistant_cv/brain publisher payloads into bmo_contracts models
alternative: full broker e2e | needs mosquitto in CI; a contract test is cheaper
effort: M
status: fixed — test_contracts_live_payloads.py covers CV context, cmd_result, HW_CMD; fails on pre-fix commit; passes now
```
```
id: F12
severity: high
where: brain-core/runtime, brain/scene.py:98-127; brain/state.py:61-90
weakness: SceneCache and BrainStateMachine are changed from the MQTT thread and the audio thread with no lock
evidence: probe Q1/Q11 CONFIRMED (main.py:224,242,253; turn.py:66,120,125; edge_monitor.py:74)
improvement: add one threading.Lock to each
alternative: python-statemachine (run-to-completion queue) | new dependency for 5 states; rejected
effort: S
status: fixed — RLock on SceneCache & BrainStateMachine in T3; concurrent access validated
```
```
id: F13
severity: high
where: brain-core, brain/llm.py:49,94
weakness: unguarded json.loads on stream lines and tool args; a malformed or partial chunk aborts the turn
evidence: probe Q9 CONFIRMED; https://github.com/ollama/ollama/issues/4381
improvement: parse tool args on the done chunk inside try/except and return a tool error to the model
alternative: OpenAI SDK delta accumulator / guided decoding | extra dependency or a vLLM-only path
effort: S
status: fixed — json.loads guarded with try/except; malformed chunks logged and skipped; tool args parse errors returned to model
```
```
id: F14
severity: medium
where: brain-core, brain/vision_client.py:29,54-58
weakness: pending entry leaks if publish raises; ValidationError in handle_cmd_result is not caught; no timeout cleanup
evidence: probe Q10 CONFIRMED
improvement: try/finally around publish; catch ValidationError; pop the entry on timeout
alternative: MQTT v5 correlation-data | breaks simple_broker; no gain with one requester
effort: S
status: verified
```
```
id: F15
severity: medium
where: brain-runtime, brain/audio/stt.py:71
weakness: transcribe() only passes beam_size/language; Whisper hallucinates on silence or noise
evidence: probe Q5 CONFIRMED; https://github.com/SYSTRAN/faster-whisper/issues/331
improvement: condition_on_previous_text=False (+ vad_filter=True or keep Silero gating)
alternative: Moonshine | lower latency, but a new model and English only
effort: S
status: verified
```
```
id: F16
severity: medium
where: assistant_cv, assistant_cv/main.py:445-452,472-475
weakness: when several people enter/leave in one frame, only the first person's event is emitted
evidence: CV probe Q4 CONFIRMED
improvement: emit one event per person id in the diff set
alternative: n/a
effort: S
status: verified
```
```
id: F17
severity: medium
where: assistant_cv, assistant_cv/main.py:1036
weakness: signer = synced_objects[0] with no label=="person" check, so signs can be attributed to objects
evidence: CV probe Q5 CONFIRMED
improvement: filter to persons before picking the signer
alternative: use brain/attribution.py | also unwired (F8)
effort: S
status: verified
```
```
id: F18
severity: medium
where: assistant_cv, assistant_cv/live_sign.py:71,118-119
weakness: torch.load(weights_only=False) allows pickle code execution from the weights file; a missing model is re-tried every frame (30 Hz disk I/O)
evidence: CV probe Q3 CONFIRMED
improvement: weights_only=True (state_dict); try the load once at init and disable the worker if missing
alternative: export to ONNX | consistent with other models, but needs a conversion step
effort: S
status: verified
```
```
id: F19
severity: medium
where: assistant_cv, assistant_cv/main.py:231,747; assistant_cv/faces.py:80-84
weakness: gallery matrix swap is atomic, but the multi-field update (matrix + names) races with the face worker
evidence: CV probe Q2 PARTIAL (atomic reference, multi-field race)
improvement: swap a single (matrix, names) tuple reference
alternative: lock | works, but adds contention on the hot path
effort: S
status: verified
```
```
id: F20
severity: medium
where: integrity, requirements.lock.txt:1
weakness: lock has no hashes, contains a local -e file:/// URI, and has no .in sources, so builds are not reproducible
evidence: integrity report (file inspection)
improvement: pip-compile/uv lock --generate-hashes from a committed requirements.in
alternative: poetry | heavier tool switch
effort: S
status: verified
```
```
id: F21
severity: medium
where: edge, edge/pi_client.py:41,147-155
weakness: hard-coded MQTT password "pi_insecure_dev_password" with no env/CLI override
evidence: probe Q12 CONFIRMED
improvement: read it from an env var (systemd EnvironmentFile); refuse to start without it outside --sim
alternative: n/a
effort: S
status: verified
```
```
id: F22
severity: medium
where: brain-runtime, brain/privacy.py:9
weakness: sanitize_payload only masks bytes and arrays; transcripts and names go to JSONL unredacted
evidence: runtime inspector + architecture researcher (two independent reads)
improvement: allow-list the logged fields per event type; hash or drop transcript text by default
alternative: Presidio pattern-only | ~200 MB RAM and a dependency; allow-list is enough
effort: M
status: verified
```
```
id: F23
severity: medium
where: brain-core, brain/stream.py:61,78
weakness: regex splitter breaks on "1." lists and abbreviations; no max-length fallback, so long clauses stall TTS
evidence: core inspector; https://github.com/KoljaB/stream2sentence
improvement: add a list/abbreviation guard plus a clause split after ~80 chars on , ; :
alternative: stream2sentence | better coverage, but a new dependency
effort: S
status: UNVERIFIED
```
```
id: F24
severity: medium
where: brain-runtime, brain/edge_monitor.py:80
weakness: edge failover has no hysteresis or debounce, so it can flap; edge source/sink are not configured in main
evidence: runtime inspector only
improvement: require N consecutive misses or recoveries before switching
alternative: n/a
effort: S
status: UNVERIFIED
```
```
id: F25
severity: medium
where: edge, edge/esp32_firmware/bmo_servo_controller.ino:125
weakness: unbounded Arduino String concatenation on serial input (heap exhaustion); no frame checksum
evidence: edge inspector only
improvement: fixed char buffer with length cap; append a CRC8 to the frame
alternative: COBS framing | robust, but slightly more code
effort: M
status: UNVERIFIED
```
```
id: F26
severity: low
where: brain-core, brain/tools/actuator_tools.py:25 vs brain/tools/schemas.py
weakness: SetExpressionArgs/MoveServoArgs are defined twice and the copies differ
evidence: core inspector only
improvement: delete the actuator_tools copies; import from schemas (or bmo_contracts per F4)
alternative: n/a
effort: S
status: UNVERIFIED
```
```
id: F27
severity: low
where: assistant_cv, assistant_cv/objects.py:174; assistant_cv/config.py:28
weakness: max(0.50, conf) silently overrides a lower BMO_OBJ_CONF
evidence: CV probe Q6 CONFIRMED
improvement: use the config value directly
alternative: n/a
effort: S
status: verified
```
```
id: F28
severity: low
where: integrity, tests/test_mqtt_commands.py:89; tests/test_t11_llm.py:336; tests/test_t51_pi_client.py:205; tests/test_t36_latency.py:235
weakness: one test mocks the unit under test, two tests have no assertions, one skips unconditionally; 9 assistant_cv files are untested
evidence: integrity report (pytest run, file inspection)
improvement: assert on real outputs; make the skip conditional; cover faces/live_sign with fixtures
alternative: n/a
effort: S
status: verified
```
```
id: F29
severity: low
where: deploy, deploy/systemd/*.service
weakness: hard-coded /home/judas paths; Type=simple with no watchdog or sandboxing
evidence: edge inspector; architecture researcher (freedesktop systemd docs)
improvement: use %h paths; add NoNewPrivileges, ProtectSystem=strict, PrivateTmp and SupplementaryGroups=audio video
alternative: Docker Compose | painful passthrough for audio, camera and GPIO; rejected
effort: S
status: verified
```
```
id: F30
severity: low
where: contracts, bmo_contracts/bmo_contracts/models.py:25
weakness: only events carry v=1; status and command messages have no version field
evidence: edge inspector; architecture researcher
improvement: add v to every envelope
alternative: protobuf | compile step; not needed in a single-language monorepo
effort: S
status: UNVERIFIED
```
```
id: F31
severity: low
where: edge, edge/audio_transport.py:128
weakness: UDP receiver queues packets out of order; no reorder or loss concealment
evidence: edge inspector; https://www.opus-codec.org/comparison/
improvement: drop late sequence numbers and zero-fill gaps (wired LAN is enough for now)
alternative: Opus/RTP with FEC/PLC | 8x less bandwidth, but adds a codec dependency on both ends
effort: M
status: UNVERIFIED
```
```
id: F32
severity: low
where: assistant_cv, assistant_cv/enroll.py
weakness: enrollment gates only on single face + frame similarity; blurry or extreme-pose frames corrupt the stored identity
evidence: vision researcher (NIST FRVT quality report)
improvement: reject frames with low Laplacian variance or large yaw/pitch
alternative: CR-FIQ quality model | heavier
effort: M
status: UNVERIFIED
```
```
id: F33
severity: low
where: assistant_cv/edge, docs/licenses.md
weakness: InsightFace weights are non-commercial and Ultralytics is AGPL-3.0; this blocks any distribution
evidence: https://github.com/deepinsight/insightface ; https://github.com/ultralytics/ultralytics
improvement: none needed for personal use; if distributing, switch to YuNet+SFace (Apache) and YOLOX/RT-DETR (Apache)
alternative: YuNet+SFace | slightly lower accuracy (LFW 99.40 vs 99.77)
effort: L
status: verified
```

## 3. KEEP
- openWakeWord / faster-whisper / Piper: local, CPU-viable, no keys; the problems are configuration (F7, F15), not the choice of library.
- Silero VAD: best accuracy/latency of the options compared; fix the frame size only (F7).
- Linear NumPy cosine scan for the face gallery: ~15 µs at N<500; FAISS/hnswlib would add a dependency for no gain.
- OpenCV-DNN MediaPipe ONNX + GRU sign model: avoids the MediaPipe Tasks glibc crashes; fast enough.
- Ultralytics built-in tracker: no evidence of tracking failures.
- MQTT + mosquitto + generated ACL; correlation id in the payload: one requester, and MQTT v5 would break simple_broker.
- Hand-rolled 5-state machine: add a lock (F12); a library is not justified.
- Custom httpx Ollama client: fix parsing (F13); the OpenAI SDK has no second use.
- Strict Pydantic contracts in a shared package: extra="forbid" is fine for monorepo lockstep deploys; the defect is that producers bypass them (F1, F2, F4).
- systemd user units over Docker: device passthrough is simpler; only hardening is missing (F29).

## 4. Cross-module issues
- Contract drift: F1, F2, F4 (CV→brain context and cmd_result, brain→Pi HW_CMD) — **FIXED in Phase 1**.
- mqtt_protocols.json:48,456 is stale vs bmo_contracts/topics.py:33 and models.py:80 (missing bmo/vision/status, bmo/brain/*, bmo/hw/*; legacy envelope). Generate it from bmo_contracts or delete it.
- Duplicated logic: PCM bytes/ndarray normalization in brain/audio/io.py:86 and edge/audio_transport.py:76; servo clamp in brain/tools/actuator_tools.py:73 and edge/esp32_protocol.py:41. Move both into bmo_contracts (already shared by both sides).
- Scattered config: broker host "localhost" (assistant_cv/config.py:73) vs "127.0.0.1" (brain/main.py:59); sign threshold 0.90 (assistant_cv/config.py:31) vs 0.75 (config/settings.d/policy.yaml:6); token/history limits hard-coded at brain/scene.py:198; 16 kHz/320 samples hard-coded at brain/audio/io.py:12,221.
- Servo speed 180.0 fallback (actuator_tools.py:67) vs 90.0 (limits.yaml:30): REFUTED as a runtime bug because limits.yaml always loads; it is still a duplicate constant.
- Unused export: StrictArgs in brain/tools/__init__.py:10.
- Dead code: brain/turn.py:55 _active_thread (reuse it for F3); assistant_cv/hud.py:462 commented-out code (ERA001); 5 ruff F401/ARG002/F841 in tests/test_object_registration.py.

## 5. Unverified items
- F23 sentence splitter: needs a test run of SentenceSplitter on "1. Item" / "Dr. Smith" / an 80+ char unpunctuated stream.
- F24 edge_monitor flapping: needs edge_monitor.py:80 logic confirmed plus a toggling-heartbeat test.
- F25 ESP32 String overflow: needs a .ino:125 read confirming there is no length cap.
- F26 duplicate args models: needs a field-level diff of actuator_tools.py:25 vs schemas.py.
- F30 schema versioning: needs a list of envelope classes that have or lack `v`.
- F31 UDP reorder: needs audio_transport.py:128 confirmed plus a loss measurement on the target LAN.
- F32 enrollment gating: needs enroll.py gate conditions listed.
- Blocker: three sub-agent runs failed on API connection loss and were resumed. No coverage numbers because pytest-cov is not installed; broker/model e2e tests were skipped (mosquitto and ollama absent).
