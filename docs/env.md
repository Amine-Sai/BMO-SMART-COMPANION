# BMO Assistant Environment Specification

## Overview & Architecture

The BMO Assistant platform runs in a single unified Python virtual environment located at `.venv` running **Python 3.14.8**. All submodules (Computer Vision, Speech Processing, LLM Orchestration, Audio I/O, MQTT Bridge, and Hardware Control) share this virtual environment to eliminate multi-venv IPC overhead, share CUDA GPU memory contexts, and streamline dependency auditing.

## Hardware & Acceleration Setup

- **Python**: 3.14.8 (Linux x86_64)
- **PyTorch**: `torch==2.15.0`, `torchvision==0.29.2`, `triton==3.9.0`
  - CUDA acceleration: Verified via `torch.cuda.is_available() == True`.
- **ONNX Runtime**: `onnxruntime-gpu==1.31.1`
  - Execution Providers: Includes `CUDAExecutionProvider` and `CPUExecutionProvider`.
- **OpenCV**: `opencv-contrib-python==5.0.0.93`
- **NumPy**: `numpy==2.5.4`

## Dependency Override Rationale (`requirements/overrides.txt`)

To maintain CUDA GPU acceleration and avoid resolver deadlocks or package clobbering, `requirements/overrides.txt` specifies:

```text
onnxruntime; sys_platform == "never"
tflite-runtime; sys_platform == "never"
```

### 1. `onnxruntime; sys_platform == "never"`
- Downstream packages such as `openwakeword`, `faster-whisper`, and standard ONNX utilities list `onnxruntime` as a direct or transitive dependency.
- If unconstrained, package managers (`uv`, `pip`) would install the CPU-only `onnxruntime` distribution, conflicting with or replacing `onnxruntime-gpu==1.31.1`.
- Marking `onnxruntime; sys_platform == "never"` ensures that the solver treats `onnxruntime` as unavailable, allowing `onnxruntime-gpu` to provide the `onnxruntime` module namespace with full TensorRT / CUDA execution provider support.

### 2. `tflite-runtime; sys_platform == "never"`
- Early releases of `openwakeword` depended on `tflite-runtime`. Without overriding `tflite-runtime`, resolvers attempt to downgrade `openwakeword` to `0.4.0` (2022 release), which lacks the modern ONNX backend (`inference_framework="onnx"`).
- Marking `tflite-runtime` as excluded allows `openwakeword>=0.6.0` to resolve cleanly with its pure ONNX runtime path.

## Requirements Files Structure

- **`requirements/overrides.txt`**: Platform marker overrides to prevent CPU `onnxruntime` and `tflite-runtime` installation.
- **`requirements/brain.txt`**: Speech, audio, LLM client, and protocol dependencies:
  - `faster-whisper`
  - `openwakeword>=0.6.0`
  - `silero-vad`
  - `piper-tts`
  - `sounddevice`
  - `ollama`
  - `httpx`
  - `paho-mqtt`
  - `orjson`
  - `pydantic`
  - `pyyaml`
  - `pytest`
- **`requirements/dev.txt`**: Developer hygiene and linting tools:
  - `ruff`
  - `vulture`
- **`requirements/snapshot_pre_T0.4.txt`**: Baseline environment snapshot captured prior to installing speech/LLM packages.
- **`requirements.lock.txt`**: Full locked freeze of `.venv` post-installation.

## Core Pinned Package Versions

| Package | Version | Purpose |
|---|---|---|
| `torch` | `2.15.0` | PyTorch deep learning runtime (CUDA enabled) |
| `torchvision` | `0.29.2` | Computer vision vision models |
| `onnxruntime-gpu` | `1.31.1` | ONNX runtime with CUDA execution provider |
| `opencv-contrib-python` | `5.0.0.93` | Computer vision and image processing |
| `numpy` | `2.5.4` | Numerical array computations |
| `faster-whisper` | `1.2.1` | ASR transcription backend |
| `openwakeword` | `0.6.0` | Wake word detection (ONNX runtime) |
| `silero-vad` | `6.2.3` | Voice activity detection |
| `piper-tts` | `1.8.0` | Text-to-speech synthesis |
| `sounddevice` | `0.5.6` | PortAudio audio I/O streaming |
| `ollama` | `0.6.3` | Ollama LLM client library |
| `httpx` | `0.28.1` | HTTP async client |
| `paho-mqtt` | `2.2.0` | MQTT messaging protocol client |
| `orjson` | `3.13.0` | High-speed JSON serialization |
| `pydantic` | `2.14.3` | Data validation and contract models |
| `pyyaml` | `6.0.3` | Configuration parser |
| `pytest` | `9.2.0` | Test runner framework |
| `ruff` | `0.17.4` | Code formatting and linting |
| `vulture` | `2.16` | Dead code analysis |

## Verification Procedures

1. **Imports & GPU Acceleration**:
   ```bash
   .venv/bin/python -c "import faster_whisper, openwakeword, silero_vad, piper, sounddevice, ollama, httpx, paho.mqtt.client, orjson, pydantic, yaml, pytest, onnxruntime, torch; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(); assert torch.cuda.is_available(); print('ALL_IMPORTS_OK')"
   ```

2. **Package Isolation Check**:
   ```bash
   uv pip list --python .venv/bin/python | grep -E "onnxruntime|openwakeword|tflite"
   ```
   Must contain `onnxruntime-gpu` and `openwakeword` (0.6.0) and must not contain plain `onnxruntime` or `tflite-runtime`.

3. **Core Version Parity**:
   Versions for `torch`, `numpy`, `opencv-contrib-python`, and `onnxruntime-gpu` in `requirements.lock.txt` must match `requirements/snapshot_pre_T0.4.txt`.

4. **CV Test Suite Regression**:
   ```bash
   .venv/bin/python -m pytest tests/test_mqtt_commands.py tests/test_pipeline_upgrades.py -q
   ```
   Must execute cleanly with 100% pass rate.
