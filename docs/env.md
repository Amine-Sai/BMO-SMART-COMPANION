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
opencv-python; sys_platform == "never"
```

### 1. `onnxruntime; sys_platform == "never"`
- Downstream packages such as `openwakeword`, `faster-whisper`, `insightface`, and standard ONNX utilities list `onnxruntime` as a direct or transitive dependency.
- If unconstrained, package managers (`uv`, `pip`) would install the CPU-only `onnxruntime` distribution, conflicting with or replacing `onnxruntime-gpu==1.31.1`.
- Marking `onnxruntime; sys_platform == "never"` ensures that the solver treats `onnxruntime` as unavailable, allowing `onnxruntime-gpu` to provide the `onnxruntime` module namespace with full TensorRT / CUDA execution provider support.

### 2. `tflite-runtime; sys_platform == "never"`
- Early releases of `openwakeword` depended on `tflite-runtime`. Without overriding `tflite-runtime`, resolvers attempt to downgrade `openwakeword` to `0.4.0` (2022 release), which lacks the modern ONNX backend (`inference_framework="onnx"`).
- Marking `tflite-runtime` as excluded allows `openwakeword==0.6.0` to resolve cleanly with its pure ONNX runtime path.

### 3. `opencv-python; sys_platform == "never"`
- Packages such as `insightface` and `ultralytics` list `opencv-python` as a dependency.
- The environment uses `opencv-contrib-python==5.0.0.93` to provide OpenCV bindings with extra modules.
- Marking `opencv-python; sys_platform == "never"` prevents installing plain `opencv-python`, avoiding namespace clashes and satisfying repository hygiene constraints.

## Requirements Files Structure

- **`requirements/overrides.txt`**: Platform marker overrides to prevent CPU `onnxruntime`, `tflite-runtime`, and plain `opencv-python` installation.
- **`requirements/base.in`**: Base runtime dependencies (PyTorch with CUDA support, OpenCV contrib, ONNX Runtime GPU, vision models, math utilities). Symlinked from `requirements/snapshot_pre_T0.4.txt` for backward compatibility.
- **`requirements/brain.in`**: Speech, audio, LLM client, and protocol dependencies:
  - `faster-whisper==1.2.1`
  - `openwakeword==0.6.0`
  - `silero-vad==6.2.3`
  - `piper-tts==1.8.0`
  - `sounddevice==0.5.6`
  - `ollama==0.6.3`
  - `httpx==0.28.1`
  - `paho-mqtt==2.2.0`
  - `orjson==3.13.0`
  - `pydantic==2.14.3`
  - `pyyaml==6.0.3`
  - `pytest==9.2.0`
  (Symlinked from `requirements/brain.txt` for backward compatibility).
- **`requirements/dev.in`**: Developer hygiene and linting tools:
  - `ruff==0.17.4`
  - `vulture==2.16`
  (Symlinked from `requirements/dev.txt` for backward compatibility).
- **`requirements.lock.txt`**: Full reproducible hash-locked freeze of `.venv` generated via `uv pip compile --override requirements/overrides.txt requirements/*.in --generate-hashes`. Contains cryptographic sha256 hashes for all packages and no local filesystem URIs (`file:///`).

## Installation & Compilation Procedures

### Standard Environment Installation

To install all dependencies reproducibly with hash verification:

```bash
# 1. Install hash-locked dependencies (using overrides to protect GPU packages)
uv pip install --override requirements/overrides.txt --require-hashes -r requirements.lock.txt

# Or equivalently using the environment variable:
# UV_OVERRIDE=requirements/overrides.txt uv pip install --require-hashes -r requirements.lock.txt

# 2. Install local contracts package (editable, without dependencies to maintain hash locking)
pip install -e ./bmo_contracts --no-deps
```

### Re-compiling Requirements

When modifying input dependencies in `requirements/*.in`:

```bash
uv pip compile --override requirements/overrides.txt requirements/*.in --generate-hashes -o requirements.lock.txt
```

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
   Versions for `torch`, `numpy`, `opencv-contrib-python`, and `onnxruntime-gpu` in `requirements.lock.txt` must match `requirements/base.in`.

4. **CV Test Suite Regression**:
   ```bash
   .venv/bin/python -m pytest tests/test_mqtt_commands.py tests/test_pipeline_upgrades.py -q
   ```
   Must execute cleanly with 100% pass rate.
