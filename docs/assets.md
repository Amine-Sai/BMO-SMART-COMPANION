# BMO Assistant: Model Assets and Engine Specifications

## Overview
This document specifies the required pretrained neural network weights and runtimes for the BMO Assistant brain subsystem.

| Component | Engine | Target Model | Format | Device | Size | Location / Cache |
|---|---|---|---|---|---|---|
| **LLM** | Ollama | `qwen2.5:3b` | GGUF (4-bit/8-bit) | GPU (CUDA) | ~1.9 GB | Ollama model store (`~/.ollama/models`) |
| **STT** | Faster-Whisper | `base.en` | CTranslate2 / int8 | CPU | ~140 MB | Hugging Face cache (`~/.cache/huggingface/hub`) |
| **TTS** | Piper | `en_US-lessac-medium` | ONNX + JSON | CPU | ~63 MB | `models/piper/` |
| **Wake** | openWakeWord | `hey_jarvis_v0.1` | ONNX | CPU | ~15 MB | `site-packages/openwakeword/resources/models/` |

---

## 1. Asset Fetch Tool
Model assets can be verified and downloaded using `tools/fetch_assets.py`:

```bash
# Verify presence of assets without downloading
.venv/bin/python tools/fetch_assets.py --check-only

# Download missing assets
.venv/bin/python tools/fetch_assets.py
```

---

## 2. Runtimes and Human Escalations

### H2: Ollama LLM Service
Ollama runs as an external inference daemon.
- **Arch / Omarchy**: `sudo pacman -S ollama-cuda && sudo systemctl enable --now ollama`
- **User-space fallback**: Run `~/.local/bin/ollama serve`
- **Default model**:
  ```bash
  ollama pull qwen2.5:3b
  ```

### Faster-Whisper
`faster-whisper` utilizes CTranslate2 with quantized `int8` weights on CPU to keep GTX 1660 Ti VRAM completely free for computer vision YOLO and LLM inference.
- Weights are automatically cached to `~/.cache/huggingface/hub/models--Systran--faster-whisper-base.en`.

### Piper TTS
`piper-tts` operates on CPU via ONNX Runtime.
- Model file: `models/piper/en_US-lessac-medium.onnx`
- Config file: `models/piper/en_US-lessac-medium.onnx.json`
- Source: `rhasspy/piper-voices` v1.0.0.

### openWakeWord
`openwakeword` operates via ONNX Runtime on CPU, analyzing 16-bit 16 kHz audio frames:
- Core model: `hey_jarvis_v0.1.onnx`
- Preprocessors: `melspectrogram.onnx`, `embedding_model.onnx`
- Models are located within the package resources.
