# BMO Assistant: Model Assets and Engine Specifications

## Overview
This document specifies the required pretrained neural network weights and runtimes for the BMO Assistant brain and vision subsystems.

| Component | Engine | Target Model | Format | Device | Size | Location / Cache |
|---|---|---|---|---|---|---|
| **LLM** | Ollama | `qwen2.5:3b` | GGUF (4-bit/8-bit) | GPU (CUDA) | ~1.9 GB | Ollama model store (`~/.ollama/models`) |
| **STT** | Faster-Whisper | `base.en` | CTranslate2 / int8 | CPU | ~140 MB | Hugging Face cache (`~/.cache/huggingface/hub`) |
| **TTS** | Piper | `en_US-lessac-medium` | ONNX + JSON | CPU | ~63 MB | `models/piper/` |
| **Wake** | openWakeWord | `hey_jarvis_v0.1` | ONNX | CPU | ~15 MB | `openwakeword/resources/models/` |

---

## 1. Pinned Assets Registry (`ASSETS`)

All standalone model files are pinned by SHA256 checksums in `tools/fetch_assets.py` (`ASSETS` table schema `{name, url, path, sha256}`).
Pins were established via Trust On First Use (TOFU) verification against upstream releases on 2026-11-10.

| Name | Target Path | SHA256 (Pinned 2026-11-10) | Upstream Source |
|---|---|---|---|
| `piper_voice` | `models/piper/en_US-lessac-medium.onnx` | `5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f` | [rhasspy/piper-voices (v1.0.0)](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx) |
| `piper_config` | `models/piper/en_US-lessac-medium.onnx.json` | `efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0` | [rhasspy/piper-voices (v1.0.0)](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json) |
| `hey_jarvis_v0.1` | `<openwakeword>/resources/models/hey_jarvis_v0.1.onnx` | `94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb` | [dscripka/openWakeWord (v0.5.1)](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx) |
| `melspectrogram` | `<openwakeword>/resources/models/melspectrogram.onnx` | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` | [dscripka/openWakeWord (v0.5.1)](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx) |
| `embedding_model` | `<openwakeword>/resources/models/embedding_model.onnx` | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` | [dscripka/openWakeWord (v0.5.1)](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx) |

### Integrity and Verification Policy
- **Hash verification**: Every downloaded file is hashed using Python's `hashlib.sha256`.
- **Fail-fast and delete on mismatch**: If the SHA256 does not match the pinned hash, the file is unlinked immediately and a `ValueError` is raised. Partial or corrupt files are never retained.
- **Timeout enforcement**: Remote asset downloads enforce socket timeouts (default 30 seconds) via `urllib.request`.
- **Zero runtime downloads**: All model assets are fetched ahead of time via `tools/fetch_assets.py`; runtime code never downloads weights.

---

## 2. Asset Fetch Tool

Model assets can be verified and downloaded using `tools/fetch_assets.py`:

```bash
# Verify presence and SHA256 integrity of assets without downloading
.venv/bin/python tools/fetch_assets.py --check-only

# Download missing assets and verify SHA256 pins
.venv/bin/python tools/fetch_assets.py
```

---

## 3. External Library-Managed Assets

Assets managed by external libraries or daemon services are intentionally kept library-managed and are not managed via direct file downloads in `tools/fetch_assets.py`:

### Ollama LLM Service (`qwen2.5:3b`)
Ollama runs as an external inference daemon and manages its own model store (`~/.ollama/models`).
- **Arch / Omarchy**: `sudo pacman -S ollama-cuda && sudo systemctl enable --now ollama`
- **User-space fallback**: Run `~/.local/bin/ollama serve`
- **Model pull**:
  ```bash
  ollama pull qwen2.5:3b
  ```

### Faster-Whisper (`base.en`)
`faster-whisper` utilizes CTranslate2 with quantized `int8` weights on CPU to keep GTX 1660 Ti VRAM completely free for computer vision and LLM inference.
- Weights are managed by Hugging Face Hub cache (`~/.cache/huggingface/hub/models--Systran--faster-whisper-base.en`).
- `fetch_faster_whisper()` verifies or triggers download through `faster_whisper.download_model("base.en")`.
