#!/usr/bin/env python3
"""
BMO Voice Assistant - Full Local Pipeline
Wake -> VAD -> STT -> LLM -> TTS (all offline)
Models: Qwen3.5-4B (LLM), Silero VAD, faster-whisper (STT), Piper (TTS), openWakeWord (wake)
Optimized: GPU layers (-1=all), 8 threads, flash attention, vad_filter for STT
"""

import logging
import warnings
import threading
import time
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from brain.audio.io import AudioSettings, SoundDeviceSource, SoundDeviceSink
from brain.audio.wake import WakeWordDetector
from brain.audio.vad import VoiceActivityDetector
from brain.audio.stt import SpeechToText
from brain.audio.tts import TextToSpeech
from brain.llm import LLMSettings, LLMClient
from brain.scene import SceneCache
from brain.state import BrainStateMachine
from brain.prompt import PromptBuilder
from brain.tools.registry import ToolRegistry
from brain.tools.scene_tools import register_scene_tools
from brain.turn import TurnManager
from brain.settings import SceneSettings


MODEL_PATH = Path("models/Qwen3.5-4B-Q4_K_M.gguf")
PIPER_VOICE_PATH = Path(__file__).parent / "models/piper/bmo.onnx"

# Audio settings tuned for hey_jarvis model
AUDIO_SETTINGS = AudioSettings(
    sample_rate=16000,
    frame_ms=20,
    wake_threshold=0.03,      # Critical: hey_jarvis needs ~0.03
    vad_threshold=0.5,
    vad_min_silence_ms=1000,  # Longer silence wait after speech
    stt_model="base.en",
    stt_device="cpu",
    stt_compute_type="int8",
    tts_voice="en_US-lessac-medium",
    vad_chunk_samples=512,
    vad_preroll_ms=250,
    tts_max_clause_chars=80,
)

# Monkey-patch VAD collect_utterance to use longer timeout
original_collect_utterance = VoiceActivityDetector.collect_utterance

def extended_collect_utterance(self, source, max_duration_s=20.0):
    return original_collect_utterance(self, source, max_duration_s=max_duration_s)

VoiceActivityDetector.collect_utterance = extended_collect_utterance


class LoggingLLMClient:
    """Wrapper that prints LLM responses to stdout."""
    
    def __init__(self, inner: LLMClient):
        self._inner = inner
        self._buffer = []
        self._printing = False
        
    def chat(self, messages, **kwargs):
        self._buffer = []
        self._printing = True
        print("\n>>> BMO thinking...", flush=True)
        for chunk in self._inner.chat(messages, **kwargs):
            if chunk.text:
                self._buffer.append(chunk.text)
                print(chunk.text, end="", flush=True)
            yield chunk
        if self._buffer:
            full = "".join(self._buffer).strip()
            print(f"\n>>> BMO response: {full}\n", flush=True)
        self._printing = False
        
    def unload(self):
        self._inner.unload()
        
    def prewarm(self):
        self._inner.prewarm()


def main():
    print("=" * 60)
    print("  BMO VOICE ASSISTANT - Local Pipeline")
    print("=" * 60)
    print("Wake words: hey jarvis | alexa | hey mycroft | hey rhasspy")
    print("Say wake word, then your question. Ctrl+C to exit.")
    print("=" * 60)

    # Verify Piper voice exists
    if not PIPER_VOICE_PATH.exists():
        print(f"ERROR: Piper voice not found at {PIPER_VOICE_PATH}")
        print("Run: curl -L -o models/piper/en_US-lessac-medium.onnx \\")
        print('  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"')
        return

    # Initialize components
    scene_cache = SceneCache(SceneSettings())
    state_machine = BrainStateMachine(lambda t, p, q, r: None)
    wake = WakeWordDetector(AUDIO_SETTINGS, model_name="hey_jarvis")
    vad = VoiceActivityDetector(AUDIO_SETTINGS)
    stt = SpeechToText(AUDIO_SETTINGS)
    tts = TextToSpeech(AUDIO_SETTINGS, model_path=PIPER_VOICE_PATH)

    llm_settings = LLMSettings(
        backend="llamacpp",
        host=str(MODEL_PATH.absolute()),
        model="Qwen3.5-4B-Q4_K_M",
        num_ctx=4096,
        temperature=0.7,
        n_gpu_layers=-1,      # Use all GPU layers if available
        n_threads=8,          # More threads for faster inference
        use_flash_attn=True,  # Flash attention for speed
        use_mmap=True,
        use_mlock=False,
    )
    llm = LoggingLLMClient(LLMClient(llm_settings))
    import os

    llm.prewarm()

    prompt_builder = PromptBuilder(scene_cache, SceneSettings())
    tools = ToolRegistry()
    register_scene_tools(tools, scene_cache)

    audio_device = os.environ.get("BMO_AUDIO_DEVICE")
    if audio_device is not None:
        try:
            audio_device = int(audio_device)
        except ValueError:
            pass

    audio_sink = SoundDeviceSink(AUDIO_SETTINGS.sample_rate, device=audio_device)

    turn_manager = TurnManager(
        state_machine=state_machine,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=prompt_builder,
        tools=tools,
        tts=tts,
        sink=audio_sink,
        publish_fn=lambda t, p, q, r: None,
    )

    try:
        while True:
            source = SoundDeviceSource(AUDIO_SETTINGS.sample_rate)
            audio_sink = SoundDeviceSink(AUDIO_SETTINGS.sample_rate, device=audio_device)
            
            # Update turn manager with fresh sink
            turn_manager._sink = audio_sink
            
            print("\n[WAITING FOR WAKE WORD...]", flush=True)

            # Phase 1: Wait for wake word
            wake_detected = False
            while not wake_detected:
                frame = source.read_frame()
                if frame is None:
                    continue
                result = turn_manager.process_audio_frame(frame, source=source)
                if result == "listening":
                    wake_detected = True
                    print("[LISTENING - SPEAK YOUR QUESTION!]", flush=True)

            # Phase 2: Background thread handles VAD->STT->LLM->TTS
            print("[PROCESSING YOUR QUESTION...]", flush=True)
            if turn_manager._active_thread:
                turn_manager._active_thread.join(timeout=45.0)

            # Clean up
            audio_sink.close()
            source.close()
            print("[READY FOR NEXT]", flush=True)

    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        llm.unload()
        audio_sink.close()


if __name__ == "__main__":
    main()