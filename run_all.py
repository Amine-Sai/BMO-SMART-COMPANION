#!/usr/bin/env python3
"""
BMO Full System - Coordinated Startup
Runs: MQTT Broker -> CV Pipeline -> Brain Voice Assistant
CV writes scene to data/cv_scene.json; brain reads it via file polling.
CLI is used only for text input to the LLM.
"""

import logging
import warnings
import threading
import time
import signal
import sys
import json
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

from bmo_contracts.models import Scene

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Component processes
_broker_proc = None
_cv_thread = None
_brain_thread = None
_text_thread = None
_shutdown = threading.Event()

# Intermediate file for CV -> brain scene communication
SCENE_FILE = Path("data/cv_scene.json")


def start_mqtt_broker():
    """Verify mosquitto is reachable on 127.0.0.1:1883.

    CV and brain no longer use MQTT for scene communication (file-based
    instead), but the broker is still used by CV's MQTT bridge. Returns None.
    """
    global _broker_proc
    _broker_proc = None
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(2.0)
        s.connect(("127.0.0.1", 1883))
        logger.info("MQTT broker (mosquitto) reachable on 127.0.0.1:1883")
    except OSError as exc:
        logger.warning("MQTT broker not reachable on 127.0.0.1:1883: %s", exc)
    finally:
        s.close()
    return None


def stop_mqtt_broker():
    """No-op: mosquitto is managed externally."""
    pass


# ---------------------------------------------------------------------------
# File-based scene reader: CV -> brain scene cache
# ---------------------------------------------------------------------------
_scene_file_mtime: float | None = None


def read_scene_file() -> dict | None:
    """Read the latest scene contract written by the CV pipeline.

    Returns None if the file doesn't exist or can't be parsed.
    """
    try:
        if not SCENE_FILE.is_file():
            return None
        data = json.loads(SCENE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def start_scene_file_reader(scene_cache) -> None:
    """Poll the intermediate scene file and update scene_cache on change.

    Runs in a daemon thread.
    """
    global _scene_file_mtime

    def poll():
        global _scene_file_mtime
        while not _shutdown.is_set():
            try:
                if SCENE_FILE.is_file():
                    mtime = SCENE_FILE.stat().st_mtime
                    if _scene_file_mtime is None or mtime > _scene_file_mtime:
                        data = read_scene_file()
                        if data is not None:
                            scene = Scene.model_validate(data)
                            scene_cache.update_scene(scene)
                            people = [(p.name, p.track_id) for p in (scene.people or [])]
                            logger.info("Scene update from CV file: people=%s", people)
                            _scene_file_mtime = mtime
            except Exception:
                pass
            time.sleep(0.5)

    t = threading.Thread(target=poll, daemon=True, name="Scene-File-Reader")
    t.start()


def stop_scene_file_reader() -> None:
    """No-op: reader thread stops when _shutdown is set."""
    pass


def run_cv_pipeline():
    """Run the CV pipeline in a thread with output redirected to log file."""
    import os
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # Create log directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    cv_log_path = log_dir / "cv_pipeline.log"
    
    logger.info("Starting CV pipeline (output -> %s)", cv_log_path)
    
    # Redirect stdout/stderr BEFORE importing CV module
    with open(cv_log_path, "a") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            try:
                from assistant_cv.main import run_pipeline
                
                # Suppress console logging for CV modules
                cv_logger = logging.getLogger("assistant_cv")
                cv_logger.setLevel(logging.WARNING)
                cv_logger.propagate = False
                
                # Also suppress ultralytics, cv2, etc.
                for name in ["ultralytics", "cv2", "torch", "PIL"]:
                    l = logging.getLogger(name)
                    l.setLevel(logging.WARNING)
                    l.propagate = False
                
                run_pipeline(
                    cam_index=0,
                    headless=True,
                    enable_mqtt=True,
                    mqtt_host="127.0.0.1",
                    mqtt_port=1883,
                    lazy=False,
                )
            except Exception as e:
                logger.error("CV pipeline error: %s", e)
            finally:
                logger.info("CV pipeline stopped")


def run_brain_voice(scene_cache: SceneCache) -> None:
    """Run the brain voice assistant in a thread."""
    logger.info("Starting Brain voice assistant...")
    try:
        # Import here to avoid early initialization
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
        from pathlib import Path

        MODEL_PATH = Path("models/Qwen3.5-4B-Q4_K_M.gguf")
        PIPER_VOICE_PATH = Path(__file__).parent / "models/piper/en_US-lessac-medium.onnx"

        AUDIO_SETTINGS = AudioSettings(
            sample_rate=16000,
            frame_ms=20,
            wake_threshold=0.03,
            vad_threshold=0.5,
            vad_min_silence_ms=1000,
            stt_model="base.en",
            stt_device="cpu",
            stt_compute_type="int8",
            tts_voice="en_US-lessac-medium",
            vad_chunk_samples=512,
            vad_preroll_ms=250,
            tts_max_clause_chars=80,
        )

        # Monkey-patch VAD for longer timeout
        original_collect = VoiceActivityDetector.collect_utterance
        def extended_collect(self, source, max_duration_s=20.0):
            return original_collect(self, source, max_duration_s=max_duration_s)
        VoiceActivityDetector.collect_utterance = extended_collect

        class LoggingLLMClient:
            def __init__(self, inner):
                self._inner = inner
                self._buffer = []
            def chat(self, messages, **kwargs):
                self._buffer = []
                print("\n>>> BMO thinking...", flush=True)
                for chunk in self._inner.chat(messages, **kwargs):
                    if chunk.text:
                        self._buffer.append(chunk.text)
                        print(chunk.text, end="", flush=True)
                    yield chunk
                if self._buffer:
                    full = "".join(self._buffer).strip()
                    print(f"\n>>> BMO response: {full}\n", flush=True)
            def unload(self): self._inner.unload()
            def prewarm(self): self._inner.prewarm()

        print("=" * 60)
        print("  BMO VOICE ASSISTANT - Connected to CV via MQTT")
        print("=" * 60)
        print("Wake words: hey jarvis | alexa | hey mycroft | hey rhasspy")
        print("CV events will appear in scene context automatically")
        print("=" * 60)

        if not PIPER_VOICE_PATH.exists():
            logger.error("Piper voice not found at %s", PIPER_VOICE_PATH)
            return

        state_machine = BrainStateMachine(lambda t, p, q, r: None)
        wake = WakeWordDetector(AUDIO_SETTINGS, model_name="hey_jarvis")
        vad = VoiceActivityDetector(AUDIO_SETTINGS)
        stt = SpeechToText(AUDIO_SETTINGS)
        tts = TextToSpeech(AUDIO_SETTINGS, model_path=PIPER_VOICE_PATH)

        # Tune llama-cpp for responsive local inference on CPU
        # n_gpu_layers=0 (CPU only), flash_attn disabled (CPU build doesn't support it)
        llm_settings = LLMSettings(
            backend="llamacpp",
            host=str(MODEL_PATH.absolute()),
            model="Qwen3.5-4B-Q4_K_M",
            num_ctx=2048,
            temperature=0.7,
            n_gpu_layers=0,
            n_threads=4,
            use_flash_attn=False,
            use_mmap=True,
            use_mlock=False,
        )
        llm = LoggingLLMClient(LLMClient(llm_settings))
        print(">>> Loading Qwen model into memory (first load may take 30-60s on CPU)...", flush=True)
        llm.prewarm()
        print(">>> Qwen model ready.", flush=True)

        prompt_builder = PromptBuilder(scene_cache, SceneSettings())
        tools = ToolRegistry()
        register_scene_tools(tools, scene_cache)

        audio_sink = SoundDeviceSink(AUDIO_SETTINGS.sample_rate)

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

        # Create persistent audio source to avoid ALSA stream conflicts
        source = SoundDeviceSource(AUDIO_SETTINGS.sample_rate)

        # Start text input thread (typed messages share the same LLM->TTS pipeline)
        global _text_thread
        _text_thread = threading.Thread(
            target=run_text_input, args=(turn_manager,), daemon=True, name="Text-Input"
        )
        _text_thread.start()

        while not _shutdown.is_set():
            print("\n[WAITING FOR WAKE WORD...]", flush=True)

            wake_detected = False
            while not wake_detected and not _shutdown.is_set():
                frame = source.read_frame()
                if frame is None:
                    continue
                result = turn_manager.process_audio_frame(frame, source=source)
                if result == "listening":
                    wake_detected = True
                    print("[LISTENING - SPEAK YOUR QUESTION!]", flush=True)

            print("[PROCESSING YOUR QUESTION...]", flush=True)
            if turn_manager._active_thread:
                turn_manager._active_thread.join(timeout=60.0)

            print("[READY FOR NEXT]", flush=True)

        # Ensure any running turn completes before closing audio
        if turn_manager._active_thread and turn_manager._active_thread.is_alive():
            print("[WAITING FOR TURN TO COMPLETE...]", flush=True)
            turn_manager._active_thread.join(timeout=30.0)

        audio_sink.close()
        source.close()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Brain voice error: %s", e, exc_info=True)
    finally:
        logger.info("Brain voice assistant stopped")


def run_text_input(turn_manager: "TurnManager") -> None:
    """Read typed lines from stdin and feed them into the same turn pipeline.

    Reuses TurnManager.start_text_turn -> run_text_turn, so typed text goes
    through LLM -> TTS -> speaker exactly like voice input. Runs until
    _shutdown is set or the user types 'quit'/'exit'.
    """
    logger.info("Text input thread started. Type messages and press Enter.")
    print("\n[TYPING MODE] Type a message and press Enter to talk to BMO. "
          "Type 'quit' or 'exit' to stop. Wake-word voice input still works.\n", flush=True)

    while not _shutdown.is_set():
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        text = line.strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            logger.info("Text input requested shutdown")
            _shutdown.set()
            break

        # Feed typed text into the same pipeline as voice input.
        # start_text_turn handles joining any existing _active_thread.
        print(f">>> Sending to BMO: {text}", flush=True)
        try:
            turn_manager.start_text_turn(text)
        except Exception as exc:
            logger.error("Text input turn failed: %s", exc)

    logger.info("Text input thread stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Shutdown signal received...")
    _shutdown.set()


def main():
    """Main entry point - starts all components in coordination."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("  BMO FULL SYSTEM STARTUP")
    logger.info("=" * 60)

    # 1. Verify MQTT broker (mosquitto) is reachable
    start_mqtt_broker()

    # Shared scene cache: CV writes via intermediate file, brain reads for LLM context
    from brain.settings import SceneSettings
    from brain.scene import SceneCache
    scene_cache = SceneCache(SceneSettings())

    # Start file-based scene reader (CV file -> brain scene cache)
    start_scene_file_reader(scene_cache)

    # 2. Start CV Pipeline (in background thread)
    global _cv_thread
    _cv_thread = threading.Thread(target=run_cv_pipeline, daemon=True, name="CV-Pipeline")
    _cv_thread.start()
    logger.info("CV pipeline thread started")

    # Give CV time to connect to broker and start writing scenes
    time.sleep(3)

    # 3. Start Brain Voice Assistant (main thread)
    global _brain_thread
    _brain_thread = threading.Thread(
        target=run_brain_voice, args=(scene_cache,), daemon=True, name="Brain-Voice"
    )
    _brain_thread.start()

    try:
        # Keep main thread alive
        while not _shutdown.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        _shutdown.set()
        stop_scene_file_reader()
        stop_mqtt_broker()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()