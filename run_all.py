#!/usr/bin/env python3
"""
BMO Full System - Coordinated Startup
Runs: MQTT Broker -> CV Pipeline -> Brain Voice Assistant
CV publishes Scene via MQTT; brain subscribes and takes the most recent frame.
CLI is used only for text input to the LLM.
"""

import logging
import warnings
import threading
import time
import signal
import sys
import socket
from pathlib import Path

import paho.mqtt.client as mqtt

from bmo_contracts.models import Scene, VisionStatus
from bmo_contracts.topics import VISION_CONTEXT, VISION_STATUS

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sounddevice")

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
_scene_sub_client = None
_shutdown = threading.Event()


def start_mqtt_broker():
    """Verify mosquitto is reachable on 127.0.0.1:1883.

    Returns None (no subprocess to manage; mosquitto is started externally).
    """
    global _broker_proc
    _broker_proc = None
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
# MQTT scene subscriber: CV -> brain scene cache (most recent frame only)
# ---------------------------------------------------------------------------
def start_scene_subscriber(scene_cache) -> None:
    """Connect to mosquitto, subscribe to VISION_CONTEXT + VISION_STATUS,
    and update scene_cache with only the most recent scene.
    """
    global _scene_sub_client

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(VISION_CONTEXT, qos=0)
            client.subscribe(VISION_STATUS, qos=1)
            logger.info("Subscribed to MQTT topics %s, %s", VISION_CONTEXT, VISION_STATUS)
        else:
            logger.warning("MQTT scene subscriber connect failed: rc=%s", rc)

    def on_message(client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            if msg.topic == VISION_CONTEXT:
                scene = Scene.model_validate_json(msg.payload)
                scene_cache.update_scene(scene)
                people = [(p.name, p.track_id) for p in (scene.people or [])]
                logger.debug("Scene update from CV: people=%s", people)
            elif msg.topic == VISION_STATUS:
                scene_cache.update_status(msg.payload)
                logger.debug("Vision status update from CV")
        except Exception as exc:
            logger.warning("Invalid payload on %s: %s", msg.topic, exc)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect("127.0.0.1", 1883, keepalive=30)
        client.loop_start()
        _scene_sub_client = client
    except Exception as exc:
        logger.warning("Could not start MQTT scene subscriber: %s", exc)


def stop_scene_subscriber() -> None:
    global _scene_sub_client
    if _scene_sub_client is not None:
        try:
            _scene_sub_client.loop_stop()
            _scene_sub_client.disconnect()
        except Exception:
            pass
        _scene_sub_client = None


def run_cv_pipeline():
    """Run the CV pipeline in a thread with output redirected to log file."""
    import os
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    cv_log_path = log_dir / "cv_pipeline.log"

    logger.info("Starting CV pipeline (output -> %s)", cv_log_path)

    with open(cv_log_path, "a") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            try:
                from assistant_cv.main import run_pipeline

                cv_logger = logging.getLogger("assistant_cv")
                cv_logger.setLevel(logging.WARNING)
                cv_logger.propagate = False

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


def run_brain_voice(scene_cache) -> None:
    """Run the brain voice assistant in a thread."""
    logger.info("Starting Brain voice assistant...")
    try:
        from brain.audio.io import AudioSettings, SoundDeviceSource, SoundDeviceSink
        from brain.audio.wake import WakeWordDetector
        from brain.audio.vad import VoiceActivityDetector
        from brain.audio.stt import SpeechToText
        from brain.audio.tts import TextToSpeech
        from brain.llm import LLMClient
        from brain.scene import SceneCache
        from brain.state import BrainStateMachine
        from brain.prompt import PromptBuilder
        from brain.tools.registry import ToolRegistry
        from brain.tools.scene_tools import register_scene_tools
        from brain.turn import TurnManager
        from brain.settings import load_settings
        from pathlib import Path

        MODEL_PATH = Path("models/Qwen3.5-4B-Q4_K_M.gguf")
        PIPER_VOICE_PATH = Path(__file__).parent / "models/piper/en_US-lessac-medium.onnx"

        # Load settings from config (llm.yaml)
        settings = load_settings()
        llm_settings = settings.llm
        # Ensure model path is absolute for llama-cpp
        llm_settings = llm_settings.model_copy(update={"host": str(MODEL_PATH.absolute())})
        audio_settings = settings.audio or AudioSettings()

        # Monkey-patch VAD for longer timeout
        original_collect = VoiceActivityDetector.collect_utterance
        def extended_collect(self, source, max_duration_s=20.0):
            return original_collect(self, source, max_duration_s=max_duration_s)
        VoiceActivityDetector.collect_utterance = extended_collect

        if not PIPER_VOICE_PATH.exists():
            logger.error("Piper voice not found at %s", PIPER_VOICE_PATH)
            return

        state_machine = BrainStateMachine(lambda t, p, q, r: None)
        wake = WakeWordDetector(audio_settings, model_name="hey_jarvis")
        vad = VoiceActivityDetector(audio_settings)
        stt = SpeechToText(audio_settings)
        tts = TextToSpeech(audio_settings, model_path=PIPER_VOICE_PATH)

        llm = LLMClient(llm_settings)
        # Model loads lazily on first chat() call (in the worker thread),
        # avoiding CUDA context thread-affinity issues from pre-warming on
        # the main thread.

        prompt_builder = PromptBuilder(
            scene_cache,
            max_turns=llm_settings.history_turns,
            max_total_tokens=llm_settings.history_max_tokens,
        )
        tools = ToolRegistry()
        register_scene_tools(tools, scene_cache)

        audio_sink = SoundDeviceSink(audio_settings.sample_rate)

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

        source = SoundDeviceSource(audio_settings.sample_rate)

        # Minimal ready message
        print("cv is ready, brain is ready", flush=True)
        print("type or speak to bmo", flush=True)

        # Start text input thread (typed messages share the same LLM->TTS pipeline)
        global _text_thread
        _text_thread = threading.Thread(
            target=run_text_input, args=(turn_manager,), daemon=True, name="Text-Input"
        )
        _text_thread.start()

        while not _shutdown.is_set():
            wake_detected = False
            while not wake_detected and not _shutdown.is_set():
                frame = source.read_frame()
                if frame is None:
                    continue
                result = turn_manager.process_audio_frame(frame, source=source)
                if result == "listening":
                    wake_detected = True

            if turn_manager._active_thread:
                turn_manager._active_thread.join(timeout=60.0)

        if turn_manager._active_thread and turn_manager._active_thread.is_alive():
            turn_manager._active_thread.join(timeout=30.0)

        audio_sink.close()
        source.close()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Brain voice error: %s", e, exc_info=True)
    finally:
        logger.info("Brain voice assistant stopped")


def run_text_input(turn_manager) -> None:
    """Read typed lines from stdin and feed them into the same turn pipeline.

    Reuses TurnManager.start_text_turn -> run_text_turn, so typed text goes
    through LLM -> TTS -> speaker exactly like voice input. Runs until
    _shutdown is set or the user types 'quit'/'exit'.
    """
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

        try:
            # Use run_text_turn for synchronous response display
            reply = turn_manager.run_text_turn(text)
            if reply:
                print(f"bmo: {reply.strip()}", flush=True)
        except Exception as exc:
            logger.error("Text input turn failed: %s", exc)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Shutdown signal received...")
    _shutdown.set()


def main():
    """Main entry point - starts all components in coordination."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. Verify MQTT broker (mosquitto) is reachable
    start_mqtt_broker()

    # Shared scene cache: CV writes via MQTT subscriber, brain reads for LLM context
    from brain.settings import SceneSettings
    from brain.scene import SceneCache
    scene_cache = SceneCache(SceneSettings())

    # Start MQTT scene subscriber (CV -> brain scene cache, most recent frame only)
    start_scene_subscriber(scene_cache)

    # 2. Start CV Pipeline (in background thread)
    global _cv_thread
    _cv_thread = threading.Thread(target=run_cv_pipeline, daemon=True, name="CV-Pipeline")
    _cv_thread.start()

    # Give CV time to connect to broker
    time.sleep(3)

    # 3. Start Brain Voice Assistant (main thread)
    global _brain_thread
    _brain_thread = threading.Thread(
        target=run_brain_voice, args=(scene_cache,), daemon=True, name="Brain-Voice"
    )
    _brain_thread.start()

    try:
        while not _shutdown.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown.set()
        stop_scene_subscriber()
        stop_mqtt_broker()


if __name__ == "__main__":
    main()
