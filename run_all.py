#!/usr/bin/env python3
"""
BMO Full System - Coordinated Startup
Runs: MQTT Broker -> CV Pipeline -> Brain Voice Assistant
CV publishes Scene via MQTT; brain subscribes and takes the most recent frame.
CLI is used only for text input to the LLM.
"""

import logging
import os
import subprocess
import warnings
import threading
import time
import signal
import sys
import socket
import json
from pathlib import Path

import paho.mqtt.client as mqtt

from bmo_contracts.models import Scene, VisionStatus
from bmo_contracts.topics import VISION_CONTEXT, VISION_STATUS, VISION_CMD
from brain.vision_intent import vision_target
from brain.emotions import EmotionTagStreamFilter, strip_emotion_tag
from brain.memory_extract import apply_memory_effects
from brain.tools.registry import needs_tools

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
_cv_proc = None
_cv_log_file = None
_brain_thread = None
_text_thread = None
_scene_sub_client = None
_shutdown = threading.Event()
# Set while text-mode TTS is playing so the mic does not transcribe our own voice.
_tts_active = threading.Event()
_tts_quiet_until = 0.0


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


def _scene_has_target(scene, target: str) -> bool:
    if scene is None:
        return False
    people = getattr(scene, "people", None) or []
    objects = getattr(scene, "objects", None) or []
    if target in ("people", "person", "face"):
        return len(people) > 0
    if target in ("objects", "object", "thing", "item"):
        return len(objects) > 0
    return len(people) > 0 or len(objects) > 0


def request_vision(scene_cache, target: str = "both", seconds: float = 6.0, timeout_s: float = 4.0) -> None:
    """Ask the CV pipeline to look for a specific target, then wait for it.

    CV runs idle (no object detection) by default to keep the CPU free. A
    vision question enables only the relevant models (people or objects) for a
    few seconds via MQTT. We wait for a scene that actually contains the
    requested kind of thing (so we do not answer from a premature empty frame).
    """
    client = _scene_sub_client
    if client is None:
        return
    before = scene_cache.last_update_time()
    payload = json.dumps({"command": "look", "args": {"seconds": seconds, "target": target}})
    try:
        client.publish(VISION_CMD, payload, qos=1)
    except Exception as exc:
        logger.warning("Failed to publish vision look request: %s", exc)
        return

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not _shutdown.is_set():
        if scene_cache.last_update_time() > before:
            scene = scene_cache.current()
            if _scene_has_target(scene, target):
                logger.info("Fresh %s scene received for vision query", target)
                return
        time.sleep(0.05)
    logger.warning("No %s in fresh scene within %.1fs; answering with last known scene", target, timeout_s)


def start_cv_process() -> None:
    """Start the CV pipeline as a separate process.

    Running it out-of-process keeps its chatty startup (and OpenCV/GTK) from
    touching this process's stdout/window loop, and lets the on-demand camera
    window live on the CV process's main thread.
    """
    global _cv_proc, _cv_log_file

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    cv_log_path = log_dir / "cv_pipeline.log"
    _cv_log_file = open(cv_log_path, "a", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "assistant_cv.main",
        "--headless",
        "--mqtt-host",
        "127.0.0.1",
        "--mqtt-port",
        "1883",
    ]
    logger.info("Starting CV process (output -> %s)", cv_log_path)
    try:
        _cv_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=_cv_log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except Exception as exc:
        logger.error("Failed to start CV process: %s", exc)
        _cv_proc = None


def stop_cv_process() -> None:
    global _cv_proc, _cv_log_file
    proc = _cv_proc
    _cv_proc = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if _cv_log_file is not None:
        try:
            _cv_log_file.close()
        except Exception:
            pass
        _cv_log_file = None


def run_brain_voice(scene_cache) -> None:
    """Run the brain voice assistant in a thread."""
    logger.info("Starting Brain voice assistant...")
    try:
        from brain.audio.io import AudioSettings, SoundDeviceSource, SoundDeviceSink
        from brain.audio.wake import WakeWordDetector, extract_wake_command
        from brain.audio.vad import VoiceActivityDetector
        from brain.audio.stt import SpeechToText
        from brain.audio.tts import TextToSpeech
        from brain.llm import LLMClient
        from brain.scene import SceneCache
        from brain.state import BrainStateMachine
        from brain.prompt import PromptBuilder
        from brain.tools.registry import ToolRegistry
        from brain.tools.scene_tools import register_scene_tools
        from brain.tools.memory_tools import register_memory_tools
        from brain.memory import LongTermMemory
        from brain.turn import TurnManager
        from brain.settings import load_settings
        from pathlib import Path
        import os

        MODEL_PATH = Path("models/Qwen3.5-4B-Q4_K_M.gguf")
        PIPER_VOICE_PATH = Path(__file__).parent / "models/piper/bmo.onnx"

        # Load settings from config (llm.yaml)
        settings = load_settings()
        llm_settings = settings.llm
        # Ensure model path is absolute for llama-cpp
        llm_settings = llm_settings.model_copy(update={"host": str(MODEL_PATH.absolute())})
        audio_settings = settings.audio or AudioSettings()

        # Audio output device (optional, for systems with multiple audio devices)
        audio_device = os.environ.get("BMO_AUDIO_DEVICE")
        if audio_device is not None:
            try:
                audio_device = int(audio_device)
            except ValueError:
                pass  # Use string device name

        # Monkey-patch VAD for longer timeout
        original_collect = VoiceActivityDetector.collect_utterance
        def extended_collect(self, source, max_duration_s=20.0):
            return original_collect(self, source, max_duration_s=max_duration_s)
        VoiceActivityDetector.collect_utterance = extended_collect

        if not PIPER_VOICE_PATH.exists():
            logger.error("Piper voice not found at %s", PIPER_VOICE_PATH)
            return

        state_machine = BrainStateMachine(lambda t, p, q, r: None)
        # The wake gate is handled by Whisper transcription below ("hey bmo"),
        # since openWakeWord has no BMO model. This detector is only passed to
        # TurnManager for interface compatibility and stays unused.
        wake = WakeWordDetector(audio_settings)
        vad = VoiceActivityDetector(audio_settings)
        stt = SpeechToText(audio_settings)
        tts = TextToSpeech(audio_settings, model_path=PIPER_VOICE_PATH)

        llm = LLMClient(llm_settings)
        # Load the model now (same thread as voice turns) so the first spoken
        # reply does not pay the multi-second load cost.
        try:
            llm.prewarm()
        except Exception as exc:
            logger.warning("LLM prewarm failed: %s", exc)

        memory = LongTermMemory(Path(__file__).parent / "data" / "memory.json")
        prompt_builder = PromptBuilder(
            scene_cache,
            max_turns=llm_settings.history_turns,
            max_total_tokens=max(1024, llm_settings.num_ctx - 1024),
            memory=memory,
        )
        tools = ToolRegistry()
        register_scene_tools(tools, scene_cache)
        register_memory_tools(tools, memory)

        audio_sink = SoundDeviceSink(audio_settings.sample_rate, device=audio_device)

        # Confirm audio output works before entering the turn loop.
        try:
            tts.speak_sentence("BMO voice system ready", audio_sink)
            audio_sink.drain()
            logger.info("Audio output check passed")
        except Exception as exc:
            logger.error("Audio output check failed: %s", exc, exc_info=True)

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
            vision_request_fn=lambda target: request_vision(scene_cache, target=target),
        )

        source = SoundDeviceSource(audio_settings.sample_rate)

        # Minimal ready message
        print("cv is ready, brain is ready", flush=True)
        print('say "hey bmo" then your question, or type below', flush=True)

        # Start text input thread (typed messages share the same LLM->TTS pipeline)
        global _text_thread
        _text_thread = threading.Thread(
            target=run_text_input, args=(turn_manager,), daemon=True, name="Text-Input"
        )
        _text_thread.start()

        # Continuous listen loop: segment one utterance with VAD, transcribe it,
        # and respond when it starts with the "hey bmo" wake phrase. For a short
        # window after a reply, plain follow-up questions are accepted too.
        try:
            followup_s = float(os.environ.get("BMO_FOLLOWUP_S", "12"))
        except ValueError:
            followup_s = 12.0
        followup_until = 0.0
        while not _shutdown.is_set():
            try:
                pcm = vad.collect_utterance(source, max_duration_s=10.0)
            except Exception as exc:
                logger.warning("VAD collect_utterance failed: %s", exc)
                continue
            if pcm is None or _shutdown.is_set():
                continue

            # Ignore audio captured while (or just after) our own TTS plays.
            if _tts_active.is_set() or time.monotonic() < _tts_quiet_until:
                print("  (ignored own audio)", flush=True)
                continue

            stt_res = stt.transcribe(pcm)
            if stt_res is None or not stt_res.text.strip():
                continue

            heard = stt_res.text.strip()
            duration_s = len(pcm) / audio_settings.sample_rate
            command = extract_wake_command(heard)
            in_followup = False
            if command is None:
                if followup_s > 0 and time.monotonic() < followup_until:
                    command = heard
                    in_followup = True
                else:
                    print(f'  heard {duration_s:.1f}s: "{heard}"  (no wake word)', flush=True)
                    continue

            tag = "follow-up" if in_followup else "wake"
            print(f'  heard {duration_s:.1f}s: "{heard}"  <- {tag}', flush=True)
            try:
                if command:
                    turn_manager.run_text_turn(command)
                else:
                    turn_manager.listen_and_respond(source)
                followup_until = time.monotonic() + followup_s
            except Exception as exc:
                logger.error("Turn failed: %s", exc, exc_info=True)

        if turn_manager._active_thread and turn_manager._active_thread.is_alive():
            turn_manager._active_thread.join(timeout=30.0)

        try:
            audio_sink.close()
        except Exception as exc:
            logger.error("Error closing audio sink: %s", exc)
        try:
            source.close()
        except Exception as exc:
            logger.error("Error closing audio source: %s", exc)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Brain voice error: %s", e, exc_info=True)
    finally:
        logger.info("Brain voice assistant stopped")


def run_text_input(turn_manager) -> None:
    """Read typed lines from stdin and feed them into the same turn pipeline.

    For text mode, print LLM chunks immediately (streaming) and optionally speak via TTS.
    Runs until _shutdown is set or the user types 'quit'/'exit'.
    """
    import sys
    import os
    stdout = sys.__stdout__  # Use unbuffered original stdout (thread-safe)
    
    # Speak typed LLM replies by default; set BMO_TEXT_TTS=0 to disable.
    use_tts = os.environ.get("BMO_TEXT_TTS", "1") == "1"
    tts = turn_manager._tts
    sink = turn_manager._sink
    
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
            memory = getattr(turn_manager._prompt_builder, "memory", None)
            if memory is not None:
                try:
                    for note in apply_memory_effects(text, memory):
                        logger.info("memory: %s", note)
                except Exception as exc:
                    logger.warning("memory update failed: %s", exc)

            # Ask CV for a fresh look before answering camera questions,
            # enabling only the models relevant to the question.
            target = vision_target(text)
            if target and turn_manager._vision_request_fn is not None:
                try:
                    turn_manager._vision_request_fn(target)
                except Exception as exc:
                    logger.warning("Vision request failed: %s", exc)

            # Stream LLM response immediately for text mode, stripping every
            # [emotion] tag (the model sometimes emits one per sentence).
            print("bmo: ", end="", flush=True, file=stdout)
            full_reply = []
            tag_filter = EmotionTagStreamFilter()
            printed_any = False

            def _emit(piece: str) -> None:
                nonlocal printed_any
                if not printed_any:
                    piece = piece.lstrip()
                if piece:
                    print(piece, end="", flush=True, file=stdout)
                    printed_any = True

            for chunk in turn_manager._llm.chat(
                turn_manager._prompt_builder.build_messages(text, scene_target=target),
                tools=(
                    turn_manager._tools.ollama_tools()
                    if needs_tools(text)
                    else None
                ),
                stream=True,
                cancel_event=turn_manager.cancel_event,
            ):
                if not chunk.text:
                    continue
                full_reply.append(chunk.text)
                _emit(tag_filter.feed(chunk.text))
            _emit(tag_filter.flush())
            print(flush=True, file=stdout)

            reply_text = strip_emotion_tag("".join(full_reply))
            
            # Optionally speak the response via TTS
            if use_tts and tts is not None and sink is not None and reply_text.strip():
                logger.info("Text input TTS: speaking reply (%d chars)", len(reply_text))
                global _tts_quiet_until
                _tts_active.set()
                try:
                    tts.speak_sentence(reply_text, sink)
                    # Wait for audio to finish playing
                    if hasattr(sink, 'drain'):
                        sink.drain()
                    logger.info("Text input TTS: completed")
                except Exception as tts_exc:
                    logger.error("TTS failed: %s", tts_exc, exc_info=True)
                finally:
                    _tts_active.clear()
                    _tts_quiet_until = time.monotonic() + 0.8
            
            # Record turn in history
            turn_manager._prompt_builder.record_turn(text, reply_text)
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

    # 2. Start CV Pipeline (separate process)
    start_cv_process()

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
        stop_cv_process()


if __name__ == "__main__":
    import os

    try:
        main()
    finally:
        # Daemon threads hold C-extension state (CUDA/camera/torch); normal
        # interpreter teardown can segfault. Exit immediately after shutdown.
        logging.shutdown()
        os._exit(0)
