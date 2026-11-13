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
import uuid
import warnings
import threading
import time
import signal
import sys
import socket
import json
from pathlib import Path

import paho.mqtt.client as mqtt

from bmo_contracts.models import BrainSay, Emotion, Scene
from bmo_contracts.topics import BRAIN_SAY, BRAIN_STATUS, VISION_CONTEXT, VISION_STATUS, VISION_CMD
from brain.state import BrainState
from brain.turn import resolve_vision_target
from brain.emotions import EmotionTagStreamFilter, split_emotion_tag, strip_emotion_tags
from brain.memory_extract import apply_memory_effects
from brain.tools.registry import needs_tools
from face_ui.signal import FaceSignal, default_path

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
_broker_log_file = None
_cv_proc = None
_cv_log_file = None
_face_proc = None
_face_log_file = None
_brain_thread = None
_text_thread = None
_scene_sub_client = None
_shutdown = threading.Event()
# Set while text-mode TTS is playing so the mic does not transcribe our own voice.
_tts_active = threading.Event()
_tts_quiet_until = 0.0


def _broker_reachable(timeout_s: float = 0.5) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout_s)
        s.connect(("127.0.0.1", 1883))
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_mqtt_broker():
    """Ensure a local mosquitto broker is listening on 127.0.0.1:1883.

    Starts one automatically when it is installed but not running, so CV and
    the brain can always exchange scenes. A broker started here is stopped on
    shutdown; an externally managed broker is left alone.
    """
    global _broker_proc, _broker_log_file
    _broker_proc = None
    _broker_log_file = None
    if _broker_reachable():
        logger.info("MQTT broker (mosquitto) reachable on 127.0.0.1:1883")
        return

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "mosquitto.log"
    try:
        _broker_log_file = open(log_path, "a", encoding="utf-8")
        _broker_proc = subprocess.Popen(
            ["mosquitto", "-p", "1883"],
            stdout=_broker_log_file,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        logger.error("mosquitto is not installed; CV scenes cannot reach the brain")
        return
    except Exception as exc:
        logger.error("Failed to start mosquitto: %s", exc)
        return

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if _broker_proc.poll() is not None:
            logger.error("mosquitto exited during startup (see %s)", log_path)
            return
        if _broker_reachable():
            logger.info("Started mosquitto on 127.0.0.1:1883")
            return
        time.sleep(0.05)
    logger.error("mosquitto did not become reachable (see %s)", log_path)


def stop_mqtt_broker():
    """Stop the mosquitto process only if this run started it."""
    global _broker_proc, _broker_log_file
    proc = _broker_proc
    _broker_proc = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if _broker_log_file is not None:
        try:
            _broker_log_file.close()
        except Exception:
            pass
        _broker_log_file = None


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


def request_vision(scene_cache, target: str = "both", seconds: float = 6.0, timeout_s: float = 6.0) -> None:
    """Ask the CV pipeline to look for a specific target, then wait for it.

    CV runs idle (no object detection) by default to keep the CPU free. A
    vision question enables only the relevant models (people or objects) for a
    few seconds via MQTT. We wait for a scene that actually contains the
    requested kind of thing (so we do not answer from a premature empty frame).
    """
    client = _scene_sub_client
    if client is None:
        logger.warning("Vision look skipped: scene subscriber is not connected")
        return
    before = scene_cache.last_update_time()
    request_id = uuid.uuid4().hex[:8]
    payload = json.dumps(
        {
            "command": "look",
            "args": {
                "seconds": seconds,
                "target": target,
                "request_id": request_id,
            },
        }
    )
    try:
        client.publish(VISION_CMD, payload, qos=1)
    except Exception as exc:
        logger.warning("Failed to publish vision look request: %s", exc)
        return

    deadline = time.monotonic() + timeout_s
    last_request = time.monotonic()
    while time.monotonic() < deadline and not _shutdown.is_set():
        if scene_cache.last_update_time() > before:
            scene = scene_cache.current()
            if _scene_has_target(scene, target):
                logger.info("Fresh %s scene received for vision query", target)
                return
        # The CV process may still be loading its models when the first look is
        # sent. Repeat it so the request is not lost before CV subscribes.
        if time.monotonic() - last_request >= 1.0:
            try:
                client.publish(VISION_CMD, payload, qos=1)
            except Exception:
                pass
            last_request = time.monotonic()
        time.sleep(0.05)
    logger.warning("No %s in fresh scene within %.1fs; answering with last known scene", target, timeout_s)


def _kill_stale_cv() -> None:
    """Terminate a leftover CV pipeline that may still hold the camera."""
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"assistant_cv.main" not in cmdline:
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        logger.warning("Stopping leftover CV process pid=%s", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        for _ in range(40):
            time.sleep(0.05)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def start_cv_process() -> None:
    """Start the CV pipeline as a separate process.

    Running it out-of-process keeps its chatty startup (and OpenCV/GTK) from
    touching this process's stdout/window loop. It runs headless, so a look
    never opens a camera window over the face.
    """
    global _cv_proc, _cv_log_file

    _kill_stale_cv()
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


def start_face_process() -> None:
    """Start the BMO face window as a separate process.

    The face polls the shared state file written by the brain (MQTT is only a
    fallback), so it can be restarted independently. It is skipped when no
    display is available or BMO_FACE_UI is set to a false-y value.
    """
    global _face_proc, _face_log_file

    enabled = os.environ.get("BMO_FACE_UI")
    if enabled is None:
        enabled = "1" if os.environ.get("DISPLAY") else "0"
    if enabled.strip().lower() in ("0", "false", "no", "off", ""):
        logger.info("Face UI disabled (BMO_FACE_UI=%s, DISPLAY=%s)",
                    enabled, os.environ.get("DISPLAY", "-"))
        return

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    face_log_path = log_dir / "face_ui.log"
    _face_log_file = open(face_log_path, "a", encoding="utf-8")

    # A leftover state file from the previous run must not wake the face on
    # boot. It always starts asleep; the brain publishes a fresh idle status.
    try:
        default_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not clear stale face state: %s", exc)

    size = os.environ.get("BMO_FACE_SIZE", "").strip()
    try:
        sleep_after_s = float(os.environ.get("BMO_FACE_SLEEP_S", "60"))
    except ValueError:
        sleep_after_s = 60.0
    # BMO should stay awake for a full minute of inactivity; 0 still disables
    # sleeping explicitly for kiosk/debug runs.
    if 0 < sleep_after_s < 60.0:
        logger.warning(
            "BMO_FACE_SLEEP_S=%.1fs is too short; using 60s", sleep_after_s
        )
        sleep_after_s = 60.0
    sleep_s = f"{sleep_after_s:g}"
    cmd = [
        sys.executable,
        "-m",
        "face_ui.main",
        "--mqtt-host",
        "127.0.0.1",
        "--mqtt-port",
        "1883",
        "--sleep-after",
        sleep_s,
    ]
    if size:
        cmd.extend(["--size", size])
    if os.environ.get("BMO_FACE_FULLSCREEN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        cmd.append("--fullscreen")
    if os.environ.get("BMO_FACE_NOQUIT", "0").strip().lower() in ("1", "true", "yes", "on"):
        cmd.append("--no-quit")
    if os.environ.get("BMO_FACE_ART", "1").strip().lower() in ("0", "false", "no", "off"):
        cmd.append("--no-art")
    display = os.environ.get("BMO_FACE_DISPLAY")
    if display:
        cmd.extend(["--display", display])

    child_env = os.environ.copy()
    if display:
        child_env["DISPLAY"] = display

    logger.info("Starting face UI (output -> %s)", face_log_path)
    try:
        _face_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=_face_log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
    except Exception as exc:
        logger.error("Failed to start face UI: %s", exc)
        _face_proc = None


def stop_face_process() -> None:
    global _face_proc, _face_log_file
    proc = _face_proc
    _face_proc = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if _face_log_file is not None:
        try:
            _face_log_file.close()
        except Exception:
            pass
        _face_log_file = None


def run_brain_voice(scene_cache) -> None:
    """Run the brain voice assistant in a thread."""
    logger.info("Starting Brain voice assistant...")
    pub_client = None
    try:
        from brain.audio.io import AudioSettings, SoundDeviceSource, SoundDeviceSink
        from brain.audio.wake import WakeWordDetector, extract_wake_command
        from brain.audio.vad import VoiceActivityDetector
        from brain.audio.stt import SpeechToText
        from brain.audio.tts import TextToSpeech
        from brain.llm import LLMClient
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

        face_signal = FaceSignal()

        def publish_fn(topic, payload, qos=0, retain=False):
            if pub_client is not None:
                try:
                    pub_client.publish(topic, payload, qos=qos, retain=retain)
                except Exception:
                    pass
            try:
                if topic == BRAIN_STATUS:
                    data = json.loads(payload)
                    face_signal.set_status(
                        data.get("state", "idle"), bool(data.get("online", True))
                    )
                elif topic == BRAIN_SAY:
                    data = json.loads(payload)
                    face_signal.set_say(
                        data.get("text", ""), data.get("emotion", "neutral")
                    )
            except Exception as exc:
                logger.debug("Face signal update failed: %s", exc)

        try:
            pub_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            pub_client.will_set(
                BRAIN_STATUS,
                BrainStateMachine.last_will_payload(),
                qos=1,
                retain=True,
            )
            pub_client.connect("127.0.0.1", 1883, keepalive=30)
            pub_client.loop_start()
            logger.info("Brain MQTT publisher connected")
        except Exception as exc:
            logger.warning("Brain MQTT publisher unavailable: %s", exc)

        state_machine = BrainStateMachine(publish_fn)
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

        audio_sink = SoundDeviceSink(
            audio_settings.sample_rate,
            device=audio_device,
            speech_cb=face_signal.set_speech,
        )

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
            publish_fn=publish_fn,
            vision_request_fn=lambda target: request_vision(scene_cache, target=target),
        )
        state_machine.publish_current()

        def _say_static(text: str, emotion: Emotion = Emotion.HAPPY) -> None:
            """Speak a canned line (wake greeting) without calling the LLM."""
            global _tts_quiet_until
            if not text.strip():
                return
            _tts_active.set()
            print(f"bmo: {text}", flush=True)
            try:
                if state_machine.state != BrainState.SPEAKING:
                    try:
                        state_machine.transition(BrainState.SPEAKING)
                    except Exception:
                        pass
                try:
                    say = BrainSay(
                        utt_id=uuid.uuid4().hex[:8],
                        text=text,
                        emotion=emotion,
                    )
                    publish_fn(BRAIN_SAY, say.model_dump_json().encode("utf-8"), 1, False)
                except Exception as exc:
                    logger.warning("BrainSay publish failed: %s", exc)
                tts.speak_sentence(text, audio_sink)
                audio_sink.drain()
            except Exception as exc:
                logger.error("Static speech failed: %s", exc, exc_info=True)
            finally:
                _tts_active.clear()
                _tts_quiet_until = time.monotonic() + 0.8
                try:
                    if state_machine.state == BrainState.SPEAKING:
                        state_machine.transition(BrainState.IDLE)
                except Exception:
                    pass

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
                    try:
                        if state_machine.state == BrainState.IDLE:
                            state_machine.transition(BrainState.LISTENING)
                    except Exception:
                        pass
                    try:
                        reply = turn_manager.run_text_turn(command, followup=in_followup)
                        if not reply and state_machine.state == BrainState.LISTENING:
                            try:
                                state_machine.transition(BrainState.IDLE)
                            except Exception:
                                pass
                    finally:
                        if state_machine.state == BrainState.LISTENING:
                            try:
                                state_machine.transition(BrainState.IDLE)
                            except Exception:
                                pass
                else:
                    # Bare wake word: answer immediately, then listen for the
                    # actual request inside the follow-up window.
                    _say_static("Hey there!")
                followup_until = time.monotonic() + followup_s
            except Exception as exc:
                logger.error("Turn failed: %s", exc, exc_info=True)
                if state_machine.state in (BrainState.LISTENING, BrainState.THINKING, BrainState.SPEAKING):
                    try:
                        state_machine.transition(BrainState.IDLE)
                    except Exception:
                        pass

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
        if pub_client is not None:
            try:
                pub_client.loop_stop()
                pub_client.disconnect()
            except Exception:
                pass
        logger.info("Brain voice assistant stopped")


def run_text_input(turn_manager) -> None:
    """Read typed lines from stdin and feed them into the same turn pipeline.

    For text mode, print LLM chunks immediately (streaming) and optionally speak via TTS.
    Runs until _shutdown is set or the user types 'quit'/'exit'.
    """
    import sys
    import os
    import uuid
    stdout = sys.__stdout__  # Use unbuffered original stdout (thread-safe)
    
    # Speak typed LLM replies by default; set BMO_TEXT_TTS=0 to disable.
    use_tts = os.environ.get("BMO_TEXT_TTS", "1") == "1"
    tts = turn_manager._tts
    sink = turn_manager._sink
    state_machine = turn_manager._state_machine
    publisher = turn_manager._publish_fn

    global _tts_quiet_until

    def _safe_transition(target: BrainState) -> None:
        try:
            state_machine.transition(target)
        except Exception:
            pass

    last_vision_target: str | None = None
    last_vision_at = 0.0

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
            with turn_manager._turn_gate:
                try:
                    memory = getattr(turn_manager._prompt_builder, "memory", None)
                    if memory is not None:
                        try:
                            for note in apply_memory_effects(text, memory):
                                logger.info("memory: %s", note)
                        except Exception as exc:
                            logger.warning("memory update failed: %s", exc)

                    # Ask CV for a fresh look before answering camera questions,
                    # enabling only the models relevant to the question. Typed
                    # follow-ups ("is it red?") reuse the previous scene target too.
                    target = resolve_vision_target(
                        text,
                        followup=True,
                        last_target=last_vision_target,
                        last_at=last_vision_at,
                        now=time.monotonic(),
                    )
                    if target:
                        last_vision_target = target
                        last_vision_at = time.monotonic()
                    if target and turn_manager._vision_request_fn is not None:
                        try:
                            turn_manager._vision_request_fn(target)
                        except Exception as exc:
                            logger.warning("Vision request failed: %s", exc)

                    # Stream LLM response immediately for text mode, stripping every
                    # [emotion] tag (the model sometimes emits one per sentence).
                    _safe_transition(BrainState.THINKING)
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
                        if chunk.tool_call is not None:
                            try:
                                turn_manager._tools.dispatch(chunk.tool_call, confirmed=False)
                            except Exception as tool_exc:
                                logger.error("Tool dispatch failed: %s", tool_exc)
                        if not chunk.text:
                            continue
                        full_reply.append(chunk.text)
                        _emit(tag_filter.feed(chunk.text))
                    _emit(tag_filter.flush())
                    print(flush=True, file=stdout)

                    raw_reply = "".join(full_reply)
                    reply_text = strip_emotion_tags(raw_reply)
                    emotion, _ = split_emotion_tag(raw_reply)
                    emotion = emotion or Emotion.NEUTRAL

                    # Optionally speak the response via TTS
                    if use_tts and tts is not None and sink is not None and reply_text.strip():
                        _safe_transition(BrainState.SPEAKING)
                        if publisher is not None:
                            try:
                                say = BrainSay(
                                    utt_id=uuid.uuid4().hex[:8],
                                    text=reply_text,
                                    emotion=emotion,
                                )
                                publisher(
                                    BRAIN_SAY,
                                    say.model_dump_json().encode("utf-8"),
                                    1,
                                    False,
                                )
                            except Exception as publish_exc:
                                logger.warning("BrainSay publish failed: %s", publish_exc)
                        logger.info("Text input TTS: speaking reply (%d chars)", len(reply_text))
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
                finally:
                    if state_machine.state in (BrainState.THINKING, BrainState.SPEAKING):
                        _safe_transition(BrainState.IDLE)
        except Exception as exc:
            logger.error("Text input turn failed: %s", exc)


def signal_handler(_signum, _frame):
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

    # 3. Start the BMO face display (separate process, MQTT-driven)
    start_face_process()

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
        stop_cv_process()
        stop_face_process()
        stop_scene_subscriber()
        stop_mqtt_broker()


if __name__ == "__main__":
    import os

    try:
        main()
    finally:
        # Daemon threads hold C-extension state (CUDA/camera/torch); normal
        # interpreter teardown can segfault. Exit immediately after shutdown.
        logging.shutdown()
        os._exit(0)
