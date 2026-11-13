"""Single process entrypoint for BMO Brain service."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import logging
import signal
import sys
import threading
import time
from typing import Any

from paho.mqtt.client import CallbackAPIVersion, Client, MQTTMessage
from pydantic import ValidationError

from bmo_contracts.models import (
    FaceData,
    FaceIdentified,
    PersonData,
    PersonDetected,
    Scene,
    VisionCmdResult,
    VisionStatus,
    parse_event_json,
)
from bmo_contracts.topics import (
    BRAIN_STATUS,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
    VISION_STATUS,
)
from brain.audio.io import AudioSink, AudioSource, MemorySink, MemorySource
from brain.audio.stt import SpeechToText
from brain.audio.tts import TextToSpeech
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.edge_monitor import EdgeMonitor
from brain.lifecycle import IdleLifecycleManager
from brain.llm import LLMClient
from brain.policy import ProactivePolicy
from brain.privacy import sanitize_payload
from brain.prompt import PromptBuilder
from brain.scene import SceneCache
from brain.settings import AudioSettings, PolicySettings, SceneSettings, Settings, load_settings
from brain.state import BrainState, BrainStateMachine, InvalidTransitionError
from brain.tools.identity_tools import register_identity_tools
from brain.tools.registry import ToolRegistry
from brain.tools.scene_tools import register_scene_tools
from brain.turn import TurnManager
from brain.vision_client import VisionClient


class BrainService:
    """Unified BMO Brain service orchestrating vision, audio, dialogue, and state."""

    def __init__(
        self,
        settings: Settings | None = None,
        host: str = "127.0.0.1",
        port: int = 1883,
        username: str = "brain",
        password: str = "brain_insecure_dev_password",
        source: AudioSource | None = None,
        sink: AudioSink | None = None,
        llm_client: LLMClient | None = None,
        wake: WakeWordDetector | None = None,
        vad: VoiceActivityDetector | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        clock: Callable[[], float] = time.monotonic,
        mqtt_client: Client | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._host = host
        self._port = port
        self._clock = clock

        self._client = mqtt_client or Client(CallbackAPIVersion.VERSION2)
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.will_set(BRAIN_STATUS, BrainStateMachine.last_will_payload(), qos=1, retain=True)

        self.state_machine = BrainStateMachine(publish_fn=self._publish_raw)
        scene_settings = self._settings.scene or SceneSettings()
        self.scene_cache = SceneCache(settings=scene_settings, clock=self._clock)
        self.vision_client = VisionClient(publish_fn=self._publish_raw, clock=self._clock)
        self.llm = llm_client or LLMClient(settings=self._settings.llm)

        self.tools = ToolRegistry(clock=self._clock)
        register_scene_tools(self.tools, self.scene_cache)
        register_identity_tools(self.tools, self.vision_client)

        audio_settings = self._settings.audio or AudioSettings()
        self.sink = sink or MemorySink()
        self.source = source or MemorySource(b"")
        self.wake = wake or WakeWordDetector(settings=audio_settings, clock=self._clock)
        self.vad = vad or VoiceActivityDetector(settings=audio_settings)
        self.stt = stt or SpeechToText(settings=audio_settings, clock=self._clock)
        self.tts = tts or TextToSpeech(settings=audio_settings, clock=self._clock)

        self.prompt_builder = PromptBuilder(scene_cache=self.scene_cache)
        self.turn_manager = TurnManager(
            state_machine=self.state_machine,
            wake=self.wake,
            vad=self.vad,
            stt=self.stt,
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            tools=self.tools,
            tts=self.tts,
            sink=self.sink,
            publish_fn=self._publish_raw,
            clock=self._clock,
        )

        self.edge_monitor = EdgeMonitor(
            state_machine=self.state_machine,
            fallback_source=self.source,
            fallback_sink=self.sink,
            clock=self._clock,
        )

        self._policy_settings = self._settings.policy or PolicySettings(
            face_min_conf=0.65,
            greeting_cooldown_s=900.0,
            global_cooldown_s=30.0,
            walk_by_window_s=1.5,
            idle_unload_s=300.0,
            tick_interval_s=1.0,
        )
        self.policy = ProactivePolicy(
            settings=self._policy_settings,
            state_machine=self.state_machine,
            on_greet=self._on_proactive_greet,
            clock=self._clock,
        )
        self.lifecycle = IdleLifecycleManager(
            settings=self._policy_settings,
            scene_cache=self.scene_cache,
            state_machine=self.state_machine,
            vision_client=self.vision_client,
            llm_client=self.llm,
            clock=self._clock,
        )

        self._running = False
        self._audio_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def state(self) -> BrainState:
        return self.state_machine.state

    def _publish_raw(self, topic: str, payload: bytes | str, qos: int = 1, retain: bool = False) -> None:
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        self._client.publish(topic, raw, qos=qos, retain=retain)

    def _on_proactive_greet(self, greeting: str) -> None:
        self.lifecycle.notify_activity()
        if self.state_machine.state == BrainState.IDLE:
            self.turn_manager.start_text_turn(greeting)

    def _on_connect(
        self,
        client: Client,
        _userdata: object,
        _flags: object,
        _rc: object,
        _properties: object = None,
    ) -> None:
        client.subscribe(
            [
                (VISION_CONTEXT, 0),
                (VISION_STATUS, 1),
                (VISION_EVENTS, 1),
                (VISION_CMD_RESULT, 1),
            ]
        )
        self.state_machine.publish_current()

    def _on_message(
        self,
        _client: Client,
        _userdata: object,
        msg: MQTTMessage,
    ) -> None:
        try:
            self.handle_message(msg.topic, msg.payload)
        except ValidationError as exc:
            logging.warning("MQTT validation error on topic '%s': %s", msg.topic, exc)
        except Exception as exc:
            logging.exception("MQTT handler error on topic '%s': %s", msg.topic, exc)

    def handle_message(self, topic: str, payload: bytes) -> None:
        _ = sanitize_payload({"topic": topic, "len": len(payload)})
        self.lifecycle.notify_activity()

        if topic == VISION_CONTEXT:
            scene = Scene.model_validate_json(payload)
            self.scene_cache.update_scene(scene)
            if scene.people and self.state_machine.state == BrainState.IDLE:
                for p in scene.people:
                    if p.name and p.name not in ("person", "unknown", ""):
                        evt = FaceIdentified(
                            id=f"scene-{p.track_id or p.name}",
                            ts=scene.ts,
                            data=FaceData(track_id=p.track_id, name=p.name, conf=1.0),
                        )
                    else:
                        evt = PersonDetected(
                            id=f"scene-{p.track_id or 'anon'}",
                            ts=scene.ts,
                            data=PersonData(track_id=p.track_id, pos=p.pos, proximity=p.proximity),
                        )
                    self.policy.handle_event(evt)
        elif topic == VISION_STATUS:
            status = VisionStatus.model_validate_json(payload)
            self.scene_cache.update_status(status)
        elif topic == VISION_EVENTS:
            event = parse_event_json(payload)
            self.policy.handle_event(event)
            self.lifecycle.handle_event(event)
        elif topic == VISION_CMD_RESULT:
            cmd_result = VisionCmdResult.model_validate_json(payload)
            self.vision_client.handle_cmd_result(cmd_result)

    def _audio_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.state_machine.state == BrainState.LISTENING:
                    time.sleep(0.02)
                    continue

                frame = self.edge_monitor.read_frame(timeout_s=0.05)
                if frame is not None:
                    try:
                        event = self.turn_manager.process_audio_frame(
                            frame, source=self.edge_monitor.active_source
                        )
                        if event is not None or self.state_machine.state != BrainState.IDLE:
                            self.lifecycle.notify_activity()
                    except InvalidTransitionError as exc:
                        logging.warning("Audio loop transition error: %s", exc)
                else:
                    time.sleep(0.02)
            except Exception as exc:
                logging.exception("Audio loop error: %s", exc)
                time.sleep(0.02)

    def _tick_loop(self) -> None:
        interval = self._policy_settings.tick_interval_s
        while not self._stop_event.is_set():
            try:
                self.policy.tick()
                self.lifecycle.tick()
            except Exception as exc:
                logging.exception("Error in tick loop: %s", exc)
            self._stop_event.wait(timeout=interval)

    def start(self, run_audio_loop: bool = True, run_tick_loop: bool = True) -> None:
        self._running = True
        self._stop_event.clear()
        self._client.connect(self._host, self._port)
        self._client.loop_start()
        if run_audio_loop:
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()
        if run_tick_loop:
            self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
            self._tick_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)
            self._audio_thread = None
        if self._tick_thread and self._tick_thread.is_alive():
            self._tick_thread.join(timeout=1.0)
            self._tick_thread = None
        if self._running:
            try:
                self._publish_raw(BRAIN_STATUS, BrainStateMachine.last_will_payload(), qos=1, retain=True)
            except Exception as exc:
                logging.warning("Failed to publish last will on stop: %s", exc)
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as exc:
                logging.warning("Error stopping MQTT client: %s", exc)
            self._running = False
        self.sink.close()
        self.source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BMO Brain Service")
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args(argv)

    service = BrainService(host=args.host, port=args.port)
    stop_signal = threading.Event()

    def _signal_handler(_sig: int, _frame: Any) -> None:
        stop_signal.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    service.start()
    try:
        while not stop_signal.is_set():
            time.sleep(0.5)
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
