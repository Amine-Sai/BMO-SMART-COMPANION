"""End-to-end integration tests for BMO Brain and connected peers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from bmo_contracts.models import (
    Scene,
    ScenePerson,
    SignData,
    SignRecognized,
)
from brain.audio.io import AudioSource, MemorySink, MemorySource
from brain.audio.stt import SpeechToText
from brain.audio.tts import TextToSpeech
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.llm import ChatChunk, LLMClient, MsgList, ToolCall
from brain.main import BrainService
from brain.settings import AudioSettings, Settings
from brain.state import BrainState
from tools.mock_cv import MockCV
from tools.mock_pi import MockPi
from tools.simple_broker import SimpleMQTTBroker, parse_passwords_file


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for(condition: Callable[[], bool], timeout: float = 4.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return False


class E2EBackend:
    def __init__(self) -> None:
        self.reply_text: str = "Hello from BMO!"
        self.pending_tool_call: ToolCall | None = None

    def chat(self, messages: MsgList, **kwargs: object) -> Iterator[ChatChunk]:
        del messages, kwargs
        if self.pending_tool_call is not None:
            tc = self.pending_tool_call
            self.pending_tool_call = None
            yield ChatChunk(tool_call=tc)
        yield ChatChunk(text=self.reply_text, done=True)

    def unload(self) -> None:
        pass

    def prewarm(self) -> None:
        pass


@pytest.fixture(scope="module")
def broker_port() -> int:
    return get_free_port()


@pytest.fixture(scope="module")
def e2e_broker(broker_port: int) -> Iterator[SimpleMQTTBroker]:
    users = parse_passwords_file("infra/passwords.example")
    broker = SimpleMQTTBroker(
        host="127.0.0.1",
        port=broker_port,
        require_auth=True,
        users=users,
        acl_path="infra/acl",
    )
    loop = asyncio.new_event_loop()

    def run_broker() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broker.start())
        loop.run_forever()

    thread = threading.Thread(target=run_broker, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield broker
    try:
        asyncio.run_coroutine_threadsafe(broker.stop(), loop).result(timeout=2.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)


@pytest.fixture
def peers_and_brain(
    e2e_broker: SimpleMQTTBroker, broker_port: int
) -> Iterator[tuple[BrainService, MockCV, MockPi, E2EBackend, MemorySink]]:
    del e2e_broker
    backend = E2EBackend()
    audio_settings = AudioSettings()
    settings = Settings(audio=audio_settings)
    llm = LLMClient(settings=settings.llm, backend=backend)

    sink = MemorySink()
    source = MemorySource(b"")

    wake = WakeWordDetector(
        settings=audio_settings,
        score_fn=lambda _arr: 0.9,
    )
    vad = VoiceActivityDetector(
        settings=audio_settings,
        speech_prob_fn=lambda _arr: 0.9,
    )
    stt = SpeechToText(
        settings=audio_settings,
        transcribe_fn=lambda _pcm: ("What time is it?", 0.98),
    )
    tts = TextToSpeech(
        settings=audio_settings,
        synth_fn=lambda _text: np.zeros(320, dtype=np.int16),
    )

    brain = BrainService(
        host="127.0.0.1",
        port=broker_port,
        source=source,
        sink=sink,
        llm_client=llm,
        wake=wake,
        vad=vad,
        stt=stt,
        tts=tts,
    )
    brain.start(run_audio_loop=False)

    cv = MockCV(host="127.0.0.1", port=broker_port)
    cv.start()

    pi = MockPi(host="127.0.0.1", port=broker_port)
    pi.start()

    time.sleep(0.3)
    yield brain, cv, pi, backend, sink

    pi.stop()
    cv.stop()
    brain.stop()


def test_e2e_walking_in_greeting(
    peers_and_brain: tuple[BrainService, MockCV, MockPi, E2EBackend, MemorySink],
) -> None:
    brain, cv, pi, backend, _ = peers_and_brain
    backend.reply_text = "Hi Finn, welcome back!"

    # CV emits a scene containing a recognized person
    scene = Scene(
        ts=time.monotonic(),
        people=[
            ScenePerson(
                track_id=1,
                name="Finn",
                pos="center",
                proximity="near",
                since_s=2.0,
            )
        ],
        objects=[],
        recent=[],
    )
    cv.publish_scene(scene)

    assert wait_for(lambda: len(pi.say_messages) > 0), "Expected Pi to receive greeting BrainSay message"
    assert "Finn" in pi.say_messages[-1].text or "Hi" in pi.say_messages[-1].text
    assert wait_for(lambda: brain.state == BrainState.IDLE)


def test_e2e_spoken_query_pipeline(
    peers_and_brain: tuple[BrainService, MockCV, MockPi, E2EBackend, MemorySink],
) -> None:
    brain, _, pi, backend, sink = peers_and_brain
    backend.reply_text = "It is adventure time!"
    initial_say_count = len(pi.say_messages)

    # Feed audio frame through turn pipeline
    test_frame = np.ones(320, dtype=np.int16) * 1000
    test_utterance_source: AudioSource = MemorySource(np.ones(320 * 5, dtype=np.int16) * 1000)

    # Processing frame triggers wake -> VAD -> STT -> LLM -> TTS
    res = brain.turn_manager.process_audio_frame(test_frame, source=test_utterance_source)
    assert res == "It is adventure time!"

    assert wait_for(lambda: len(pi.say_messages) > initial_say_count)
    assert "adventure" in pi.say_messages[-1].text
    assert len(sink.get_samples()) > 0
    assert brain.state == BrainState.IDLE


def test_e2e_signed_phrase(
    peers_and_brain: tuple[BrainService, MockCV, MockPi, E2EBackend, MemorySink],
) -> None:
    brain, cv, pi, backend, _ = peers_and_brain
    backend.reply_text = "I see your sign!"
    initial_say_count = len(pi.say_messages)

    # Mock CV publishes sign recognized event
    event = SignRecognized(
        id="sign-evt-101",
        ts=time.monotonic(),
        data=SignData(text="HELP", conf=0.9),
    )
    cv.publish_event(event)

    assert wait_for(lambda: len(pi.say_messages) > initial_say_count)
    assert "sign" in pi.say_messages[-1].text
    assert wait_for(lambda: brain.state == BrainState.IDLE)


def test_e2e_enrollment_hands_free(
    peers_and_brain: tuple[BrainService, MockCV, MockPi, E2EBackend, MemorySink],
) -> None:
    brain, cv, _, _, _ = peers_and_brain
    initial_cmd_count = len(cv.received_cmds)

    # Dispatch enrollment tool hands-free
    res = brain.tools.dispatch(ToolCall(name="remember_person", arguments={"name": "Marceline"}))

    assert wait_for(lambda: len(cv.received_cmds) > initial_cmd_count)
    assert cv.received_cmds[-1].cmd == "enroll"
    assert cv.received_cmds[-1].args["name"] == "Marceline"
    assert res.get("ok") is True
    assert res.get("name") == "Marceline"


def test_e2e_acceptance_gate_all_pass() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/check_acceptance.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Acceptance checker failed:\n{proc.stdout}\n{proc.stderr}"
    assert "40/40 PASS" in proc.stdout
