from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from bmo_contracts.models import BrainSay, Emotion
from bmo_contracts.topics import BRAIN_SAY
from brain.audio.io import AudioSink, AudioSource, MemorySink, MemorySource
from brain.audio.stt import STTResult, SpeechToText
from brain.audio.tts import TextToSpeech
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.llm import ChatChunk, LLMClient
from brain.prompt import PromptBuilder
from brain.state import BrainState, BrainStateMachine
from brain.tools.registry import ToolRegistry
from brain.turn import TurnManager


def _noop_publish(_topic: str, _payload: bytes, _qos: int = 1, _retain: bool = False) -> None:
    pass


class DummyWake(WakeWordDetector):
    def __init__(self, score: float = 0.0) -> None:
        self.score = score

    def score_frame(self, frame: np.ndarray) -> float:
        _ = frame
        return self.score

    def process_frame(self, frame: np.ndarray) -> bool:
        return self.score_frame(frame) >= 0.5


class DummyVAD(VoiceActivityDetector):
    def __init__(self, utterance: bytes | None = None) -> None:
        self.utterance = utterance

    def is_speech(self, frame: np.ndarray) -> bool:
        _ = frame
        return True

    def collect_utterance(self, source: AudioSource) -> bytes | None:
        _ = source
        return self.utterance


class DummySTT(SpeechToText):
    def __init__(self, text: str = "Hello BMO") -> None:
        self.text = text

    def transcribe(self, pcm: bytes) -> STTResult | None:
        _ = pcm
        return STTResult(text=self.text, latency_ms=12.5, confidence=0.98)


class DummyTTS(TextToSpeech):
    def __init__(self, chunk_delay: float = 0.0) -> None:
        self.chunk_delay = chunk_delay
        self.cancelled = False
        self.spoken: list[str] = []

    def speak_sentence(
        self,
        text: str,
        sink: AudioSink,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.spoken.append(text)
        if self.chunk_delay > 0:
            t_end = time.perf_counter() + self.chunk_delay
            while time.perf_counter() < t_end:
                if cancel_event is not None and cancel_event.is_set():
                    return
                time.sleep(0.005)
        if cancel_event is not None and cancel_event.is_set():
            return
        for m in ("play", "write_chunk", "play_chunk", "write_frame", "play_frame", "push", "write"):
            if hasattr(sink, m):
                try:
                    getattr(sink, m)(b"\x00" * 320)
                    break
                except Exception:
                    pass

    def cancel(self) -> None:
        self.cancelled = True


class DummyLLM(LLMClient):
    def __init__(self, text: str = "Hi there friend!") -> None:
        self._text = text

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        cancel_event: threading.Event | None = None,
    ):
        _ = (messages, tools, stream, cancel_event)
        if self._text:
            yield ChatChunk(text=self._text)


class SlowLLMClient(LLMClient):
    def __init__(self, chunks: list[str], chunk_delay: float = 0.02) -> None:
        self._chunks = chunks
        self._delay = chunk_delay

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        cancel_event: threading.Event | None = None,
    ):
        _ = (messages, tools, stream)
        for c in self._chunks:
            if cancel_event is not None and cancel_event.is_set():
                break
            time.sleep(self._delay)
            if cancel_event is not None and cancel_event.is_set():
                break
            yield ChatChunk(text=c)


class DummyPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        pass

    def build_messages(self, user_text: str, scene_target=None) -> list[dict[str, Any]]:
        _ = scene_target
        return [{"role": "user", "content": user_text}]

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        _ = (user_text, assistant_text)


def test_full_voice_turn() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake(score=0.95)
    vad = DummyVAD(utterance=b"\x00" * 1600)
    stt = DummySTT(text="Hello BMO")
    llm = DummyLLM(text="[happy] Hi there friend!")
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS()
    sink = MemorySink()

    published: list[tuple[str, bytes, int, bool]] = []

    def pub_fn(topic: str, payload: bytes, qos: int, retain: bool) -> None:
        published.append((topic, payload, qos, retain))

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
        publish_fn=pub_fn,
    )

    frame = np.zeros(160, dtype=np.int16)
    source = MemorySource(pcm=b"\x00" * 320)

    res = turn.process_audio_frame(frame, source=source)
    assert res == "listening"
    assert turn._active_thread is not None
    turn._active_thread.join(timeout=2.0)
    assert sm.state == BrainState.IDLE
    assert len(published) >= 1
    assert published[0][0] == BRAIN_SAY
    say = BrainSay.model_validate_json(published[0][1].decode("utf-8"))
    assert say.emotion == Emotion.HAPPY
    assert say.text == "Hi there friend!"
    assert "Hi there friend!" in say.text
    assert tts.spoken == ["Hi there friend!"]


def test_echo_handling() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    sm.transition(BrainState.LISTENING)
    sm.transition(BrainState.THINKING)
    sm.transition(BrainState.SPEAKING)
    assert sm.state == BrainState.SPEAKING

    wake = DummyWake(score=0.1)
    vad = DummyVAD()
    stt = DummySTT()
    llm = DummyLLM(text="Test reply")
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS()
    sink = MemorySink()

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
    )

    loud_frame = np.full(160, 6000, dtype=np.int16)
    assert vad.is_speech(loud_frame) is True
    assert wake.score_frame(loud_frame) == 0.1
    assert wake.score_frame(loud_frame) < 0.5

    for _ in range(50):
        res = turn.process_audio_frame(loud_frame)
        assert res is None
        assert sm.state == BrainState.SPEAKING


def test_barge_in_stops_streaming_and_transitions_to_listening() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake(score=0.95)
    vad = DummyVAD(utterance=b"\x00" * 320)
    stt = DummySTT(text="Tell me a long story")
    chunks = [f"Word{i}. " for i in range(50)]
    llm = SlowLLMClient(chunks=chunks, chunk_delay=0.02)
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS(chunk_delay=0.02)
    sink = MemorySink()

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
    )

    frame = np.zeros(160, dtype=np.int16)
    source = MemorySource(pcm=b"\x00" * 320)

    # First frame in IDLE triggers wake word and starts the voice turn off the audio thread
    res0 = turn.process_audio_frame(frame, source=source)
    assert res0 == "listening"
    t = turn._active_thread
    assert t is not None
    assert t.is_alive()

    # Audio loop continues feeding frames until barge_in occurs during SPEAKING
    barge_res = None
    barge_dur_ms = 0.0
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        res = turn.process_audio_frame(frame, source=source)
        if res == "barge_in":
            barge_dur_ms = (time.perf_counter() - t0) * 1000.0
            barge_res = res
            break
        time.sleep(0.005)

    assert barge_res == "barge_in"
    assert barge_dur_ms < 300.0
    assert turn.cancel_event.is_set() is True
    assert sink.stopped is True
    assert turn._active_thread is None
    assert not t.is_alive()
    assert sm.state == BrainState.LISTENING


def test_listen_and_respond_empty_stt() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake()
    vad = DummyVAD(utterance=b"\x00" * 320)
    stt = DummySTT(text="")
    llm = DummyLLM(text="")
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS()
    sink = MemorySink()

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
    )

    source = MemorySource(pcm=b"\x00" * 320)
    res = turn.listen_and_respond(source)
    assert res is None
    assert sm.state == BrainState.IDLE


def test_barge_in_during_proactive_turn() -> None:
    """Test that barge_in works during a proactive text turn (no audio source)."""
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake(score=0.95)
    vad = DummyVAD(utterance=b"\x00" * 320)
    stt = DummySTT(text="Interrupted!")
    chunks = [f"Proactive word {i}. " for i in range(50)]
    llm = SlowLLMClient(chunks=chunks, chunk_delay=0.02)
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS(chunk_delay=0.02)
    sink = MemorySink()

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
    )

    # Start proactive turn from IDLE (simulating _on_proactive_greet)
    assert sm.state == BrainState.IDLE
    turn.start_text_turn("Hello there! This is a long proactive greeting that should be interruptible.")
    
    t = turn._active_thread
    assert t is not None
    assert t.is_alive()

    # Wait for turn to reach SPEAKING state
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and sm.state != BrainState.SPEAKING:
        time.sleep(0.01)
    
    assert sm.state == BrainState.SPEAKING, f"Expected SPEAKING, got {sm.state}"

    # Now simulate wake word during SPEAKING - should barge in
    frame = np.zeros(160, dtype=np.int16)
    barge_res = None
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        res = turn.process_audio_frame(frame)
        if res == "barge_in":
            barge_res = res
            break
        time.sleep(0.005)

    assert barge_res == "barge_in", "Barge-in should have triggered during proactive turn"
    assert turn.cancel_event.is_set() is True
    assert sink.stopped is True
    assert turn._active_thread is None
    assert not t.is_alive()
    assert sm.state == BrainState.LISTENING
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake()
    vad = DummyVAD()
    stt = DummySTT()
    llm = DummyLLM(text="Done!")
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS()
    sink = MemorySink()

    metrics_captured: list[dict[str, float]] = []

    def on_metrics(m: dict[str, float]) -> None:
        metrics_captured.append(m)

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
        on_turn_metrics=on_metrics,
    )

    reply = turn.run_text_turn("Hi", vad_end_ts=time.perf_counter(), stt_ms=15.0)
    assert reply == "Done!"
    assert len(metrics_captured) == 1
    assert "llm_ttft_ms" in metrics_captured[0]
    assert "tts_first_chunk_ms" in metrics_captured[0]
    assert "e2e_voice_ms" in metrics_captured[0]


def test_barge_in_during_thinking_succeeds_and_releases_gate() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake(score=0.95)
    vad = DummyVAD()
    stt = DummySTT()

    chat_started = threading.Event()

    class BlockingLLMClient(LLMClient):
        def __init__(self) -> None:
            pass

        def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            stream: bool = False,
            cancel_event: threading.Event | None = None,
        ):
            _ = (messages, tools, stream)
            chat_started.set()
            if cancel_event is not None:
                cancel_event.wait(timeout=2.0)
            yield ChatChunk(text="Finished")

    llm = BlockingLLMClient()
    builder = DummyPromptBuilder()
    tools = ToolRegistry()
    tts = DummyTTS()
    sink = MemorySink()

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=builder,
        tools=tools,
        tts=tts,
        sink=sink,
    )

    turn.start_text_turn("Slow prompt")
    assert chat_started.wait(timeout=1.0)
    assert sm.state == BrainState.THINKING
    assert turn._turn_gate.locked()

    # Barge-in during THINKING
    frame = np.zeros(160, dtype=np.int16)
    res = turn.process_audio_frame(frame)
    assert res == "barge_in"
    assert sm.state == BrainState.LISTENING
    assert not turn._turn_gate.locked()

    # Verify a subsequent turn can immediately acquire gate
    next_llm = DummyLLM(text="Second turn")
    turn._llm = next_llm
    reply = turn.run_text_turn("New turn after barge-in")
    assert reply == "Second turn"


def test_cancelled_turn_recovers_to_idle_when_not_barged() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake()
    vad = DummyVAD()
    stt = DummySTT()

    chat_started = threading.Event()

    class BlockingLLMClient(LLMClient):
        def __init__(self) -> None:
            pass

        def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            stream: bool = False,
            cancel_event: threading.Event | None = None,
        ):
            _ = (messages, tools, stream)
            chat_started.set()
            if cancel_event is not None:
                cancel_event.wait(timeout=2.0)
            yield ChatChunk(text="Finished")

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=BlockingLLMClient(),
        prompt_builder=DummyPromptBuilder(),
        tools=ToolRegistry(),
        tts=DummyTTS(),
        sink=MemorySink(),
    )

    turn.start_text_turn("Cancel test")
    assert chat_started.wait(timeout=1.0)
    assert sm.state == BrainState.THINKING
    assert turn._turn_gate.locked()

    # Cancel directly without barge_in()
    turn.cancel_event.set()
    assert turn._active_thread is not None
    turn._active_thread.join(timeout=1.0)

    # State must recover to IDLE (not stuck in THINKING or set to LISTENING)
    assert sm.state == BrainState.IDLE
    assert not turn._turn_gate.locked()


def test_old_slow_worker_stays_cancelled_when_new_turn_starts() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = DummyWake(score=0.95)
    vad = DummyVAD()
    stt = DummySTT()

    first_started = threading.Event()
    first_worker_cancel_token: threading.Event | None = None
    first_worker_resumed_after_new_turn = False

    class SlowFirstLLMClient(LLMClient):
        def __init__(self) -> None:
            pass

        def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            stream: bool = False,
            cancel_event: threading.Event | None = None,
        ):
            nonlocal first_worker_cancel_token, first_worker_resumed_after_new_turn
            _ = (messages, tools, stream)
            first_worker_cancel_token = cancel_event
            first_started.set()
            if cancel_event is not None:
                cancel_event.wait(timeout=2.0)
            time.sleep(0.05)
            if cancel_event is not None and not cancel_event.is_set():
                first_worker_resumed_after_new_turn = True
            yield ChatChunk(text="Lingering text")

    turn = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=SlowFirstLLMClient(),
        prompt_builder=DummyPromptBuilder(),
        tools=ToolRegistry(),
        tts=DummyTTS(),
        sink=MemorySink(),
    )

    turn.start_text_turn("First turn")
    assert first_started.wait(timeout=1.0)
    first_thread = turn._active_thread
    assert first_thread is not None

    # Cancel worker 1 via barge_in
    turn.barge_in()
    assert first_worker_cancel_token is not None
    assert first_worker_cancel_token.is_set()

    # Now start Turn 2 with a fast LLM
    turn._llm = DummyLLM(text="Second turn reply")
    turn.start_text_turn("Second turn")

    # Turn 2's cancel event must be a fresh event, distinct from worker 1's token
    assert turn.cancel_event is not first_worker_cancel_token
    assert not turn.cancel_event.is_set()

    # Worker 1's token must stay set permanently
    assert first_worker_cancel_token.is_set()

    first_thread.join(timeout=1.0)
    assert not first_worker_resumed_after_new_turn
