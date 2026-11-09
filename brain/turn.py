from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable

import numpy as np

from bmo_contracts.models import BrainSay, Emotion
from bmo_contracts.topics import BRAIN_SAY
from brain.audio.io import AudioSink, AudioSource
from brain.audio.stt import SpeechToText
from brain.audio.tts import TextToSpeech
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.llm import LLMClient
from brain.prompt import PromptBuilder
from brain.state import BrainState, BrainStateMachine
from brain.stream import SentenceSplitter
from brain.tools.registry import ToolRegistry


class TurnManager:
    """Coordinates voice interaction, streaming, echo handling, and barge-in."""

    def __init__(
        self,
        state_machine: BrainStateMachine,
        wake: WakeWordDetector,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        tools: ToolRegistry,
        tts: TextToSpeech,
        sink: AudioSink,
        publish_fn: Callable[[str, bytes, int, bool], None] | None = None,
        on_turn_metrics: Callable[[dict[str, float]], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._state_machine = state_machine
        self._wake = wake
        self._vad = vad
        self._stt = stt
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._tools = tools
        self._tts = tts
        self._sink = sink
        self._publish_fn = publish_fn
        self._on_turn_metrics = on_turn_metrics
        self._clock = clock
        self.cancel_event = threading.Event()
        self._active_thread: threading.Thread | None = None

    def process_audio_frame(self, frame: np.ndarray, source: AudioSource | None = None) -> str | None:
        if self._state_machine.state == BrainState.SPEAKING:
            if self._wake.process_frame(frame):
                self.barge_in()
                return "barge_in"
            return None

        if self._state_machine.state == BrainState.IDLE:
            if self._wake.process_frame(frame):
                self._state_machine.transition(BrainState.LISTENING)
                if source is not None:
                    return self.listen_and_respond(source)
                return "listening"

        return None

    def barge_in(self) -> float:
        t0 = self._clock()
        self.cancel_event.set()
        self._tts.cancel()
        self._sink.stop()
        if (
            self._active_thread is not None
            and self._active_thread.is_alive()
            and self._active_thread is not threading.current_thread()
        ):
            self._active_thread.join(timeout=0.25)
        self._active_thread = None
        if self._state_machine.state in (
            BrainState.SPEAKING,
            BrainState.THINKING,
            BrainState.IDLE,
        ):
            self._state_machine.transition(BrainState.LISTENING)
        return (self._clock() - t0) * 1000.0

    def listen_and_respond(self, source: AudioSource) -> str | None:
        if self._state_machine.state == BrainState.IDLE:
            self._state_machine.transition(BrainState.LISTENING)

        pcm = self._vad.collect_utterance(source)
        t_vad_end = self._clock()
        if pcm is None:
            if self._state_machine.state != BrainState.IDLE:
                self._state_machine.transition(BrainState.IDLE)
            return None

        stt_res = self._stt.transcribe(pcm)
        if stt_res is None or not stt_res.text.strip():
            if self._state_machine.state != BrainState.IDLE:
                self._state_machine.transition(BrainState.IDLE)
            return None

        return self.run_text_turn(
            stt_res.text,
            vad_end_ts=t_vad_end,
            stt_ms=stt_res.latency_ms,
        )

    def run_text_turn(
        self,
        user_text: str,
        *,
        confirmed: bool = False,
        vad_end_ts: float | None = None,
        stt_ms: float = 0.0,
    ) -> str:
        self.cancel_event.clear()
        self._state_machine.transition(BrainState.THINKING)
        messages = self._prompt_builder.build_messages(user_text)

        t_turn_start = self._clock()
        t_llm_start = self._clock()
        llm_ttft_ms = 0.0
        tts_first_chunk_ms = 0.0
        first_chunk_seen = False
        first_sentence_emitted = False
        utt_id = uuid.uuid4().hex[:8]
        splitter = SentenceSplitter()
        reply_chunks: list[str] = []

        def _speak_sentence(sentence: str) -> None:
            nonlocal first_sentence_emitted, tts_first_chunk_ms
            clean = sentence.strip()
            if not clean or self.cancel_event.is_set():
                return
            if not first_sentence_emitted:
                first_sentence_emitted = True
                self._state_machine.transition(BrainState.SPEAKING)
                if self._publish_fn is not None:
                    say = BrainSay(
                        utt_id=utt_id,
                        text=clean,
                        emotion=Emotion.HAPPY,
                    )
                    self._publish_fn(
                        BRAIN_SAY,
                        say.model_dump_json().encode("utf-8"),
                        1,
                        False,
                    )
                t_tts0 = self._clock()
                self._tts.speak_sentence(clean, self._sink, cancel_event=self.cancel_event)
                tts_first_chunk_ms = (self._clock() - t_tts0) * 1000.0
            else:
                if self._publish_fn is not None:
                    say = BrainSay(
                        utt_id=utt_id,
                        text=clean,
                        emotion=Emotion.HAPPY,
                    )
                    self._publish_fn(
                        BRAIN_SAY,
                        say.model_dump_json().encode("utf-8"),
                        1,
                        False,
                    )
                self._tts.speak_sentence(clean, self._sink, cancel_event=self.cancel_event)

        for chunk in self._llm.chat(
            messages,
            tools=self._tools.ollama_tools(),
            stream=True,
            cancel_event=self.cancel_event,
        ):
            if self.cancel_event.is_set():
                break
            if not first_chunk_seen:
                first_chunk_seen = True
                llm_ttft_ms = (self._clock() - t_llm_start) * 1000.0

            if chunk.tool_call is not None:
                self._tools.dispatch(chunk.tool_call, confirmed=confirmed)

            if chunk.text:
                reply_chunks.append(chunk.text)
                fed = splitter.feed(chunk.text)
                if fed is not None:
                    sentences = [fed] if isinstance(fed, str) else list(fed)
                    for s in sentences:
                        _speak_sentence(s)
                        if self.cancel_event.is_set():
                            break

        if not self.cancel_event.is_set():
            flushed = splitter.flush()
            if flushed is not None:
                flushed_sentences = [flushed] if isinstance(flushed, str) else list(flushed)
                for s in flushed_sentences:
                    _speak_sentence(s)
                    if self.cancel_event.is_set():
                        break

        full_reply = "".join(reply_chunks)
        self._prompt_builder.record_turn(user_text, full_reply)

        if not self.cancel_event.is_set() and self._state_machine.state in (
            BrainState.SPEAKING,
            BrainState.THINKING,
        ):
            self._state_machine.transition(BrainState.IDLE)

        if self._on_turn_metrics is not None:
            metrics: dict[str, float] = {
                "llm_ttft_ms": llm_ttft_ms,
                "tts_first_chunk_ms": tts_first_chunk_ms,
                "stt_ms": stt_ms,
                "turn_ms": (self._clock() - t_turn_start) * 1000.0,
            }
            if vad_end_ts is not None:
                metrics["e2e_voice_ms"] = (self._clock() - vad_end_ts) * 1000.0
            self._on_turn_metrics(metrics)

        return full_reply
