from __future__ import annotations

import logging
import sys
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
from brain.emotions import split_emotion_tag, strip_emotion_tag
from brain.memory_extract import apply_memory_effects
from brain.tools.registry import needs_tools
from brain.vision_intent import vision_target

logger = logging.getLogger(__name__)


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
        vision_request_fn: Callable[[str], None] | None = None,
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
        self._vision_request_fn = vision_request_fn
        self._clock = clock
        self.cancel_event = threading.Event()
        self._active_thread: threading.Thread | None = None
        # Single-flight gate: held for the whole turn body, so at most one
        # LLM decode/TTS turn runs at a time. Contenders are dropped, not queued.
        self._turn_gate = threading.Lock()

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
                    if (
                        self._active_thread is not None
                        and self._active_thread.is_alive()
                        and self._active_thread is not threading.current_thread()
                    ):
                        self._active_thread.join(timeout=0.25)
                    self.cancel_event.clear()
                    self._active_thread = threading.Thread(
                        target=self.listen_and_respond,
                        args=(source,),
                        daemon=True,
                    )
                    self._active_thread.start()
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

        try:
            pcm = self._vad.collect_utterance(source)
        except Exception as exc:
            logger.warning("VAD collect_utterance failed: %s", exc)
            if self._state_machine.state != BrainState.IDLE:
                self._state_machine.transition(BrainState.IDLE)
            return None

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

        print(f"\n>>> You said: {stt_res.text}", flush=True, file=sys.__stdout__)

        return self.run_text_turn(
            stt_res.text,
            vad_end_ts=t_vad_end,
            stt_ms=stt_res.latency_ms,
        )

    def _begin_turn(self) -> bool:
        """Claim the single-flight turn gate and reset per-turn state.

        Returns False (after logging) when another turn already holds the gate.
        """
        if not self._turn_gate.acquire(blocking=False):
            logger.info("turn busy, dropping")
            return False
        try:
            self.cancel_event.clear()
            self._state_machine.transition(BrainState.THINKING)
        except BaseException:
            self._turn_gate.release()
            raise
        return True

    def _end_turn(self) -> None:
        self._turn_gate.release()

    def _text_turn_worker(self, text: str) -> None:
        try:
            self._run_turn(text)
        finally:
            self._end_turn()

    def start_text_turn(self, text: str) -> str | None:
        """Start a text-only turn on a new thread (for proactive handlers).

        Returns immediately; if a turn is already in flight the request is
        dropped rather than blocking the caller.
        """
        if not self._begin_turn():
            return None
        thread = threading.Thread(
            target=self._text_turn_worker,
            args=(text,),
            daemon=True,
        )
        self._active_thread = thread
        try:
            thread.start()
        except BaseException:
            self._active_thread = None
            self._end_turn()
            raise
        return None

    def run_text_turn(
        self,
        user_text: str,
        *,
        confirmed: bool = False,
        vad_end_ts: float | None = None,
        stt_ms: float = 0.0,
    ) -> str:
        if not self._begin_turn():
            return ""
        try:
            return self._run_turn(
                user_text,
                confirmed=confirmed,
                vad_end_ts=vad_end_ts,
                stt_ms=stt_ms,
            )
        finally:
            self._end_turn()

    def _run_turn(
        self,
        user_text: str,
        *,
        confirmed: bool = False,
        vad_end_ts: float | None = None,
        stt_ms: float = 0.0,
    ) -> str:
        memory = getattr(self._prompt_builder, "memory", None)
        if memory is not None:
            try:
                for note in apply_memory_effects(user_text, memory):
                    logger.info("memory: %s", note)
            except Exception as exc:
                logger.warning("memory update failed: %s", exc)

        vision_target_kind = vision_target(user_text)
        if self._vision_request_fn is not None and vision_target_kind:
            try:
                logger.info("Vision intent (%s); requesting fresh camera look", vision_target_kind)
                self._vision_request_fn(vision_target_kind)
            except Exception as exc:
                logger.warning("Vision request failed: %s", exc)

        messages = self._prompt_builder.build_messages(user_text, scene_target=vision_target_kind)

        t_turn_start = self._clock()
        t_llm_start = self._clock()
        llm_ttft_ms = 0.0
        tts_first_chunk_ms = 0.0
        first_chunk_seen = False
        first_sentence_emitted = False
        utt_id = uuid.uuid4().hex[:8]
        splitter = SentenceSplitter()
        reply_chunks: list[str] = []

        logger.info("LLM chat starting for user text: %r", user_text[:80])

        turn_emotion = Emotion.NEUTRAL

        def _speak_sentence(sentence: str) -> None:
            nonlocal first_sentence_emitted, tts_first_chunk_ms, turn_emotion
            clean = sentence.strip()
            if not clean or self.cancel_event.is_set():
                return
            parsed_emotion, clean = split_emotion_tag(clean)
            if parsed_emotion is not None:
                turn_emotion = parsed_emotion
            if not clean:
                return

            if not first_sentence_emitted:
                first_sentence_emitted = True
                print(f"bmo: {clean}", flush=True, file=sys.__stdout__)
                self._state_machine.transition(BrainState.SPEAKING)
                if self._publish_fn is not None:
                    say = BrainSay(
                        utt_id=utt_id,
                        text=clean,
                        emotion=turn_emotion,
                    )
                    self._publish_fn(
                        BRAIN_SAY,
                        say.model_dump_json().encode("utf-8"),
                        1,
                        False,
                    )
                t_tts0 = self._clock()
                try:
                    self._tts.speak_sentence(clean, self._sink, cancel_event=self.cancel_event)
                except Exception as exc:
                    logger.error("TTS speak_sentence failed: %s", exc)
                    return
                tts_first_chunk_ms = (self._clock() - t_tts0) * 1000.0
            else:
                print(f"     {clean}", flush=True, file=sys.__stdout__)
                if self._publish_fn is not None:
                    say = BrainSay(
                        utt_id=utt_id,
                        text=clean,
                        emotion=turn_emotion,
                    )
                    self._publish_fn(
                        BRAIN_SAY,
                        say.model_dump_json().encode("utf-8"),
                        1,
                        False,
                    )
                try:
                    self._tts.speak_sentence(clean, self._sink, cancel_event=self.cancel_event)
                except Exception as exc:
                    logger.error("TTS speak_sentence failed: %s", exc)
                    return

            # Wait for audio to finish playing
            if hasattr(self._sink, 'drain'):
                try:
                    self._sink.drain()
                except Exception as exc:
                    logger.error("Sink drain failed: %s", exc)

        # The single-flight gate guarantees only one llama-cpp decode at a time.
        t_chat0 = self._clock()
        try:
            tool_specs = self._tools.ollama_tools() if needs_tools(user_text) else None
            for chunk in self._llm.chat(
                messages,
                tools=tool_specs,
                stream=True,
                cancel_event=self.cancel_event,
            ):
                if self.cancel_event.is_set():
                    break
                if not first_chunk_seen:
                    first_chunk_seen = True
                    llm_ttft_ms = (self._clock() - t_llm_start) * 1000.0
                    logger.info("First LLM token after %.1fms", llm_ttft_ms)

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
            t_chat1 = self._clock()
            logger.info("LLM chat completed in %.1fs, %d chunks", t_chat1 - t_chat0, len(reply_chunks))
        except Exception as exc:
            logger.error("LLM chat failed: %s", exc, exc_info=True)
            reply_chunks.append(f"[LLM error: {exc}]")

        if not self.cancel_event.is_set():
            flushed = splitter.flush()
            if flushed is not None:
                flushed_sentences = [flushed] if isinstance(flushed, str) else list(flushed)
                for s in flushed_sentences:
                    _speak_sentence(s)
                    if self.cancel_event.is_set():
                        break

        full_reply = strip_emotion_tag("".join(reply_chunks))
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
