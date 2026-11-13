"""Deterministic replay suite for BMO Assistant brain (T6.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import unittest

from pydantic import BaseModel, ConfigDict

from bmo_contracts.models import (
    FaceIdentified,
    PersonLeft,
    Scene,
    VisionStatus,
)
from brain.audio.io import MemorySink, MemorySource
from brain.audio.stt import STTResult, SpeechToText
from brain.audio.tts import TextToSpeech
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.edge_monitor import EdgeMonitor
from brain.llm import ChatChunk, FakeLLMBackend, LLMClient, ToolCall
from brain.policy import ProactivePolicy
from brain.prompt import PromptBuilder
from brain.scene import SceneCache
from brain.settings import AudioSettings, load_settings
from brain.state import BrainState, BrainStateMachine
from brain.tools.registry import ToolRegistry, ToolSpec
from brain.turn import TurnManager

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "replay_session.jsonl"


class DummyWake(WakeWordDetector):
    def __init__(self) -> None:
        pass

    def process_frame(self, frame: Any) -> bool:
        _ = frame
        return False


class DummyVAD(VoiceActivityDetector):
    def __init__(self) -> None:
        pass

    def collect_utterance(self, source: Any) -> bytes | None:
        _ = source
        return None


class DummySTT(SpeechToText):
    def __init__(self) -> None:
        pass

    def transcribe(self, pcm: bytes) -> STTResult | None:
        _ = pcm
        return None


class DummyTTS(TextToSpeech):
    def __init__(self) -> None:
        super().__init__(settings=AudioSettings())
        self.spoken: list[str] = []

    def speak_sentence(self, text: str, sink: Any, cancel_event: Any = None) -> None:
        _ = (sink, cancel_event)
        self.spoken.append(text)


class MoveHeadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pan: float
    tilt: float = 0.0


class TestDeterministicReplaySuite(unittest.TestCase):
    """Replay recorded perception events, scenes, and utterances deterministically."""

    def setUp(self) -> None:
        self.current_time = 1000.0

        def fake_clock() -> float:
            return self.current_time

        self.fake_clock = fake_clock
        self.published: list[tuple[str, bytes, int, bool]] = []

        def publish_fn(topic: str, payload: bytes, qos: int = 1, retain: bool = False) -> None:
            self.published.append((topic, payload, qos, retain))

        self.publish_fn = publish_fn
        self.state_machine = BrainStateMachine(publish_fn=publish_fn)
        settings = load_settings()
        self.settings = settings
        self.scene_cache = SceneCache(settings=settings.scene, clock=fake_clock)
        self.prompt_builder = PromptBuilder(scene_cache=self.scene_cache)

        # Actuator tools
        self.tool_calls_dispatched: list[tuple[str, dict[str, Any]]] = []

        self.tools = ToolRegistry()
        self.tools.register(
            ToolSpec(
                name="move_head",
                description="Move BMO head",
                schema=MoveHeadArgs,
                handler=lambda args: {"result": self._record_tool("move_head", args.model_dump())},
            )
        )

        # Proactive policy
        self.greetings: list[str] = []
        self.policy = ProactivePolicy(
            settings=settings.policy,
            state_machine=self.state_machine,
            on_greet=lambda text: self.greetings.append(text),
            clock=fake_clock,
        )

        # Edge monitor
        self.edge_monitor = EdgeMonitor(
            state_machine=self.state_machine,
            fallback_source=MemorySource(pcm=b""),
            fallback_sink=MemorySink(),
            clock=fake_clock,
        )

        self.tts = DummyTTS()

    def _record_tool(self, name: str, args: dict[str, Any]) -> str:
        self.tool_calls_dispatched.append((name, args))
        return f"Dispatched {name}"

    def test_deterministic_session_replay(self) -> None:
        """Execute full replay session line-by-line and verify exact deterministic state."""
        self.assertTrue(FIXTURE_PATH.is_file(), f"Missing fixture at {FIXTURE_PATH}")

        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 6)

        # Step 1: vision_status
        step1 = lines[0]
        self.assertEqual(step1["type"], "vision_status")
        vstatus = VisionStatus.model_validate(step1["payload"])
        self.assertTrue(vstatus.online)
        self.scene_cache.update_status(vstatus)

        # Step 2: scene_context
        step2 = lines[1]
        self.assertEqual(step2["type"], "scene_context")
        self.scene_cache.update_scene(step2["payload"])
        scene = self.scene_cache.current()
        self.assertIsInstance(scene, Scene)
        self.assertEqual(len(scene.people), 1)
        self.assertEqual(scene.people[0].name, "Alice")
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.objects[0], "laptop")

        # Step 3: vision_event -> face_identified
        step3 = lines[2]
        self.assertEqual(step3["type"], "vision_event")
        evt3 = FaceIdentified.model_validate(step3["payload"])
        self.current_time = 1000.1
        greeting = self.policy.handle_event(evt3)
        self.assertIsNotNone(greeting)
        self.assertIn("Alice", str(greeting))
        self.assertEqual(len(self.greetings), 1)

        # Step 4: user_voice -> question about presence
        step4 = lines[3]
        self.assertEqual(step4["type"], "user_voice")
        expected_reply = step4["payload"]["expected_reply"]
        fake_llm_step4 = FakeLLMBackend(chunks=[ChatChunk(text=expected_reply)])
        turn4 = TurnManager(
            state_machine=self.state_machine,
            wake=DummyWake(),
            vad=DummyVAD(),
            stt=DummySTT(),
            llm=LLMClient(settings=self.settings.llm, backend=fake_llm_step4),
            prompt_builder=self.prompt_builder,
            tools=self.tools,
            tts=self.tts,
            sink=MemorySink(),
            publish_fn=self.publish_fn,
            clock=self.fake_clock,
        )
        reply4 = turn4.run_text_turn(step4["payload"]["text"])
        self.assertEqual(reply4, expected_reply)
        self.assertEqual(self.state_machine.state, BrainState.IDLE)

        # Step 5: user_voice -> tool invocation (move_head)
        step6 = lines[4]
        self.assertEqual(step6["type"], "user_voice")
        tool_call_item = ChatChunk(
            text="Looking to the left.",
            tool_call=ToolCall(
                name="move_head",
                arguments=step6["payload"]["expected_args"],
            ),
        )
        fake_llm_step6 = FakeLLMBackend(chunks=[tool_call_item])
        turn6 = TurnManager(
            state_machine=self.state_machine,
            wake=DummyWake(),
            vad=DummyVAD(),
            stt=DummySTT(),
            llm=LLMClient(settings=self.settings.llm, backend=fake_llm_step6),
            prompt_builder=self.prompt_builder,
            tools=self.tools,
            tts=self.tts,
            sink=MemorySink(),
            publish_fn=self.publish_fn,
            clock=self.fake_clock,
        )
        reply6 = turn6.run_text_turn(step6["payload"]["text"])
        self.assertEqual(reply6, "Looking to the left.")
        self.assertEqual(len(self.tool_calls_dispatched), 1)
        self.assertEqual(self.tool_calls_dispatched[0][0], "move_head")
        self.assertEqual(self.tool_calls_dispatched[0][1], {"pan": -30.0, "tilt": 0.0})

        # Step 7: vision_event -> person_left
        step7 = lines[5]
        self.assertEqual(step7["type"], "vision_event")
        evt7 = PersonLeft.model_validate(step7["payload"])
        self.current_time = 1010.0
        self.policy.handle_event(evt7)
        self.assertEqual(self.state_machine.state, BrainState.IDLE)
