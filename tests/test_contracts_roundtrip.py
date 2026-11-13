"""Broker roundtrip contracts and peers test."""

from __future__ import annotations

import asyncio
import collections
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from paho.mqtt.client import CallbackAPIVersion, Client, MQTTMessage

from bmo_contracts.models import (
    BrainSay,
    BrainStatus,
    Emotion,
    FaceData,
    FaceIdentified,
    HwCmd,
    HwEvent,
    HwStatus,
    ObjectData,
    ObjectSeen,
    PersonData,
    PersonDetected,
    PersonLeft,
    PersonLeftData,
    Scene,
    ScenePerson,
    VisionCmd,
    VisionCmdResult,
    VisionStatus,
    parse_event_json,
)
from bmo_contracts.topics import (
    BRAIN_SAY,
    BRAIN_STATUS,
    HW_CMD,
    HW_EVENT,
    HW_STATUS,
    TOPICS,
    VISION_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
    VISION_STATUS,
)
from tools.mock_cv import MockCV, make_sample
from tools.mock_pi import MockPi
from tools.simple_broker import SimpleMQTTBroker, parse_passwords_file


def wait_for(condition: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture(scope="module")
def broker():
    users = parse_passwords_file("infra/passwords.example")
    b = SimpleMQTTBroker(
        host="127.0.0.1",
        port=18893,
        require_auth=True,
        users=users,
        acl_path="infra/acl",
    )
    loop = asyncio.new_event_loop()

    def run_broker() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(b.start())
        loop.run_forever()

    thread = threading.Thread(target=run_broker, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield b
    try:
        asyncio.run_coroutine_threadsafe(b.stop(), loop).result(timeout=2.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)


@pytest.fixture(scope="module")
def peers(broker: SimpleMQTTBroker):
    _ = broker
    cv = MockCV(host="127.0.0.1", port=18893)
    cv.start()

    pi = MockPi(host="127.0.0.1", port=18893)
    pi.start()

    brain = Client(CallbackAPIVersion.VERSION2)
    brain.username_pw_set("brain", "brain_insecure_dev_password")

    received_lock = threading.Lock()
    brain_received: dict[str, list[bytes]] = collections.defaultdict(list)

    def on_brain_msg(
        _client: Client,
        _userdata: object,
        msg: MQTTMessage,
    ) -> None:
        with received_lock:
            brain_received[msg.topic].append(msg.payload)

    brain.on_message = on_brain_msg
    brain.connect("127.0.0.1", 18893)
    brain.loop_start()

    brain.subscribe(
        [
            (VISION_STATUS, 0),
            (VISION_EVENTS, 0),
            (VISION_CONTEXT, 0),
            (VISION_CMD_RESULT, 0),
            (HW_STATUS, 0),
            (HW_EVENT, 0),
        ]
    )

    time.sleep(0.5)

    yield {
        "cv": cv,
        "pi": pi,
        "brain": brain,
        "brain_received": brain_received,
        "received_lock": received_lock,
    }

    brain.loop_stop()
    brain.disconnect()
    pi.stop()
    cv.stop()


def test_model_schema_presence() -> None:
    all_models = [
        BrainStatus,
        HwStatus,
        VisionStatus,
        FaceData,
        PersonData,
        PersonLeftData,
            ObjectData,
        ScenePerson,
        ]
    for model_cls in all_models:
        sample = make_sample(model_cls)
        assert sample is not None
        assert isinstance(sample, model_cls)


def test_all_10_topics_roundtrip(peers: dict[str, Any]) -> None:
    cv: MockCV = peers["cv"]
    pi: MockPi = peers["pi"]
    brain: Client = peers["brain"]
    received_lock: threading.Lock = peers["received_lock"]
    brain_received: dict[str, list[bytes]] = peers["brain_received"]

    assert len(TOPICS) == 10

    for topic, spec in TOPICS.items():
        if topic == VISION_EVENTS:
            variants = [
                FaceIdentified,
                PersonDetected,
                PersonLeft,
                            ObjectSeen,
            ]
            for variant in variants:
                inst = make_sample(variant)
                assert inst is not None
                initial_count = len(brain_received[topic])
                cv.publish_event(inst)
                assert wait_for(lambda t=topic, c=initial_count: len(brain_received[t]) > c)
                with received_lock:
                    raw = brain_received[topic][-1]
                parsed = spec.model.validate_json(raw)
                assert parsed == inst
                assert parse_event_json(raw.decode("utf-8")) == inst
        elif topic == VISION_CMD:
            cmd = make_sample(spec.model)
            initial_count = len(cv.received_cmds)
            brain.publish(topic, cmd.model_dump_json())
            assert wait_for(lambda c=initial_count: len(cv.received_cmds) > c)
            assert cv.received_cmds[-1] == cmd
        elif topic in (BRAIN_STATUS, BRAIN_SAY, HW_CMD):
            inst = make_sample(spec.model)
            if topic == BRAIN_STATUS:
                initial_count = len(pi.brain_statuses)
                brain.publish(topic, inst.model_dump_json())
                assert wait_for(lambda c=initial_count: len(pi.brain_statuses) > c)
                assert pi.brain_statuses[-1] == inst
            elif topic == BRAIN_SAY:
                initial_count = len(pi.say_messages)
                brain.publish(topic, inst.model_dump_json())
                assert wait_for(lambda c=initial_count: len(pi.say_messages) > c)
                assert pi.say_messages[-1] == inst
            elif topic == HW_CMD:
                initial_count = len(pi.hw_commands)
                brain.publish(topic, inst.model_dump_json())
                assert wait_for(lambda c=initial_count: len(pi.hw_commands) > c)
                assert pi.hw_commands[-1] == inst
        else:
            inst = make_sample(spec.model)
            initial_count = len(brain_received[topic])
            if topic in (VISION_STATUS, VISION_CONTEXT, VISION_CMD_RESULT):
                cv.client.publish(topic, inst.model_dump_json())
            else:
                pi.client.publish(topic, inst.model_dump_json())
            assert wait_for(lambda t=topic, c=initial_count: len(brain_received[t]) > c)
            with received_lock:
                raw = brain_received[topic][-1]
            if hasattr(spec.model, "validate_json"):
                parsed = spec.model.validate_json(raw)
            else:
                parsed = spec.model.model_validate_json(raw)
            assert parsed == inst


def test_mock_cv_command_roundtrip(peers: dict[str, Any]) -> None:
    cv: MockCV = peers["cv"]
    brain: Client = peers["brain"]
    received_lock: threading.Lock = peers["received_lock"]
    brain_received: dict[str, list[bytes]] = peers["brain_received"]

    cmd = VisionCmd(id="cmd-roundtrip-1", cmd="enroll")
    initial_cmds = len(cv.received_cmds)
    initial_results = len(brain_received[VISION_CMD_RESULT])

    brain.publish(VISION_CMD, cmd.model_dump_json())

    assert wait_for(lambda: len(cv.received_cmds) > initial_cmds)
    assert cv.received_cmds[-1].id == "cmd-roundtrip-1"
    assert cv.received_cmds[-1].cmd == "enroll"

    assert wait_for(lambda: len(brain_received[VISION_CMD_RESULT]) > initial_results)
    with received_lock:
        raw = brain_received[VISION_CMD_RESULT][-1]
    res = VisionCmdResult.model_validate_json(raw)
    assert res.id == "cmd-roundtrip-1"
    assert res.ok is True
    assert res.detail == "ack:enroll"


def test_mock_pi_roundtrip(peers: dict[str, Any]) -> None:
    pi: MockPi = peers["pi"]
    brain: Client = peers["brain"]
    received_lock: threading.Lock = peers["received_lock"]
    brain_received: dict[str, list[bytes]] = peers["brain_received"]

    say = BrainSay(utt_id="utt-roundtrip-1", text="BMO chop!", emotion=Emotion.HAPPY)
    initial_says = len(pi.say_messages)
    brain.publish(BRAIN_SAY, say.model_dump_json())
    assert wait_for(lambda: len(pi.say_messages) > initial_says)
    assert pi.say_messages[-1].text == "BMO chop!"
    assert pi.say_messages[-1].utt_id == "utt-roundtrip-1"

    hw_cmd = make_sample(HwCmd)
    assert hw_cmd is not None
    initial_cmds = len(pi.hw_commands)
    brain.publish(HW_CMD, hw_cmd.model_dump_json())
    assert wait_for(lambda: len(pi.hw_commands) > initial_cmds)
    assert pi.hw_commands[-1] == hw_cmd

    hw_event = make_sample(HwEvent)
    assert hw_event is not None
    initial_events = len(brain_received[HW_EVENT])
    pi.publish_hw_event(hw_event)
    assert wait_for(lambda: len(brain_received[HW_EVENT]) > initial_events)
    with received_lock:
        raw = brain_received[HW_EVENT][-1]
    parsed = HwEvent.model_validate_json(raw)
    assert parsed == hw_event


def test_publish_scene_and_vision_status(peers: dict[str, Any]) -> None:
    cv: MockCV = peers["cv"]
    pi: MockPi = peers["pi"]
    received_lock: threading.Lock = peers["received_lock"]
    brain_received: dict[str, list[bytes]] = peers["brain_received"]

    scene = make_sample(Scene)
    assert scene is not None
    initial_scenes = len(brain_received[VISION_CONTEXT])
    cv.publish_scene(scene)
    assert wait_for(lambda: len(brain_received[VISION_CONTEXT]) > initial_scenes)
    with received_lock:
        raw = brain_received[VISION_CONTEXT][-1]
    parsed_scene = Scene.model_validate_json(raw)
    assert parsed_scene == scene

    assert len(pi.vision_statuses) > 0
    assert pi.vision_statuses[-1].online is True
