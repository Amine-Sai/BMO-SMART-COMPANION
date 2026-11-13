from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from bmo_contracts.models import Emotion
from face_ui.assets import (
    DEFAULT_FILE,
    EMOTION_FILE,
    MODE_FILE,
    REQUIRED_FILES,
    SLEEPING_FILE,
    TALKING_FILE,
    FaceArt,
    load_art,
)
from face_ui.face import (
    EMOTION_PRESETS,
    FaceParams,
    lerp_params,
    render_face,
    target_params,
)
from face_ui.main import FaceApp, build_parser, speaking_mouth_open
from face_ui.signal import FaceSignal, FaceSignalReader, speech_mouth
from face_ui.state import FaceMode, FaceState, mode_from_status

ART_DIR = Path(__file__).resolve().parent.parent / "BMO Face"
ART_AVAILABLE = ART_DIR.is_dir() and shutil.which("rsvg-convert") is not None
requires_art = pytest.mark.skipif(
    not ART_AVAILABLE, reason="BMO SVG artwork or rsvg-convert unavailable"
)


def test_mode_from_status():
    assert mode_from_status("idle") == FaceMode.AWAKE
    assert mode_from_status("thinking") == FaceMode.THINKING
    assert mode_from_status("speaking") == FaceMode.SPEAKING
    assert mode_from_status("error") == FaceMode.ERROR
    assert mode_from_status("idle", online=False) == FaceMode.OFFLINE
    assert mode_from_status("something-else") == FaceMode.AWAKE


def test_face_state_sleeps_and_wakes():
    state = FaceState(sleep_after_s=60.0, last_activity=0.0)
    state.on_status("thinking", True, now=1.0)
    assert state.mode == FaceMode.THINKING
    state.on_status("idle", True, now=2.0)
    assert state.mode == FaceMode.AWAKE

    state.tick(now=61.5)
    assert state.mode == FaceMode.AWAKE
    state.tick(now=62.1)
    assert state.mode == FaceMode.SLEEP

    # An idle status must not wake the face; a spoken line must.
    state.on_status("idle", True, now=63.0)
    assert state.mode == FaceMode.SLEEP
    state.on_say(Emotion.HAPPY, "Hello!", now=64.0)
    assert state.mode == FaceMode.SPEAKING
    assert state.emotion == Emotion.HAPPY


def test_say_forces_speaking_even_from_awake():
    state = FaceState()
    state.on_status("idle", True, now=1.0)
    assert state.mode == FaceMode.AWAKE
    state.on_say(Emotion.HAPPY, "Hi there!", now=2.0)
    assert state.mode == FaceMode.SPEAKING


def test_face_state_idle_resets_sleep_timer():
    state = FaceState(sleep_after_s=60.0, last_activity=0.0)
    state.on_status("idle", True, now=100.0)
    state.tick(now=130.0)
    assert state.mode == FaceMode.AWAKE
    state.tick(now=160.1)
    assert state.mode == FaceMode.SLEEP


def test_face_state_offline_and_error_override():
    state = FaceState()
    state.on_status("offline", False, now=1.0)
    assert state.mode == FaceMode.OFFLINE
    state.on_status("error", True, now=2.0)
    assert state.mode == FaceMode.ERROR


def test_target_params_for_sleep_and_speaking():
    asleep = target_params(FaceMode.SLEEP, Emotion.NEUTRAL, 0.0, -100.0, 0.0)
    assert asleep.eye_style == "closed"
    assert asleep.eye_open < 0.1

    speaking = target_params(FaceMode.SPEAKING, Emotion.HAPPY, 0.0, 0.0, 0.5)
    assert speaking.mouth_open > 0.0
    assert speaking.eye_style == "happy"


def test_lerp_params_endpoints():
    current = FaceParams()
    target = EMOTION_PRESETS[Emotion.SURPRISED]
    same = lerp_params(current, target, 0.0)
    assert same.eye_open == pytest.approx(current.eye_open)
    done = lerp_params(current, target, 1.0)
    assert done.eye_open == pytest.approx(target.eye_open)
    assert done.mouth_open == pytest.approx(target.mouth_open)


@pytest.mark.parametrize("emotion", list(Emotion))
def test_render_face_emotions(emotion):
    frame = render_face(EMOTION_PRESETS[emotion], FaceMode.AWAKE, 0.0)
    assert frame.shape == (480, 800, 3)
    assert frame.dtype.name == "uint8"
    assert frame.any()


@pytest.mark.parametrize("mode", list(FaceMode))
def test_render_face_modes(mode):
    frame = render_face(FaceParams(), mode, 1.25)
    assert frame.shape == (480, 800, 3)
    assert frame.dtype.name == "uint8"
    assert frame.any()


def test_render_face_custom_size():
    frame = render_face(FaceParams(), FaceMode.AWAKE, 0.0, size=(320, 240))
    assert frame.shape == (240, 320, 3)


def test_signal_roundtrip(tmp_path):
    path = tmp_path / "face_state.json"
    writer = FaceSignal(path)
    writer.set_status("thinking")
    writer.set_say("hello", "happy")
    writer.set_speech([0.0, 0.8, 0.4], t0=100.0, rate=10.0)

    reader = FaceSignalReader(path)
    data = reader.read()
    assert data is not None
    assert data["mode"] == "speaking"
    assert data["emotion"] == "happy"
    assert data["speech"]["env"] == [0.0, 0.8, 0.4]
    assert speech_mouth(data, 100.05) == 0.0
    assert speech_mouth(data, 100.15) == 0.8
    assert speech_mouth(data, 100.25) == 0.4
    assert speech_mouth(data, 100.50) is None


def test_signal_reader_missing(tmp_path):
    assert FaceSignalReader(tmp_path / "nope.json").read() is None


def test_face_starts_asleep():
    assert MODE_FILE[FaceMode.BOOT] == SLEEPING_FILE
    app = FaceApp(build_parser().parse_args(["--no-art"]))
    assert app.state.mode == FaceMode.SLEEP


def test_face_app_uses_state_file(tmp_path):
    path = tmp_path / "face_state.json"
    writer = FaceSignal(path)
    writer.set_status("idle")
    app = FaceApp(build_parser().parse_args(["--state", str(path)]))
    app._apply_signal(100.0)
    assert app.state.mode == FaceMode.SLEEP

    writer.set_say("hi there", "wink")
    app._apply_signal(101.0)
    assert app.state.mode == FaceMode.SPEAKING
    assert app.state.emotion == Emotion.WINK
    assert app.state.last_say_text == "hi there"

    writer.set_status("idle")
    app._apply_signal(102.0)
    assert app.state.mode == FaceMode.AWAKE


def test_face_ignores_stale_speaking_state_on_boot(tmp_path):
    path = tmp_path / "face_state.json"
    writer = FaceSignal(path)
    writer.set_say("old line", "happy")
    old = time.time() - 3600
    os.utime(path, (old, old))
    app = FaceApp(build_parser().parse_args(["--state", str(path)]))
    app._apply_signal(100.0)
    assert app.state.mode == FaceMode.SLEEP

    writer.set_status("idle")
    app._apply_signal(101.0)
    assert app.state.mode == FaceMode.SLEEP

    writer.set_say("hello", "happy")
    app._apply_signal(102.0)
    assert app.state.mode == FaceMode.SPEAKING


def test_face_app_loop_reads_state_file(tmp_path, monkeypatch):
    path = tmp_path / "face_state.json"
    writer = FaceSignal(path)
    writer.set_status("idle")
    app = FaceApp(
        build_parser().parse_args(["--state", str(path), "--no-art"])
    )
    app._apply_signal(1.0)
    writer.set_say("hi there", "wink")
    app._start_mqtt = lambda: None
    frames = []
    monkeypatch.setattr(cv, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv, "imshow", lambda _title, frame: frames.append(frame))

    def stop_after_frame(_delay):
        app._stop.set()
        return -1

    monkeypatch.setattr(cv, "waitKey", stop_after_frame)
    monkeypatch.setattr(cv, "destroyAllWindows", lambda: None)
    assert app.run() == 0
    assert len(frames) == 1
    assert app.state.mode == FaceMode.SPEAKING
    assert app.state.emotion == Emotion.WINK


def test_speaking_mouth_toggles():
    values = {speaking_mouth_open(t) for t in np.linspace(0.0, 2.0, 60)}
    assert values == {True, False}


def test_cli_parser():
    args = build_parser().parse_args(["--size", "320x240", "--fullscreen"])
    assert args.size == (320, 240)
    assert args.fullscreen is True
    assert args.mqtt_port == 1883
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--size", "nope"])


@pytest.fixture(scope="module")
def art() -> FaceArt:
    loaded = load_art((640, 360))
    assert loaded is not None
    return loaded


@requires_art
def test_art_loads_required_frames(art):
    assert set(REQUIRED_FILES) <= set(art.frames)
    assert DEFAULT_FILE in art.frames
    assert TALKING_FILE in art.frames
    assert SLEEPING_FILE in art.frames


def test_every_emotion_and_mode_has_a_drawn_face():
    assert set(EMOTION_FILE) == set(Emotion)
    assert set(MODE_FILE) == {
        FaceMode.BOOT,
        FaceMode.SLEEP,
        FaceMode.MUTED,
        FaceMode.OFFLINE,
        FaceMode.ERROR,
        FaceMode.THINKING,
        FaceMode.LISTENING,
    }


@requires_art
@pytest.mark.parametrize("mode", list(FaceMode))
def test_art_render_modes(art, mode):
    frame = art.render(mode, Emotion.NEUTRAL, 1.25)
    assert frame.shape == (360, 640, 3)
    assert frame.dtype.name == "uint8"
    assert frame.any()


@requires_art
@pytest.mark.parametrize("emotion", list(Emotion))
def test_art_render_emotions(art, emotion):
    frame = art.render(FaceMode.AWAKE, emotion, 1.25)
    assert frame.shape == (360, 640, 3)
    assert np.array_equal(frame, art.frames[EMOTION_FILE[emotion]])


@requires_art
def test_art_render_custom_size():
    small = load_art((320, 240))
    assert small is not None
    assert small.render(FaceMode.AWAKE, Emotion.NEUTRAL, 0.0).shape == (240, 320, 3)


@requires_art
def test_art_speaking_alternates_default_and_talking(art):
    closed = art.render(FaceMode.SPEAKING, Emotion.NEUTRAL, 0.0)
    opened = art.render(
        FaceMode.SPEAKING, Emotion.NEUTRAL, 0.0, speaking_open=True
    )
    assert np.array_equal(closed, art.frames[DEFAULT_FILE])
    assert np.array_equal(opened, art.frames[TALKING_FILE])
    assert not np.array_equal(closed, opened)


@requires_art
def test_art_speaking_animates_over_time(art):
    frames = [
        art.render(
            FaceMode.SPEAKING,
            Emotion.NEUTRAL,
            t,
            speaking_open=speaking_mouth_open(t),
        )
        for t in np.linspace(0.0, 2.0, 80)
    ]
    assert len({frame.tobytes() for frame in frames}) > 1


@requires_art
def test_art_blink_uses_sleeping_face(art):
    opened = art.render(FaceMode.AWAKE, Emotion.NEUTRAL, 0.0)
    blinked = art.render(FaceMode.AWAKE, Emotion.NEUTRAL, 0.0, blinking=True)
    assert np.array_equal(blinked, art.frames[SLEEPING_FILE])
    assert not np.array_equal(opened, blinked)


@requires_art
def test_art_sleep_uses_sleeping_face(art):
    awake = art.render(FaceMode.AWAKE, Emotion.NEUTRAL, 0.0)
    asleep = art.render(FaceMode.SLEEP, Emotion.NEUTRAL, 0.0)
    assert MODE_FILE[FaceMode.SLEEP] == SLEEPING_FILE
    assert not np.array_equal(asleep, awake)
    assert art.frames[SLEEPING_FILE].any()
