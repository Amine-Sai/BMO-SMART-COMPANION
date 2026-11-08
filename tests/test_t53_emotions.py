"""Tests for T5.3 LCD emotion mapping and face frame rendering."""

import pytest
from bmo_contracts.models import Emotion
from brain.tools.actuator_tools import load_limits
from edge.emotions import EMOTION_MAP, EmotionFrameSpec, get_emotion_spec, render_face_frame


def test_all_emotions_mapped_and_limits_covered() -> None:
    limits = load_limits("config/limits.yaml")
    for emotion in Emotion:
        spec = get_emotion_spec(emotion)
        assert isinstance(spec, EmotionFrameSpec)
        assert spec.emotion == emotion
        assert spec.mouth_open != spec.mouth_closed
        assert spec.fps == 12
        assert len(spec.bg_rgb) == 3
        assert EMOTION_MAP[emotion] is spec

    for emotion_name in limits.emotions:
        spec_from_str = get_emotion_spec(emotion_name)
        assert isinstance(spec_from_str, EmotionFrameSpec)
        assert spec_from_str.emotion.value == emotion_name


def test_invalid_emotion_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_emotion_spec("not_a_real_emotion")


def test_render_face_frame_speaking_toggle() -> None:
    for emotion in Emotion:
        open_frame = render_face_frame(emotion, speaking=True, step=0)
        closed_frame = render_face_frame(emotion, speaking=True, step=1)

        assert open_frame["mouth_open"] is True
        assert closed_frame["mouth_open"] is False
        assert open_frame["mouth"] != closed_frame["mouth"]
        assert open_frame["ascii"] != closed_frame["ascii"]
        assert open_frame["emotion"] == emotion.value
        assert closed_frame["emotion"] == emotion.value
        assert open_frame["bg_rgb"] == closed_frame["bg_rgb"]

        quiet_frame_0 = render_face_frame(emotion, speaking=False, step=0)
        quiet_frame_1 = render_face_frame(emotion, speaking=False, step=1)
        assert quiet_frame_0["mouth_open"] is False
        assert quiet_frame_1["mouth_open"] is False
        assert quiet_frame_0["mouth"] == closed_frame["mouth"]
        assert quiet_frame_1["mouth"] == closed_frame["mouth"]
