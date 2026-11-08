"""LCD emotion map and face frame renderer for BMO."""

from dataclasses import dataclass
from typing import Any

from bmo_contracts.models import Emotion


@dataclass(frozen=True)
class EmotionFrameSpec:
    emotion: Emotion
    left_eye: str
    right_eye: str
    mouth_closed: str
    mouth_open: str
    bg_rgb: tuple[int, int, int]
    fps: int = 12


EMOTION_MAP: dict[Emotion, EmotionFrameSpec] = {
    Emotion.NEUTRAL: EmotionFrameSpec(
        emotion=Emotion.NEUTRAL,
        left_eye="•",
        right_eye="•",
        mouth_closed="-",
        mouth_open="o",
        bg_rgb=(30, 180, 150),
    ),
    Emotion.HAPPY: EmotionFrameSpec(
        emotion=Emotion.HAPPY,
        left_eye="^",
        right_eye="^",
        mouth_closed="‿",
        mouth_open="▽",
        bg_rgb=(50, 205, 50),
    ),
    Emotion.SAD: EmotionFrameSpec(
        emotion=Emotion.SAD,
        left_eye="T",
        right_eye="T",
        mouth_closed="︵",
        mouth_open="△",
        bg_rgb=(70, 130, 180),
    ),
    Emotion.THINKING: EmotionFrameSpec(
        emotion=Emotion.THINKING,
        left_eye="•",
        right_eye="¬",
        mouth_closed="~",
        mouth_open="▱",
        bg_rgb=(186, 85, 211),
    ),
    Emotion.SURPRISED: EmotionFrameSpec(
        emotion=Emotion.SURPRISED,
        left_eye="O",
        right_eye="O",
        mouth_closed="o",
        mouth_open="O",
        bg_rgb=(255, 165, 0),
    ),
    Emotion.SLEEPY: EmotionFrameSpec(
        emotion=Emotion.SLEEPY,
        left_eye="-",
        right_eye="-",
        mouth_closed="z",
        mouth_open="Z",
        bg_rgb=(75, 0, 130),
    ),
    Emotion.WINK: EmotionFrameSpec(
        emotion=Emotion.WINK,
        left_eye="^",
        right_eye="•",
        mouth_closed="‿",
        mouth_open="▽",
        bg_rgb=(0, 206, 209),
    ),
    Emotion.ERROR: EmotionFrameSpec(
        emotion=Emotion.ERROR,
        left_eye="X",
        right_eye="X",
        mouth_closed="_",
        mouth_open="□",
        bg_rgb=(220, 20, 60),
    ),
}


def get_emotion_spec(emotion: Emotion | str) -> EmotionFrameSpec:
    """Validate and return the emotion frame specification."""
    validated = Emotion(emotion)
    return EMOTION_MAP[validated]


def render_face_frame(emotion: Emotion | str, *, speaking: bool = False, step: int = 0) -> dict[str, Any]:
    """Render a single facial expression frame dictionary."""
    spec = get_emotion_spec(emotion)
    is_open = speaking and (step % 2 == 0)
    mouth = spec.mouth_open if is_open else spec.mouth_closed
    return {
        "emotion": spec.emotion.value,
        "left_eye": spec.left_eye,
        "right_eye": spec.right_eye,
        "mouth": mouth,
        "mouth_open": is_open,
        "bg_rgb": spec.bg_rgb,
        "ascii": f"{spec.left_eye} {mouth} {spec.right_eye}",
    }
