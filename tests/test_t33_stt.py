import numpy as np
import pytest

from brain.audio.stt import SpeechToText
from brain.settings import AudioSettings


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(ref_words)][len(hyp_words)] / len(ref_words)


UTTERANCE_CASES = [
    ("hello world", 0.95, True),
    ("what time is it", 0.92, True),
    ("play my favorites playlist", 0.88, True),
    ("turn off the living room lights", 0.94, True),
    ("open front door", 0.85, True),
    ("how is the weather in seattle", 0.96, True),
    ("set a timer for five minutes", 0.90, True),
    ("good morning bmo", 0.97, True),
    ("stop all alarms immediately", 0.89, True),
    ("navigate back home", 0.86, True),
    ("read my latest unread messages", 0.93, True),
    ("increase master volume two steps", 0.82, True),
    ("unintelligible mumble", 0.20, False),
    ("inaudible faint whisper", 0.35, False),
    ("background metallic clatter", 0.15, False),
    ("unclear static hiss", 0.05, False),
    ("   ", 0.95, False),
    ("", 0.90, False),
    ("\t\n", 0.85, False),
    ("  \r\n  ", 0.98, False),
]


def test_silence_and_near_silence_rejected() -> None:
    settings = AudioSettings()
    stt = SpeechToText(
        settings=settings,
        transcribe_fn=lambda _pcm: ("detected words", 0.95),
        min_rms=150.0,
    )

    silence = np.zeros(16000, dtype=np.int16)
    assert stt.transcribe(silence) is None

    near_silent = np.full(16000, 50, dtype=np.int16)
    assert stt.transcribe(near_silent) is None

    assert stt.transcribe(np.array([], dtype=np.int16)) is None
    assert stt.transcribe(b"") is None


def test_transcribe_fn_none_result() -> None:
    settings = AudioSettings()
    stt = SpeechToText(
        settings=settings,
        transcribe_fn=lambda _pcm: None,
        min_rms=150.0,
    )
    active_pcm = np.full(16000, 1000, dtype=np.int16)
    assert stt.transcribe(active_pcm) is None


@pytest.mark.parametrize(
    ("text", "conf", "is_valid"),
    UTTERANCE_CASES,
)
def test_twenty_utterances_confidence_and_phrasing(
    text: str,
    conf: float,
    is_valid: bool,
) -> None:
    settings = AudioSettings()
    fake_clock_time = 100.0

    def fake_clock() -> float:
        nonlocal fake_clock_time
        current = fake_clock_time
        fake_clock_time += 0.015
        return current

    stt = SpeechToText(
        settings=settings,
        transcribe_fn=lambda _pcm: (text, conf),
        min_rms=150.0,
        max_no_speech_prob=0.6,
        clock=fake_clock,
    )

    active_pcm = np.full(16000, 1000, dtype=np.int16)
    result = stt.transcribe(active_pcm)

    if is_valid:
        assert result is not None
        assert result.text == text.strip()
        assert result.confidence == conf
        assert result.latency_ms > 0.0
        assert compute_wer(text, result.text) == 0.0
    else:
        assert result is None


def test_transcribe_bytes_input() -> None:
    settings = AudioSettings()
    stt = SpeechToText(
        settings=settings,
        transcribe_fn=lambda _pcm: ("bytes speech", 0.9),
        min_rms=150.0,
    )
    active_bytes = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = stt.transcribe(active_bytes)
    assert res is not None
    assert res.text == "bytes speech"
    assert res.confidence == 0.9


class FakeSegment:
    def __init__(self, text: str, no_speech_prob: float = 0.0) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob


class FakeWhisperModel:
    def __init__(self, segments: list[FakeSegment] | None = None) -> None:
        self.segments = segments if segments is not None else [FakeSegment("hello world", 0.05)]
        self.last_audio: np.ndarray | None = None
        self.last_kwargs: dict[str, object] = {}

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> tuple[list[FakeSegment], None]:
        self.last_audio = audio
        self.last_kwargs = kwargs
        return self.segments, None


def test_transcribe_passes_condition_on_previous_text_false() -> None:
    settings = AudioSettings()
    fake_model = FakeWhisperModel([FakeSegment("test transcription", no_speech_prob=0.1)])
    stt = SpeechToText(settings=settings, model=fake_model, min_rms=150.0)

    active_pcm = np.full(16000, 1000, dtype=np.int16)
    result = stt.transcribe(active_pcm)

    assert result is not None
    assert result.text == "test transcription"
    assert pytest.approx(result.confidence) == 0.9
    assert fake_model.last_kwargs.get("condition_on_previous_text") is False
    assert fake_model.last_kwargs.get("beam_size") == 1
    assert fake_model.last_kwargs.get("language") == "en"

