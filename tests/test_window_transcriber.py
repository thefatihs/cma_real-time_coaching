from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from app.asr.models import TranscriptionResult, TranscriptionSegment
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.window_transcriber import WindowTranscriber


class FakeEngine:
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls: list[np.ndarray] = []

    def transcribe_audio(self, audio: np.ndarray) -> TranscriptionResult:
        self.calls.append(audio)
        return self.result


def make_window(**changes: object) -> ASRAudioWindow:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "first_sequence": 2,
        "last_sequence": 4,
        "start_seconds": 10.0,
        "end_seconds": 11.0,
        "duration_seconds": 1.0,
        "sample_rate_hz": 2,
        "channel_count": 1,
        "codec_name": "pcm_s16le",
        "pcm_bytes": np.array([0, 16384], dtype="<i2").tobytes(),
    }
    values.update(changes)
    return ASRAudioWindow.model_validate(values)


def make_result(
    segments: list[TranscriptionSegment] | None = None, *, text: str = "  hello  "
) -> TranscriptionResult:
    return TranscriptionResult(text, "en", 0.9, 1.0, 0.2, segments or [])


def test_valid_mono_window_transcription_and_single_engine_call() -> None:
    engine = FakeEngine(make_result([TranscriptionSegment(0.1, 0.8, " hi ")]))
    result = WindowTranscriber(engine).transcribe(make_window())
    assert result.text == "hello"
    assert result.segments[0].text == "hi"
    assert len(engine.calls) == 1
    np.testing.assert_allclose(engine.calls[0], [0.0, 0.5])


def test_stereo_metadata_is_preserved_and_audio_is_downmixed() -> None:
    pcm = np.array([32767, -32768, 16384, 16384], dtype="<i2").tobytes()
    engine = FakeEngine(make_result())
    result = WindowTranscriber(engine).transcribe(
        make_window(channel_count=2, pcm_bytes=pcm)
    )
    assert (
        result.tenant_id,
        result.call_id,
        result.first_sequence,
        result.last_sequence,
    ) == ("tenant_alpha", "call_001", 2, 4)
    assert engine.calls[0].shape == (2,)


def test_absolute_timestamps_and_boundary_clamping() -> None:
    engine = FakeEngine(make_result([TranscriptionSegment(0.25, 1.0000001, " x ")]))
    segment = WindowTranscriber(engine).transcribe(make_window()).segments[0]
    assert (segment.relative_start_seconds, segment.relative_end_seconds) == (0.25, 1.0)
    assert (segment.absolute_start_seconds, segment.absolute_end_seconds) == (
        10.25,
        11.0,
    )


def test_empty_speech_returns_empty_result() -> None:
    result = WindowTranscriber(FakeEngine(make_result(text=""))).transcribe(
        make_window()
    )
    assert result.text == ""
    assert result.segments == ()


@pytest.mark.parametrize("codec", ["pcm_f32le", "wav"])
def test_unsupported_codec_is_rejected(codec: str) -> None:
    with pytest.raises(ValueError, match="Unsupported audio codec"):
        WindowTranscriber(FakeEngine(make_result())).transcribe(
            make_window(codec_name=codec)
        )


def test_empty_pcm_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        WindowTranscriber(FakeEngine(make_result())).transcribe(
            make_window(pcm_bytes=b"")
        )


@pytest.mark.parametrize("start,end", [(-0.1, 0.2), (0.4, 0.3), (float("nan"), 0.2)])
def test_invalid_engine_segment_times_are_rejected(start: float, end: float) -> None:
    engine = FakeEngine(make_result([TranscriptionSegment(start, end, "bad")]))
    with pytest.raises(ValueError, match="invalid segment times"):
        WindowTranscriber(engine).transcribe(make_window())


def test_large_engine_timestamp_overflow_is_rejected() -> None:
    engine = FakeEngine(make_result([TranscriptionSegment(0.2, 1.01, "bad")]))
    with pytest.raises(ValueError, match="invalid segment times"):
        WindowTranscriber(engine).transcribe(make_window())


def test_binary_data_is_hidden_and_source_window_unchanged() -> None:
    window = make_window()
    original = window.model_dump()
    result = WindowTranscriber(FakeEngine(make_result())).transcribe(window)
    assert "pcm" not in repr(result).lower()
    assert window.model_dump() == original
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"  # type: ignore[misc]
