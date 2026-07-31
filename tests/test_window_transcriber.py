from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from app.asr.models import TranscriptionResult, TranscriptionSegment
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.window_transcriber import WindowTranscriber, prepare_whisper_waveform


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
        "sample_rate_hz": 16_000,
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
    assert result.text == "hi"
    assert result.segments[0].text == "hi"
    assert len(engine.calls) == 1
    np.testing.assert_allclose(engine.calls[0], [0.0, 0.5])


def test_audio_preparation_timing_excludes_engine_inference() -> None:
    engine = FakeEngine(make_result())
    clock_values = iter([2.0, 2.125])

    result = WindowTranscriber(
        engine,
        clock=lambda: next(clock_values),
    ).transcribe(make_window())

    assert result.audio_preparation_time_seconds == pytest.approx(0.125)
    assert result.processing_time_seconds == pytest.approx(0.2)


def test_8khz_mono_is_normalized_and_resampled_to_16khz() -> None:
    pcm = np.linspace(-32768, 32767, 8_000, dtype=np.int16)
    window = make_window(
        sample_rate_hz=8_000,
        pcm_bytes=pcm.astype("<i2").tobytes(),
    )
    waveform = prepare_whisper_waveform(window)
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert len(waveform) == 16_000
    assert len(waveform) / 16_000 == pytest.approx(len(pcm) / 8_000)
    assert waveform.min() >= -1.0
    assert waveform.max() <= 1.0


def test_fake_engine_receives_prepared_16khz_waveform_exactly_once() -> None:
    pcm = np.arange(80, dtype="<i2")
    engine = FakeEngine(make_result(text=""))
    WindowTranscriber(engine).transcribe(
        make_window(sample_rate_hz=8_000, pcm_bytes=pcm.tobytes())
    )
    assert len(engine.calls) == 1
    assert engine.calls[0].dtype == np.float32
    assert engine.calls[0].shape == (160,)


def test_16khz_mono_is_not_unnecessarily_resampled() -> None:
    pcm = np.array([-32768, 0, 32767], dtype="<i2")
    waveform = prepare_whisper_waveform(
        make_window(sample_rate_hz=16_000, pcm_bytes=pcm.tobytes())
    )
    assert len(waveform) == len(pcm)
    np.testing.assert_allclose(waveform, [-1.0, 0.0, 32767 / 32768])


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
    np.testing.assert_allclose(engine.calls[0], [-1 / 65536, 0.5])


def test_absolute_timestamps_and_boundary_clamping() -> None:
    engine = FakeEngine(make_result([TranscriptionSegment(0.25, 1.0000001, " x ")]))
    segment = WindowTranscriber(engine).transcribe(make_window()).segments[0]
    assert (segment.relative_start_seconds, segment.relative_end_seconds) == (0.25, 1.0)
    assert (segment.absolute_start_seconds, segment.absolute_end_seconds) == (
        10.25,
        11.0,
    )


def test_segment_end_beyond_short_window_is_clipped() -> None:
    window = make_window(end_seconds=12.0, duration_seconds=2.0)
    engine = FakeEngine(make_result([TranscriptionSegment(0.5, 2.8, " kept ")]))
    result = WindowTranscriber(engine).transcribe(window)
    assert result.text == "kept"
    assert result.segments[0].relative_end_seconds == 2.0
    assert result.segments[0].absolute_end_seconds == 12.0


def test_negative_start_with_overlap_is_clipped_to_zero() -> None:
    engine = FakeEngine(make_result([TranscriptionSegment(-0.4, 0.5, " opening ")]))
    segment = WindowTranscriber(engine).transcribe(make_window()).segments[0]
    assert (segment.relative_start_seconds, segment.relative_end_seconds) == (0.0, 0.5)
    assert (segment.absolute_start_seconds, segment.absolute_end_seconds) == (
        10.0,
        10.5,
    )


def test_segments_completely_outside_window_are_ignored_from_text() -> None:
    engine = FakeEngine(
        make_result(
            [
                TranscriptionSegment(-2.0, -1.0, "before"),
                TranscriptionSegment(0.2, 0.8, "inside"),
                TranscriptionSegment(1.2, 1.8, "after"),
            ],
            text="before inside after",
        )
    )
    result = WindowTranscriber(engine).transcribe(make_window())
    assert result.text == "inside"
    assert [segment.text for segment in result.segments] == ["inside"]


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


def test_malformed_interleaved_stereo_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete interleaved channel frames"):
        prepare_whisper_waveform(make_window(channel_count=2, pcm_bytes=b"\x00\x01"))


@pytest.mark.parametrize(
    "start,end",
    [
        (0.4, 0.3),
        (float("nan"), 0.2),
        (0.1, float("inf")),
        ("invalid", 0.2),
    ],
)
def test_invalid_engine_segment_times_are_rejected(start: object, end: object) -> None:
    engine = FakeEngine(
        make_result(
            [TranscriptionSegment(start, end, "bad")]  # type: ignore[arg-type]
        )
    )
    with pytest.raises(
        ValueError,
        match=r"invalid segment times .*relative_start=.*relative_end=.*window_duration=1.0",
    ) as error:
        WindowTranscriber(engine).transcribe(make_window())
    assert "bad" not in str(error.value)


def test_empty_valid_segment_does_not_affect_combined_text() -> None:
    engine = FakeEngine(
        make_result(
            [TranscriptionSegment(0.1, 0.2, " "), TranscriptionSegment(0.2, 0.3, "ok")],
            text="engine text is ignored",
        )
    )
    assert WindowTranscriber(engine).transcribe(make_window()).text == "ok"


def test_binary_data_is_hidden_and_source_window_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = make_window()
    original = window.model_dump()
    result = WindowTranscriber(FakeEngine(make_result())).transcribe(window)
    assert "pcm" not in repr(result).lower()
    assert "pcm_bytes" not in caplog.text
    assert window.model_dump() == original
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"  # type: ignore[misc]
