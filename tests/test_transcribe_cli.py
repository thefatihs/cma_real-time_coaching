from pathlib import Path

import pytest

from app.asr.models import TranscriptionResult, TranscriptionSegment
from scripts.transcribe_file import main


class FakeEngine:
    def __init__(
        self, result: TranscriptionResult | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.audio_paths: list[Path] = []

    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        self.audio_paths.append(audio_path)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("Fake result was not configured")
        return self.result


def make_result(duration: float = 10.0) -> TranscriptionResult:
    return TranscriptionResult(
        text="Merhaba dünya.",
        language="tr",
        language_probability=0.98,
        duration_seconds=duration,
        processing_time_seconds=2.5,
        segments=[
            TranscriptionSegment(0.0, 1.2, "Merhaba"),
            TranscriptionSegment(1.2, 2.4, "dünya."),
        ],
    )


def test_successful_output_and_engine_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = FakeEngine(result=make_result())
    received_settings: dict[str, object] = {}

    def engine_factory(**settings: object) -> FakeEngine:
        received_settings.update(settings)
        return engine

    exit_code = main(["samples/deneme.m4a"], engine_factory=engine_factory)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Model: tiny" in output
    assert "Transcript:\nMerhaba dünya." in output
    assert "Language: tr" in output
    assert "Language probability: 98.00%" in output
    assert "Audio duration: 10.00 seconds" in output
    assert "Processing time: 2.50 seconds" in output
    assert "[0.00s -> 1.20s] Merhaba" in output
    assert "[1.20s -> 2.40s] dünya." in output
    assert engine.audio_paths == [Path("samples/deneme.m4a")]
    assert received_settings == {
        "model_size": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "language": "tr",
        "beam_size": 1,
        "cpu_threads": 4,
    }


def test_real_time_factor_calculation(capsys: pytest.CaptureFixture[str]) -> None:
    engine = FakeEngine(result=make_result(duration=10.0))

    assert main(["audio.wav"], engine_factory=lambda **kwargs: engine) == 0

    assert "Real-time factor: 0.250" in capsys.readouterr().out


def test_zero_duration_is_handled_safely(capsys: pytest.CaptureFixture[str]) -> None:
    engine = FakeEngine(result=make_result(duration=0.0))

    assert main(["audio.wav"], engine_factory=lambda **kwargs: engine) == 0

    assert "Real-time factor: N/A" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            FileNotFoundError("Audio file not found: missing.wav"),
            "Audio file not found",
        ),
        (ValueError("Audio path is a directory"), "Audio path is a directory"),
        (
            ValueError("Unsupported audio extension '.txt'"),
            "Unsupported audio extension",
        ),
        (RuntimeError("model unavailable"), "Model loading or transcription failed"),
    ],
)
def test_expected_errors_return_nonzero_exit_code(
    error: Exception,
    expected_message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = FakeEngine(error=error)

    exit_code = main(["audio.wav"], engine_factory=lambda **kwargs: engine)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert expected_message in captured.err
    assert captured.out == ""
