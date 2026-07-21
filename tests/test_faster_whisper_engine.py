from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import pytest

from app.asr.faster_whisper_engine import FasterWhisperEngine
from app.asr.models import TranscriptionResult, TranscriptionSegment


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class FakeTranscriptionInfo:
    language: str | None = "tr"
    language_probability: float | None = 0.98
    duration: float | None = 3.5


class FakeWhisperModel:
    def __init__(self) -> None:
        self.transcribe_calls: list[tuple[str, dict[str, Any]]] = []

    def transcribe(
        self, audio_path: str, **settings: Any
    ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
        self.transcribe_calls.append((audio_path, settings))
        segments = iter(
            [
                FakeSegment(0.0, 1.0, "  Merhaba "),
                FakeSegment(1.0, 2.0, " "),
                FakeSegment(2.0, 3.5, " dünya.  "),
            ]
        )
        return segments, FakeTranscriptionInfo()


def install_fake_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[FakeWhisperModel], list[tuple[object, ...]]]:
    created_models: list[FakeWhisperModel] = []
    constructor_calls: list[tuple[object, ...]] = []

    def fake_constructor(*args: object, **kwargs: object) -> FakeWhisperModel:
        constructor_calls.append((*args, kwargs))
        model = FakeWhisperModel()
        created_models.append(model)
        return model

    monkeypatch.setattr("app.asr.faster_whisper_engine.WhisperModel", fake_constructor)
    return created_models, constructor_calls


def test_missing_audio_file_does_not_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, constructor_calls = install_fake_model(monkeypatch)
    engine = FasterWhisperEngine()

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        engine.transcribe_file(tmp_path / "missing.wav")

    assert constructor_calls == []


def test_directory_does_not_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, constructor_calls = install_fake_model(monkeypatch)
    engine = FasterWhisperEngine()

    with pytest.raises(ValueError, match="is a directory"):
        engine.transcribe_file(tmp_path)

    assert constructor_calls == []


def test_unsupported_extension_does_not_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, constructor_calls = install_fake_model(monkeypatch)
    audio_path = tmp_path / "audio.txt"
    audio_path.touch()
    engine = FasterWhisperEngine()

    with pytest.raises(ValueError, match="Unsupported audio extension"):
        engine.transcribe_file(audio_path)

    assert constructor_calls == []


def test_model_is_loaded_lazily_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_models, constructor_calls = install_fake_model(monkeypatch)
    audio_path = tmp_path / "audio.WAV"
    audio_path.touch()
    engine = FasterWhisperEngine()

    assert constructor_calls == []

    engine.transcribe_file(audio_path)
    engine.transcribe_file(audio_path)

    assert len(constructor_calls) == 1
    assert len(created_models) == 1
    assert len(created_models[0].transcribe_calls) == 2


def test_successful_transcription_returns_structured_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_models, _ = install_fake_model(monkeypatch)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    engine = FasterWhisperEngine()

    result = engine.transcribe_file(audio_path)

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Merhaba dünya."
    assert result.language == "tr"
    assert result.language_probability == pytest.approx(0.98)
    assert result.duration_seconds == pytest.approx(3.5)
    assert result.processing_time_seconds >= 0
    assert result.segments == [
        TranscriptionSegment(0.0, 1.0, "Merhaba"),
        TranscriptionSegment(1.0, 2.0, ""),
        TranscriptionSegment(2.0, 3.5, "dünya."),
    ]

    _, settings = created_models[0].transcribe_calls[0]
    assert settings == {
        "vad_filter": False,
        "condition_on_previous_text": True,
        "language": "tr",
        "beam_size": 1,
    }


def test_missing_optional_metadata_uses_safe_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ModelWithoutMetadata:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], object]:
            return iter([]), object()

    monkeypatch.setattr(
        "app.asr.faster_whisper_engine.WhisperModel",
        lambda *args, **kwargs: ModelWithoutMetadata(),
    )
    audio_path = tmp_path / "audio.ogg"
    audio_path.touch()

    result = FasterWhisperEngine(language=None).transcribe_file(audio_path)

    assert result.language == ""
    assert result.language_probability == 0.0
    assert result.duration_seconds == 0.0
