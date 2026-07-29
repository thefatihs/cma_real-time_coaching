from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.asr.faster_whisper_engine import FasterWhisperEngine
from app.asr.models import (
    ASRWordTimestamp,
    ASRWordTimestampError,
    ASRWordTimestampErrorCategory,
    TranscriptionResult,
    TranscriptionSegment,
)


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord] | None = None


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


def install_model_constructor(
    monkeypatch: pytest.MonkeyPatch,
    constructor: Callable[..., object],
) -> None:
    monkeypatch.setattr(
        "app.asr.faster_whisper_engine._load_whisper_model_constructor",
        lambda: constructor,
    )


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

    install_model_constructor(monkeypatch, fake_constructor)
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


def test_loader_exception_identity_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_error = ImportError("synthetic provider import failure")

    def failing_loader() -> object:
        raise expected_error

    monkeypatch.setattr(
        "app.asr.faster_whisper_engine._load_whisper_model_constructor",
        failing_loader,
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with pytest.raises(ImportError) as raised:
        FasterWhisperEngine().transcribe_file(audio_path)

    assert raised.value is expected_error


def test_model_constructor_exception_identity_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_error = RuntimeError("synthetic provider construction failure")

    def failing_constructor(*args: object, **kwargs: object) -> object:
        raise expected_error

    install_model_constructor(monkeypatch, failing_constructor)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with pytest.raises(RuntimeError) as raised:
        FasterWhisperEngine().transcribe_file(audio_path)

    assert raised.value is expected_error


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
        "initial_prompt": None,
        "language": "tr",
        "beam_size": 1,
    }
    assert all(segment.words == () for segment in result.segments)


def test_accuracy_options_are_passed_to_faster_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_models, _ = install_fake_model(monkeypatch)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    engine = FasterWhisperEngine(
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="Çağrı merkezi görüşmesi.",
    )

    engine.transcribe_file(audio_path)

    _, settings = created_models[0].transcribe_calls[0]
    assert settings["vad_filter"] is True
    assert settings["condition_on_previous_text"] is False
    assert settings["initial_prompt"] == "Çağrı merkezi görüşmesi."


def test_missing_optional_metadata_uses_safe_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ModelWithoutMetadata:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], object]:
            return iter([]), object()

    install_model_constructor(
        monkeypatch,
        lambda *args, **kwargs: ModelWithoutMetadata(),
    )
    audio_path = tmp_path / "audio.ogg"
    audio_path.touch()

    result = FasterWhisperEngine(language=None).transcribe_file(audio_path)

    assert result.language == ""
    assert result.language_probability == 0.0
    assert result.duration_seconds == 0.0


def test_enabled_word_timestamps_are_requested_and_converted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WordModel(FakeWhisperModel):
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
            self.transcribe_calls.append((audio_path, settings))
            return iter(
                [
                    FakeSegment(
                        0.0,
                        1.5,
                        " Merhaba dünya ",
                        [
                            FakeWord(0.0, 0.6, " Merhaba", 0.9),
                            FakeWord(0.7, 1.5, " dünya ", None),
                        ],
                    )
                ]
            ), FakeTranscriptionInfo()

    model = WordModel()
    install_model_constructor(monkeypatch, lambda *args, **kwargs: model)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    result = FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert model.transcribe_calls[0][1]["word_timestamps"] is True
    assert result.segments[0].words == (
        ASRWordTimestamp("Merhaba", 0.0, 0.6, 0.9),
        ASRWordTimestamp("dünya", 0.7, 1.5),
    )


def test_enabled_word_timestamps_allow_missing_provider_word_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models, _ = install_fake_model(monkeypatch)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    result = FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert all(segment.words == () for segment in result.segments)
    assert created_models[0].transcribe_calls[0][1]["word_timestamps"] is True


@pytest.mark.parametrize(
    ("word", "category"),
    [
        (
            FakeWord(0.0, 0.5, "  ", 0.9),
            ASRWordTimestampErrorCategory.INVALID_TEXT,
        ),
        (
            FakeWord(float("nan"), 0.5, "word", 0.9),
            ASRWordTimestampErrorCategory.INVALID_TIMESTAMP,
        ),
        (
            FakeWord(0.6, 0.5, "word", 0.9),
            ASRWordTimestampErrorCategory.INVALID_TIMESTAMP,
        ),
        (
            FakeWord(0.0, 0.5, "word", 1.1),
            ASRWordTimestampErrorCategory.INVALID_PROBABILITY,
        ),
        (
            FakeWord(0.0, 1.1, "word", 0.9),
            ASRWordTimestampErrorCategory.OUTSIDE_PARENT_SEGMENT,
        ),
    ],
)
def test_malformed_word_timestamps_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    word: FakeWord,
    category: ASRWordTimestampErrorCategory,
) -> None:
    class MalformedWordModel:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
            return iter([FakeSegment(0.0, 1.0, "safe", [word])]), (
                FakeTranscriptionInfo()
            )

    install_model_constructor(
        monkeypatch,
        lambda *args, **kwargs: MalformedWordModel(),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with pytest.raises(ASRWordTimestampError) as error:
        FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert error.value.category is category
    assert "safe" not in str(error.value)


def test_word_timestamps_reject_provider_order_without_sorting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnorderedWordModel:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
            words = [
                FakeWord(0.6, 1.0, "second"),
                FakeWord(0.0, 0.5, "first"),
            ]
            return iter([FakeSegment(0.0, 1.0, "safe", words)]), (
                FakeTranscriptionInfo()
            )

    install_model_constructor(
        monkeypatch,
        lambda *args, **kwargs: UnorderedWordModel(),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with pytest.raises(ASRWordTimestampError) as error:
        FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert error.value.category is ASRWordTimestampErrorCategory.NONDETERMINISTIC_ORDER


def test_one_zero_duration_artifact_is_skipped_without_changing_segment_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment_text = "preserved segment text"

    class ZeroDurationArtifactModel:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
            words = [
                FakeWord(0.0, 0.4, "first", 0.9),
                FakeWord(0.5, 0.5, "artifact", 0.8),
                FakeWord(0.6, 1.0, "last", 0.7),
            ]
            return iter([FakeSegment(0.0, 1.0, segment_text, words)]), (
                FakeTranscriptionInfo(duration=1.0)
            )

    install_model_constructor(
        monkeypatch,
        lambda *args, **kwargs: ZeroDurationArtifactModel(),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    result = FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert result.segments[0].text == segment_text
    assert [
        (word.start_seconds, word.end_seconds) for word in result.segments[0].words
    ] == [
        (0.0, 0.4),
        (0.6, 1.0),
    ]
    assert result.skipped_zero_duration_word_count == 1


def test_second_zero_duration_artifact_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExcessArtifactsModel:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
            words = [
                FakeWord(0.2, 0.2, "artifact-one", 0.9),
                FakeWord(0.4, 0.4, "artifact-two", 0.9),
            ]
            return iter([FakeSegment(0.0, 1.0, "safe", words)]), (
                FakeTranscriptionInfo(duration=1.0)
            )

    install_model_constructor(
        monkeypatch,
        lambda *args, **kwargs: ExcessArtifactsModel(),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with pytest.raises(ASRWordTimestampError) as error:
        FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert (
        error.value.category
        is ASRWordTimestampErrorCategory.ZERO_DURATION_ARTIFACT_LIMIT_EXCEEDED
    )
    assert "artifact-one" not in repr(error.value)
    assert "artifact-two" not in repr(error.value)


@pytest.mark.parametrize("timestamp", [-0.1, 1.1])
def test_out_of_bound_zero_duration_artifact_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timestamp: float,
) -> None:
    class OutOfBoundsArtifactModel:
        def transcribe(
            self, audio_path: str, **settings: Any
        ) -> tuple[Iterator[FakeSegment], FakeTranscriptionInfo]:
            return iter(
                [
                    FakeSegment(
                        0.0,
                        1.0,
                        "safe",
                        [FakeWord(timestamp, timestamp, "artifact", 0.9)],
                    )
                ]
            ), FakeTranscriptionInfo(duration=1.0)

    install_model_constructor(
        monkeypatch,
        lambda *args, **kwargs: OutOfBoundsArtifactModel(),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    with pytest.raises(ASRWordTimestampError) as error:
        FasterWhisperEngine(word_timestamps=True).transcribe_file(audio_path)

    assert error.value.category is ASRWordTimestampErrorCategory.OUTSIDE_PARENT_SEGMENT
