from collections.abc import Callable, Iterable
from pathlib import Path
from math import isfinite
from time import perf_counter
from typing import Any, Protocol, cast

from numpy.typing import NDArray

from app.asr.models import (
    ASRWordTimestamp,
    ASRWordTimestampError,
    ASRWordTimestampErrorCategory,
    TranscriptionResult,
    TranscriptionSegment,
)


class _WhisperModelProtocol(Protocol):
    def transcribe(
        self,
        audio: str | NDArray[Any],
        **settings: Any,
    ) -> tuple[Iterable[object], object]: ...


type _WhisperModelConstructor = Callable[..., _WhisperModelProtocol]


def _load_whisper_model_constructor() -> _WhisperModelConstructor:
    from faster_whisper import WhisperModel

    return cast(_WhisperModelConstructor, WhisperModel)


class FasterWhisperEngine:
    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "tr",
        beam_size: int = 1,
        cpu_threads: int = 4,
        vad_filter: bool = False,
        condition_on_previous_text: bool = True,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        max_skipped_zero_duration_words: int = 1,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        if max_skipped_zero_duration_words < 0:
            raise ValueError("max_skipped_zero_duration_words must be non-negative")
        self.max_skipped_zero_duration_words = max_skipped_zero_duration_words
        self._model: _WhisperModelProtocol | None = None

    def _create_model(self) -> _WhisperModelProtocol:
        model_constructor = _load_whisper_model_constructor()
        return model_constructor(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def _get_model(self) -> _WhisperModelProtocol:
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        self._validate_audio_path(audio_path)

        return self._transcribe(str(audio_path))

    def transcribe_audio(self, audio: NDArray[Any]) -> TranscriptionResult:
        """Transcribe an in-memory mono waveform without persisting audio."""
        return self._transcribe(audio)

    def _transcribe(self, audio: str | NDArray[Any]) -> TranscriptionResult:
        started_at = perf_counter()
        if self.word_timestamps:
            raw_segments, info = self._get_model().transcribe(
                audio,
                vad_filter=self.vad_filter,
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=self.initial_prompt,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=True,
            )
        else:
            raw_segments, info = self._get_model().transcribe(
                audio,
                vad_filter=self.vad_filter,
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=self.initial_prompt,
                language=self.language,
                beam_size=self.beam_size,
            )

        raw_duration = getattr(info, "duration", None)
        audio_duration = float(raw_duration or 0.0)
        segments: list[TranscriptionSegment] = []
        skipped_zero_duration_word_count = 0
        for raw_segment in raw_segments:
            segment, skipped_count = self._convert_segment(
                raw_segment,
                audio_duration=audio_duration,
            )
            skipped_zero_duration_word_count += skipped_count
            if skipped_zero_duration_word_count > self.max_skipped_zero_duration_words:
                raise ASRWordTimestampError(
                    ASRWordTimestampErrorCategory.ZERO_DURATION_ARTIFACT_LIMIT_EXCEEDED
                )
            segments.append(segment)
        transcript = " ".join(segment.text for segment in segments if segment.text)
        processing_time = perf_counter() - started_at

        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)
        return TranscriptionResult(
            text=transcript,
            language=detected_language or self.language or "",
            language_probability=float(language_probability or 0.0),
            duration_seconds=audio_duration,
            processing_time_seconds=processing_time,
            segments=segments,
            skipped_zero_duration_word_count=skipped_zero_duration_word_count,
        )

    def _convert_segment(
        self,
        segment: object,
        *,
        audio_duration: float,
    ) -> tuple[TranscriptionSegment, int]:
        try:
            start = float(getattr(segment, "start"))
            end = float(getattr(segment, "end"))
            text = str(getattr(segment, "text")).strip()
        except Exception:
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.MALFORMED_PROVIDER_OUTPUT
            ) from None
        words, skipped_count = self._convert_words(
            segment,
            start=start,
            end=end,
            audio_duration=audio_duration,
        )
        return (
            TranscriptionSegment(
                start_seconds=start,
                end_seconds=end,
                text=text,
                words=words,
            ),
            skipped_count,
        )

    def _convert_words(
        self,
        segment: object,
        *,
        start: float,
        end: float,
        audio_duration: float,
    ) -> tuple[tuple[ASRWordTimestamp, ...], int]:
        if not self.word_timestamps:
            return (), 0
        if not isfinite(start) or not isfinite(end) or start < 0 or end <= start:
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.MALFORMED_PROVIDER_OUTPUT
            )
        raw_words = getattr(segment, "words", None)
        if raw_words is None:
            return (), 0
        converted: list[ASRWordTimestamp] = []
        skipped_count = 0
        try:
            for raw_word in raw_words:
                probability = getattr(raw_word, "probability", None)
                text = str(getattr(raw_word, "word"))
                word_start = float(getattr(raw_word, "start"))
                word_end = float(getattr(raw_word, "end"))
                word_probability = None if probability is None else float(probability)
                if word_start == word_end:
                    self._validate_zero_duration_artifact(
                        text=text,
                        timestamp=word_start,
                        probability=word_probability,
                        segment_start=start,
                        segment_end=end,
                        audio_duration=audio_duration,
                    )
                    skipped_count += 1
                    continue
                word = ASRWordTimestamp(
                    text=text,
                    start_seconds=word_start,
                    end_seconds=word_end,
                    probability=word_probability,
                )
                if word.start_seconds < start or word.end_seconds > end:
                    raise ASRWordTimestampError(
                        ASRWordTimestampErrorCategory.OUTSIDE_PARENT_SEGMENT
                    )
                converted.append(word)
        except ASRWordTimestampError:
            raise
        except Exception:
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.MALFORMED_PROVIDER_OUTPUT
            ) from None
        keys = [(word.start_seconds, word.end_seconds, word.text) for word in converted]
        if keys != sorted(keys):
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.NONDETERMINISTIC_ORDER
            )
        return tuple(converted), skipped_count

    @staticmethod
    def _validate_zero_duration_artifact(
        *,
        text: str,
        timestamp: float,
        probability: float | None,
        segment_start: float,
        segment_end: float,
        audio_duration: float,
    ) -> None:
        if not text.strip():
            raise ASRWordTimestampError(ASRWordTimestampErrorCategory.INVALID_TEXT)
        if not isfinite(timestamp):
            raise ASRWordTimestampError(ASRWordTimestampErrorCategory.INVALID_TIMESTAMP)
        if probability is not None and (
            not isfinite(probability) or not 0.0 <= probability <= 1.0
        ):
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.INVALID_PROBABILITY
            )
        if (
            timestamp < segment_start
            or timestamp > segment_end
            or timestamp < 0.0
            or not isfinite(audio_duration)
            or timestamp > audio_duration
        ):
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.OUTSIDE_PARENT_SEGMENT
            )

    def _validate_audio_path(self, audio_path: Path) -> None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.is_dir():
            raise ValueError(f"Audio path is a directory, not a file: {audio_path}")
        if audio_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported audio extension '{audio_path.suffix}'. "
                f"Supported extensions: {supported}"
            )
