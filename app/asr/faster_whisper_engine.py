from pathlib import Path
from math import isfinite
from time import perf_counter
from typing import Any

from faster_whisper import WhisperModel
from numpy.typing import NDArray

from app.asr.models import (
    ASRWordTimestamp,
    ASRWordTimestampError,
    ASRWordTimestampErrorCategory,
    TranscriptionResult,
    TranscriptionSegment,
)


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
        self._model: WhisperModel | None = None

    def _create_model(self) -> WhisperModel:
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def _get_model(self) -> WhisperModel:
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

        segments = [self._convert_segment(segment) for segment in raw_segments]
        transcript = " ".join(segment.text for segment in segments if segment.text)
        processing_time = perf_counter() - started_at

        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)
        duration = getattr(info, "duration", None)

        return TranscriptionResult(
            text=transcript,
            language=detected_language or self.language or "",
            language_probability=float(language_probability or 0.0),
            duration_seconds=float(duration or 0.0),
            processing_time_seconds=processing_time,
            segments=segments,
        )

    def _convert_segment(self, segment: object) -> TranscriptionSegment:
        try:
            start = float(getattr(segment, "start"))
            end = float(getattr(segment, "end"))
            text = str(getattr(segment, "text")).strip()
        except Exception:
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.MALFORMED_PROVIDER_OUTPUT
            ) from None
        words = self._convert_words(segment, start=start, end=end)
        return TranscriptionSegment(
            start_seconds=start,
            end_seconds=end,
            text=text,
            words=words,
        )

    def _convert_words(
        self,
        segment: object,
        *,
        start: float,
        end: float,
    ) -> tuple[ASRWordTimestamp, ...]:
        if not self.word_timestamps:
            return ()
        if not isfinite(start) or not isfinite(end) or start < 0 or end <= start:
            raise ASRWordTimestampError(
                ASRWordTimestampErrorCategory.MALFORMED_PROVIDER_OUTPUT
            )
        raw_words = getattr(segment, "words", None)
        if raw_words is None:
            return ()
        converted: list[ASRWordTimestamp] = []
        try:
            for raw_word in raw_words:
                probability = getattr(raw_word, "probability", None)
                word = ASRWordTimestamp(
                    text=str(getattr(raw_word, "word")),
                    start_seconds=float(getattr(raw_word, "start")),
                    end_seconds=float(getattr(raw_word, "end")),
                    probability=(None if probability is None else float(probability)),
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
        return tuple(converted)

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
