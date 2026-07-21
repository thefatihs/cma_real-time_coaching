from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel

from app.asr.models import TranscriptionResult, TranscriptionSegment


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

        started_at = perf_counter()
        raw_segments, info = self._get_model().transcribe(
            str(audio_path),
            vad_filter=self.vad_filter,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            language=self.language,
            beam_size=self.beam_size,
        )

        segments = [
            TranscriptionSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in raw_segments
        ]
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
