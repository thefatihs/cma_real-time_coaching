"""Privacy-safe local evaluation of one mono audio recording."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Protocol

import av
import numpy as np
from numpy.typing import NDArray

from app.asr.faster_whisper_engine import FasterWhisperEngine
from app.asr.models import ASRWordTimestamp, TranscriptionResult
from app.diarization.composition import (
    DiarizationCompositionOutcome,
    DiarizationCompositionRequest,
    DiarizationCompositionStatus,
)
from app.diarization.models import DiarizationRequest, SpeakerRole
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.window_transcriber import (
    WHISPER_SAMPLE_RATE_HZ,
    prepare_whisper_waveform,
)


SUPPORTED_AUDIO_EXTENSIONS = frozenset(FasterWhisperEngine.SUPPORTED_EXTENSIONS)
DEFAULT_MAX_DURATION_SECONDS = 7_200.0
DEFAULT_MAX_SAMPLE_COUNT = 115_200_000
REPORT_FILENAME = "offline_diarization_report.json"


class OfflineEvaluationStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class OfflineEvaluationReason(str, Enum):
    COMPLETED = "completed"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_AUDIO = "unsupported_audio"
    ASR_FAILED = "asr_failed"
    DIARIZATION_FAILED = "diarization_failed"
    COMPOSITION_FAILED = "composition_failed"
    OUTPUT_FAILED = "output_failed"


@dataclass(frozen=True, slots=True)
class DecodedMonoAudio:
    tenant_id: str
    call_id: str
    sample_rate_hz: int
    samples: tuple[float, ...] = field(repr=False)

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class OfflineEvaluationRequest:
    tenant_id: str
    call_id: str
    audio_path: Path = field(repr=False)
    output_directory: Path | None = field(default=None, repr=False)
    expected_speaker_count: int = 2
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class OfflineEvaluationSummary:
    status: OfflineEvaluationStatus
    reason: OfflineEvaluationReason
    audio_duration_seconds: float | None
    asr_time_seconds: float | None
    asr_real_time_factor: float | None
    diarization_time_seconds: float | None
    diarization_real_time_factor: float | None
    total_processing_time_seconds: float | None
    total_real_time_factor: float | None
    diarization_turn_count: int
    global_speaker_count: int
    agent_role_count: int
    customer_role_count: int
    unknown_role_count: int
    projected_customer_word_count: int
    excluded_agent_word_count: int
    excluded_unknown_word_count: int
    excluded_overlap_word_count: int
    excluded_below_confidence_word_count: int
    transcript_revision: int


class OfflineAudioLoaderProtocol(Protocol):
    def load(
        self,
        audio_path: Path,
        *,
        tenant_id: str,
        call_id: str,
    ) -> DecodedMonoAudio: ...


class OfflineASREngineProtocol(Protocol):
    def transcribe_audio(self, audio: NDArray[np.float32]) -> TranscriptionResult: ...


class DiarizationCompositionProcessorProtocol(Protocol):
    def compose(
        self,
        request: DiarizationCompositionRequest,
    ) -> DiarizationCompositionOutcome: ...


class OfflineMonoAudioLoader:
    """Decode mono audio and reuse the repository Whisper normalization."""

    def __init__(
        self,
        *,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
        max_sample_count: int = DEFAULT_MAX_SAMPLE_COUNT,
        media_opener: Callable[[str], Any] = av.open,
    ) -> None:
        self._max_duration_seconds = max_duration_seconds
        self._max_sample_count = max_sample_count
        self._media_opener = media_opener

    def load(
        self,
        audio_path: Path,
        *,
        tenant_id: str,
        call_id: str,
    ) -> DecodedMonoAudio:
        pcm_parts: list[bytes] = []
        sample_rate_hz: int | None = None
        sample_count = 0
        with self._media_opener(str(audio_path)) as container:
            streams = container.streams.audio
            if len(streams) != 1:
                raise ValueError("invalid_audio_stream_count")
            for frame in container.decode(streams[0]):
                if len(frame.layout.channels) != 1:
                    raise ValueError("non_mono_audio")
                if frame.sample_rate is None or frame.sample_rate <= 0:
                    raise ValueError("invalid_sample_rate")
                if sample_rate_hz is None:
                    sample_rate_hz = frame.sample_rate
                elif frame.sample_rate != sample_rate_hz:
                    raise ValueError("variable_sample_rate")
                array = np.asarray(frame.to_ndarray()).reshape(-1)
                if not np.issubdtype(array.dtype, np.integer):
                    raise ValueError("unsupported_sample_format")
                pcm = np.ascontiguousarray(array, dtype="<i2")
                sample_count += int(pcm.size)
                if sample_count > self._max_sample_count:
                    raise ValueError("audio_sample_limit_exceeded")
                pcm_parts.append(pcm.tobytes())
        if sample_rate_hz is None or sample_count == 0:
            raise ValueError("empty_audio")
        duration_seconds = sample_count / sample_rate_hz
        if (
            not isfinite(duration_seconds)
            or duration_seconds <= 0
            or duration_seconds > self._max_duration_seconds
        ):
            raise ValueError("audio_duration_limit_exceeded")
        window = ASRAudioWindow(
            tenant_id=tenant_id,
            call_id=call_id,
            first_sequence=0,
            last_sequence=0,
            start_seconds=0,
            end_seconds=duration_seconds,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            channel_count=1,
            codec_name="pcm_s16le",
            pcm_bytes=b"".join(pcm_parts),
        )
        waveform = prepare_whisper_waveform(window)
        return DecodedMonoAudio(
            tenant_id=tenant_id,
            call_id=call_id,
            sample_rate_hz=WHISPER_SAMPLE_RATE_HZ,
            samples=tuple(float(sample) for sample in waveform),
        )


class OfflineDiarizationEvaluator:
    def __init__(
        self,
        *,
        audio_loader: OfflineAudioLoaderProtocol,
        asr_engine: OfflineASREngineProtocol,
        composition_processor: DiarizationCompositionProcessorProtocol,
        clock: Callable[[], float] = perf_counter,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
        max_sample_count: int = DEFAULT_MAX_SAMPLE_COUNT,
    ) -> None:
        self._audio_loader = audio_loader
        self._asr_engine = asr_engine
        self._composition_processor = composition_processor
        self._clock = clock
        self._max_duration_seconds = max_duration_seconds
        self._max_sample_count = max_sample_count

    def evaluate(self, request: OfflineEvaluationRequest) -> OfflineEvaluationSummary:
        input_reason = self._validate_request(request)
        if input_reason is not None:
            return _failure(input_reason)
        try:
            audio = self._audio_loader.load(
                request.audio_path,
                tenant_id=request.tenant_id,
                call_id=request.call_id,
            )
            self._validate_audio(audio, request)
        except Exception:
            return _failure(OfflineEvaluationReason.INVALID_INPUT)

        duration = audio.duration_seconds
        waveform = np.asarray(audio.samples, dtype=np.float32)
        started = self._clock()
        try:
            transcription = self._asr_engine.transcribe_audio(waveform)
            words = _absolute_words(transcription, duration)
        except Exception:
            return _failure(
                OfflineEvaluationReason.ASR_FAILED,
                duration=duration,
            )
        after_asr = self._clock()
        try:
            composition = self._composition_processor.compose(
                DiarizationCompositionRequest(
                    tenant_id=request.tenant_id,
                    call_id=request.call_id,
                    transcript_revision=1,
                    diarization_request=DiarizationRequest(
                        tenant_id=request.tenant_id,
                        call_id=request.call_id,
                        window_start_seconds=0,
                        window_end_seconds=duration,
                        sample_rate_hz=audio.sample_rate_hz,
                        mono_audio=audio.samples,
                    ),
                    words=words,
                )
            )
        except Exception:
            return _failure(
                OfflineEvaluationReason.COMPOSITION_FAILED,
                duration=duration,
            )
        finished = self._clock()
        if composition.status not in (
            DiarizationCompositionStatus.COMPLETED,
            DiarizationCompositionStatus.EMPTY,
        ):
            return _failure(
                (
                    OfflineEvaluationReason.DIARIZATION_FAILED
                    if composition.reason.value.startswith("diarizer_")
                    else OfflineEvaluationReason.COMPOSITION_FAILED
                ),
                duration=duration,
            )
        try:
            summary = _summary(
                composition,
                duration=duration,
                asr_time=after_asr - started,
                diarization_time=finished - after_asr,
                total_time=finished - started,
                expected_speaker_count=request.expected_speaker_count,
            )
        except Exception:
            return _failure(
                OfflineEvaluationReason.COMPOSITION_FAILED,
                duration=duration,
            )
        if request.output_directory is not None:
            try:
                _write_report(
                    request.output_directory,
                    summary,
                    composition,
                    overwrite=request.overwrite,
                )
            except Exception:
                return _failure(
                    OfflineEvaluationReason.OUTPUT_FAILED,
                    duration=duration,
                )
        return summary

    @staticmethod
    def _validate_request(
        request: OfflineEvaluationRequest,
    ) -> OfflineEvaluationReason | None:
        if (
            not request.tenant_id.strip()
            or not request.call_id.strip()
            or request.expected_speaker_count <= 0
        ):
            return OfflineEvaluationReason.INVALID_INPUT
        if request.audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            return OfflineEvaluationReason.UNSUPPORTED_AUDIO
        if (
            not request.audio_path.exists()
            or not request.audio_path.is_file()
            or request.audio_path.is_symlink()
        ):
            return OfflineEvaluationReason.INVALID_INPUT
        output = request.output_directory
        if output is not None and (
            not output.exists() or not output.is_dir() or output.is_symlink()
        ):
            return OfflineEvaluationReason.INVALID_INPUT
        return None

    def _validate_audio(
        self,
        audio: DecodedMonoAudio,
        request: OfflineEvaluationRequest,
    ) -> None:
        if audio.tenant_id != request.tenant_id or audio.call_id != request.call_id:
            raise ValueError("audio_scope_mismatch")
        if audio.sample_rate_hz <= 0 or not audio.samples:
            raise ValueError("invalid_audio")
        if len(audio.samples) > self._max_sample_count:
            raise ValueError("audio_sample_limit_exceeded")
        if any(not isfinite(sample) for sample in audio.samples):
            raise ValueError("non_finite_audio")
        duration = audio.duration_seconds
        if (
            not isfinite(duration)
            or duration <= 0
            or duration > self._max_duration_seconds
        ):
            raise ValueError("audio_duration_limit_exceeded")


def _absolute_words(
    transcription: TranscriptionResult,
    duration: float,
) -> tuple[ASRWordTimestamp, ...]:
    words = tuple(word for segment in transcription.segments for word in segment.words)
    if any(
        word.start_seconds < 0
        or word.end_seconds > duration
        or word.end_seconds <= word.start_seconds
        for word in words
    ):
        raise ValueError("asr_word_outside_audio")
    keys = [(word.start_seconds, word.end_seconds, word.text) for word in words]
    if keys != sorted(keys):
        raise ValueError("asr_words_not_ordered")
    return words


def _summary(
    composition: DiarizationCompositionOutcome,
    *,
    duration: float,
    asr_time: float,
    diarization_time: float,
    total_time: float,
    expected_speaker_count: int,
) -> OfflineEvaluationSummary:
    if any(
        not isfinite(value) or value < 0
        for value in (asr_time, diarization_time, total_time)
    ):
        raise ValueError("invalid_timing")
    speakers = {
        speaker_id
        for turn in composition.tracked_turns
        for speaker_id in turn.global_speaker_ids
    }
    if len(speakers) > expected_speaker_count:
        raise ValueError("unexpected_speaker_count")
    role_counts = {role: 0 for role in SpeakerRole}
    for assignment in (
        composition.role_resolution.assignments
        if composition.role_resolution is not None
        else ()
    ):
        role_counts[assignment.role] += 1
    projection = composition.customer_projection
    if projection is None:
        raise ValueError("missing_projection")
    return OfflineEvaluationSummary(
        status=OfflineEvaluationStatus.COMPLETED,
        reason=OfflineEvaluationReason.COMPLETED,
        audio_duration_seconds=duration,
        asr_time_seconds=asr_time,
        asr_real_time_factor=asr_time / duration,
        diarization_time_seconds=diarization_time,
        diarization_real_time_factor=diarization_time / duration,
        total_processing_time_seconds=total_time,
        total_real_time_factor=total_time / duration,
        diarization_turn_count=len(composition.tracked_turns),
        global_speaker_count=len(speakers),
        agent_role_count=role_counts[SpeakerRole.AGENT],
        customer_role_count=role_counts[SpeakerRole.CUSTOMER],
        unknown_role_count=role_counts[SpeakerRole.UNKNOWN],
        projected_customer_word_count=len(projection.customer_words),
        excluded_agent_word_count=projection.excluded_agent_word_count,
        excluded_unknown_word_count=projection.excluded_unknown_word_count,
        excluded_overlap_word_count=projection.excluded_overlap_word_count,
        excluded_below_confidence_word_count=(
            projection.excluded_below_confidence_word_count
        ),
        transcript_revision=composition.transcript_revision,
    )


def _failure(
    reason: OfflineEvaluationReason,
    *,
    duration: float | None = None,
) -> OfflineEvaluationSummary:
    return OfflineEvaluationSummary(
        status=OfflineEvaluationStatus.FAILED,
        reason=reason,
        audio_duration_seconds=duration,
        asr_time_seconds=None,
        asr_real_time_factor=None,
        diarization_time_seconds=None,
        diarization_real_time_factor=None,
        total_processing_time_seconds=None,
        total_real_time_factor=None,
        diarization_turn_count=0,
        global_speaker_count=0,
        agent_role_count=0,
        customer_role_count=0,
        unknown_role_count=0,
        projected_customer_word_count=0,
        excluded_agent_word_count=0,
        excluded_unknown_word_count=0,
        excluded_overlap_word_count=0,
        excluded_below_confidence_word_count=0,
        transcript_revision=0,
    )


def _write_report(
    output_directory: Path,
    summary: OfflineEvaluationSummary,
    composition: DiarizationCompositionOutcome,
    *,
    overwrite: bool,
) -> None:
    destination = output_directory / REPORT_FILENAME
    if destination.exists() and not overwrite:
        raise FileExistsError
    payload = {
        "summary": {
            **asdict(summary),
            "status": summary.status.value,
            "reason": summary.reason.value,
        },
        "turns": [
            {
                "start_seconds": turn.start_seconds,
                "end_seconds": turn.end_seconds,
                "local_speaker_ids": list(turn.local_speaker_ids),
                "global_speaker_ids": list(turn.global_speaker_ids),
                "role": turn.role.value,
                "speaker_confidence": turn.speaker_confidence,
                "role_confidence": turn.role_confidence,
            }
            for turn in composition.tracked_turns
        ],
        "words": [
            {
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
                "local_speaker_ids": list(word.local_speaker_ids),
                "global_speaker_ids": list(word.global_speaker_ids),
                "role": word.role.value,
                "speaker_confidence": word.speaker_confidence,
                "role_confidence": word.role_confidence,
                "role_evidence": (
                    word.role_evidence.value if word.role_evidence is not None else None
                ),
            }
            for word in composition.role_tagged_words
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="x",
            encoding="utf-8",
            dir=output_directory,
            prefix=".offline-diarization-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
            temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
