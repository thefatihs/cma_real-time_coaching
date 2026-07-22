"""Core file-based streaming ASR pipeline orchestration."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.calls.models import CallState
from app.events.models import AudioChunkEvent, TranscriptEvent
from app.streaming.audio_window import ASRAudioWindow, AudioWindowBuilder
from app.streaming.chunk_generator import generate_audio_chunks
from app.streaming.rolling_buffer import RollingAudioBuffer
from app.streaming.transcript_reconciler import TranscriptReconciler
from app.streaming.window_transcriber import WindowTranscriptionResult
from app.tenancy.models import TenantASRConfig, TenantContext


@dataclass(frozen=True, slots=True)
class StreamingASRStep:
    tenant_id: str
    call_id: str
    sequence_number: int
    chunk_start_seconds: float
    chunk_end_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    window_duration_seconds: float
    raw_window_text: str
    transcript_events: tuple[TranscriptEvent, ...]
    stable_transcript: str
    partial_transcript: str
    transcription_time_seconds: float


@dataclass(frozen=True, slots=True)
class StreamingASRResult:
    tenant_id: str
    call_id: str
    steps: tuple[StreamingASRStep, ...]
    final_event: TranscriptEvent | None
    stable_transcript: str
    partial_transcript: str
    total_chunks: int
    audio_duration_seconds: float


class WindowTranscriberProtocol(Protocol):
    def transcribe(self, window: ASRAudioWindow) -> WindowTranscriptionResult: ...


ChunkGenerator = Callable[[Path, str, str, float], Iterable[AudioChunkEvent]]


class StreamingASRPipeline:
    def __init__(
        self,
        tenant_context: TenantContext,
        asr_config: TenantASRConfig,
        window_transcriber: WindowTranscriberProtocol,
        *,
        chunk_generator: ChunkGenerator = generate_audio_chunks,
    ) -> None:
        self._tenant_context = tenant_context
        self._asr_config = asr_config
        self._window_transcriber = window_transcriber
        self._chunk_generator = chunk_generator

    def run(self, audio_path: Path, call_id: str) -> StreamingASRResult:
        call_state = CallState(
            tenant_id=self._tenant_context.tenant_id,
            call_id=call_id,
        )
        buffer = RollingAudioBuffer(self._asr_config.rolling_window_seconds)
        window_builder = AudioWindowBuilder()
        reconciler = TranscriptReconciler(
            stable_region_seconds=self._asr_config.stable_region_seconds
        )
        steps: list[StreamingASRStep] = []
        audio_duration_seconds = 0.0

        chunks = self._chunk_generator(
            audio_path,
            self._tenant_context.tenant_id,
            call_id,
            self._asr_config.chunk_duration_seconds,
        )
        for chunk in chunks:
            call_state.apply_audio_chunk(chunk)
            buffer.append(chunk)
            window = window_builder.build(buffer)
            transcription = self._window_transcriber.transcribe(window)
            self._validate_transcription_scope(transcription, call_id)
            transcript_events = reconciler.ingest(transcription)
            for event in transcript_events:
                call_state.apply_transcript(event)

            chunk_end_seconds = chunk.chunk_start_seconds + chunk.chunk_duration_seconds
            audio_duration_seconds = max(audio_duration_seconds, chunk_end_seconds)
            steps.append(
                StreamingASRStep(
                    tenant_id=chunk.tenant_id,
                    call_id=chunk.call_id,
                    sequence_number=chunk.sequence_number,
                    chunk_start_seconds=chunk.chunk_start_seconds,
                    chunk_end_seconds=chunk_end_seconds,
                    window_start_seconds=window.start_seconds,
                    window_end_seconds=window.end_seconds,
                    window_duration_seconds=window.duration_seconds,
                    raw_window_text=transcription.text,
                    transcript_events=transcript_events,
                    stable_transcript=reconciler.stable_transcript,
                    partial_transcript=reconciler.partial_transcript,
                    transcription_time_seconds=(transcription.processing_time_seconds),
                )
            )

        if not steps:
            raise ValueError("Audio file generated no audio chunks")

        final_event = reconciler.finalize()
        if final_event is not None:
            call_state.apply_transcript(final_event)

        return StreamingASRResult(
            tenant_id=call_state.tenant_id,
            call_id=call_state.call_id,
            steps=tuple(steps),
            final_event=final_event,
            stable_transcript=reconciler.stable_transcript,
            partial_transcript=reconciler.partial_transcript,
            total_chunks=len(steps),
            audio_duration_seconds=audio_duration_seconds,
        )

    def _validate_transcription_scope(
        self, transcription: WindowTranscriptionResult, call_id: str
    ) -> None:
        if transcription.tenant_id != self._tenant_context.tenant_id:
            raise ValueError("Window transcription tenant_id does not match pipeline")
        if transcription.call_id != call_id:
            raise ValueError("Window transcription call_id does not match pipeline")
