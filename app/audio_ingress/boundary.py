"""Non-blocking, call-scoped lifecycle boundary for live mono audio."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Callable

from app.audio_ingress.contracts import (
    LiveAudioEndReason,
    LiveAudioEventType,
    LiveAudioIngressEvent,
)
from app.events.models import AudioChunkEvent


class IngressAcceptanceStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    COMPLETED = "completed"
    REJECTED = "rejected"
    OVERLOADED = "overloaded"


class IngressReason(str, Enum):
    STARTED = "started"
    FRAME_ACCEPTED = "frame_accepted"
    EXACT_DUPLICATE = "exact_duplicate"
    ENDED = "ended"
    CANCELLED = "cancelled"
    START_REQUIRED = "start_required"
    DUPLICATE_START = "duplicate_start"
    DUPLICATE_END = "duplicate_end"
    SCOPE_MISMATCH = "scope_mismatch"
    FORMAT_MISMATCH = "format_mismatch"
    SEQUENCE_GAP = "sequence_gap"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    STALE_SEQUENCE = "stale_sequence"
    QUEUE_FULL = "queue_full"
    ACTIVE_STREAM_LIMIT = "active_stream_limit"


@dataclass(frozen=True, slots=True)
class IngressMetrics:
    accepted_frames: int
    rejected_frames: int
    gap_count: int
    duplicate_count: int
    queue_depth: int
    reorder_depth: int
    processing_duration_seconds: float
    capture_to_arrival_latency_seconds: float


@dataclass(frozen=True, slots=True)
class AcceptedLiveAudioChunk:
    """Internal chunk plus provider timing; payload remains hidden from repr."""

    provider_name: str
    provider_stream_id: str
    provider_sequence_number: int
    captured_at_utc: datetime
    arrived_at_utc: datetime
    duration_seconds: float
    audio_chunk: AudioChunkEvent = field(repr=False)


@dataclass(frozen=True, slots=True)
class IngressAcceptance:
    status: IngressAcceptanceStatus
    reason: IngressReason
    metrics: IngressMetrics
    accepted_chunk: AcceptedLiveAudioChunk | None = field(default=None, repr=False)
    end_reason: LiveAudioEndReason | None = None


@dataclass(slots=True)
class _StreamState:
    tenant_id: str
    call_id: str
    provider_name: str
    provider_stream_id: str
    codec_name: str
    sample_rate_hz: int
    channel_count: int
    next_sequence: int
    next_start_seconds: float = 0.0
    accepted_frames: int = 0
    rejected_frames: int = 0
    gap_count: int = 0
    duplicate_count: int = 0
    queue: deque[AcceptedLiveAudioChunk] = field(default_factory=deque)
    fingerprints: OrderedDict[int, bytes] = field(default_factory=OrderedDict)


class LiveAudioIngressBoundary:
    """Perform bounded validation, lifecycle transitions, and chunk adaptation."""

    def __init__(
        self,
        *,
        tenant_id: str,
        provider_name: str,
        max_active_streams: int = 32,
        max_queue_depth: int = 8,
        duplicate_history_limit: int = 64,
        completed_scope_limit: int = 128,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not tenant_id.strip() or not provider_name.strip():
            raise ValueError("invalid_boundary_scope")
        if (
            max_active_streams <= 0
            or max_queue_depth <= 0
            or duplicate_history_limit <= 0
            or completed_scope_limit <= 0
        ):
            raise ValueError("invalid_boundary_limit")
        self._tenant_id = tenant_id.strip()
        self._provider_name = provider_name.strip()
        self._max_active_streams = max_active_streams
        self._max_queue_depth = max_queue_depth
        self._duplicate_history_limit = duplicate_history_limit
        self._completed_scope_limit = completed_scope_limit
        self._clock = clock
        self._active: dict[tuple[str, str], _StreamState] = {}
        self._completed: OrderedDict[tuple[str, str], None] = OrderedDict()

    @property
    def active_stream_count(self) -> int:
        return len(self._active)

    @property
    def retained_audio_chunk_count(self) -> int:
        return sum(len(state.queue) for state in self._active.values())

    @property
    def retained_duplicate_fingerprint_count(self) -> int:
        return sum(len(state.fingerprints) for state in self._active.values())

    def accept(self, event: LiveAudioIngressEvent) -> IngressAcceptance:
        started = self._clock()
        latency = (event.arrived_at_utc - event.captured_at_utc).total_seconds()
        if (
            event.tenant_id != self._tenant_id
            or event.provider_name != self._provider_name
        ):
            return self._stateless_result(
                IngressAcceptanceStatus.REJECTED,
                IngressReason.SCOPE_MISMATCH,
                started,
                latency,
                event=event,
            )
        key = (event.call_id, event.provider_stream_id)
        if event.event_type is LiveAudioEventType.START:
            return self._start(event, key, started, latency)
        if key in self._completed:
            return self._stateless_result(
                IngressAcceptanceStatus.REJECTED,
                (
                    IngressReason.DUPLICATE_END
                    if event.event_type is LiveAudioEventType.END
                    else IngressReason.START_REQUIRED
                ),
                started,
                latency,
                event=event,
            )
        state = self._active.get(key)
        if state is None:
            reason = (
                IngressReason.SCOPE_MISMATCH
                if self._scope_is_active_elsewhere(event)
                else IngressReason.START_REQUIRED
            )
            return self._stateless_result(
                IngressAcceptanceStatus.REJECTED,
                reason,
                started,
                latency,
                event=event,
            )
        if not self._format_matches(state, event):
            state.rejected_frames += int(
                event.event_type is LiveAudioEventType.AUDIO_CHUNK
            )
            return self._result(
                state,
                IngressAcceptanceStatus.REJECTED,
                IngressReason.FORMAT_MISMATCH,
                started,
                latency,
            )
        if event.event_type is LiveAudioEventType.AUDIO_CHUNK:
            return self._audio(state, event, started, latency)
        return self._end(state, event, key, started, latency)

    def drain(
        self,
        *,
        tenant_id: str,
        call_id: str,
        provider_name: str,
        provider_stream_id: str,
        limit: int,
    ) -> tuple[AcceptedLiveAudioChunk, ...]:
        if limit <= 0:
            raise ValueError("invalid_drain_limit")
        if tenant_id != self._tenant_id or provider_name != self._provider_name:
            return ()
        state = self._active.get((call_id, provider_stream_id))
        if state is None:
            return ()
        drained: list[AcceptedLiveAudioChunk] = []
        while state.queue and len(drained) < limit:
            drained.append(state.queue.popleft())
        return tuple(drained)

    def _start(
        self,
        event: LiveAudioIngressEvent,
        key: tuple[str, str],
        started: float,
        latency: float,
    ) -> IngressAcceptance:
        if key in self._active or key in self._completed:
            return self._stateless_result(
                IngressAcceptanceStatus.REJECTED,
                IngressReason.DUPLICATE_START,
                started,
                latency,
                event=event,
            )
        if self._scope_is_active_elsewhere(event):
            return self._stateless_result(
                IngressAcceptanceStatus.REJECTED,
                IngressReason.SCOPE_MISMATCH,
                started,
                latency,
                event=event,
            )
        if len(self._active) >= self._max_active_streams:
            return self._stateless_result(
                IngressAcceptanceStatus.OVERLOADED,
                IngressReason.ACTIVE_STREAM_LIMIT,
                started,
                latency,
                event=event,
            )
        state = _StreamState(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            provider_name=event.provider_name,
            provider_stream_id=event.provider_stream_id,
            codec_name=event.codec_name,
            sample_rate_hz=event.sample_rate_hz,
            channel_count=event.channel_count,
            next_sequence=event.sequence_number + 1,
        )
        self._active[key] = state
        return self._result(
            state,
            IngressAcceptanceStatus.ACCEPTED,
            IngressReason.STARTED,
            started,
            latency,
        )

    def _audio(
        self,
        state: _StreamState,
        event: LiveAudioIngressEvent,
        started: float,
        latency: float,
    ) -> IngressAcceptance:
        fingerprint = _fingerprint(event)
        if event.sequence_number < state.next_sequence:
            previous = state.fingerprints.get(event.sequence_number)
            if previous is None:
                reason = IngressReason.STALE_SEQUENCE
            elif previous == fingerprint:
                state.duplicate_count += 1
                return self._result(
                    state,
                    IngressAcceptanceStatus.DUPLICATE,
                    IngressReason.EXACT_DUPLICATE,
                    started,
                    latency,
                )
            else:
                reason = IngressReason.CONFLICTING_DUPLICATE
            state.rejected_frames += 1
            return self._result(
                state,
                IngressAcceptanceStatus.REJECTED,
                reason,
                started,
                latency,
            )
        if event.sequence_number > state.next_sequence:
            state.rejected_frames += 1
            state.gap_count += 1
            return self._result(
                state,
                IngressAcceptanceStatus.REJECTED,
                IngressReason.SEQUENCE_GAP,
                started,
                latency,
            )
        if len(state.queue) >= self._max_queue_depth:
            state.rejected_frames += 1
            return self._result(
                state,
                IngressAcceptanceStatus.OVERLOADED,
                IngressReason.QUEUE_FULL,
                started,
                latency,
            )
        audio_chunk = AudioChunkEvent(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            sequence_number=event.sequence_number,
            received_at_utc=event.arrived_at_utc,
            chunk_start_seconds=state.next_start_seconds,
            chunk_duration_seconds=event.duration_seconds,
            sample_rate_hz=event.sample_rate_hz,
            channel_count=event.channel_count,
            codec_name=event.codec_name,
            audio_bytes=event.audio_payload,
        )
        accepted = AcceptedLiveAudioChunk(
            provider_name=event.provider_name,
            provider_stream_id=event.provider_stream_id,
            provider_sequence_number=event.sequence_number,
            captured_at_utc=event.captured_at_utc,
            arrived_at_utc=event.arrived_at_utc,
            duration_seconds=event.duration_seconds,
            audio_chunk=audio_chunk,
        )
        state.queue.append(accepted)
        state.fingerprints[event.sequence_number] = fingerprint
        while len(state.fingerprints) > self._duplicate_history_limit:
            state.fingerprints.popitem(last=False)
        state.accepted_frames += 1
        state.next_sequence += 1
        state.next_start_seconds += event.duration_seconds
        return self._result(
            state,
            IngressAcceptanceStatus.ACCEPTED,
            IngressReason.FRAME_ACCEPTED,
            started,
            latency,
            accepted_chunk=accepted,
        )

    def _end(
        self,
        state: _StreamState,
        event: LiveAudioIngressEvent,
        key: tuple[str, str],
        started: float,
        latency: float,
    ) -> IngressAcceptance:
        if event.sequence_number < state.next_sequence:
            return self._result(
                state,
                IngressAcceptanceStatus.REJECTED,
                IngressReason.STALE_SEQUENCE,
                started,
                latency,
            )
        if event.sequence_number > state.next_sequence:
            state.gap_count += 1
            return self._result(
                state,
                IngressAcceptanceStatus.REJECTED,
                IngressReason.SEQUENCE_GAP,
                started,
                latency,
            )
        state.queue.clear()
        del self._active[key]
        self._completed[key] = None
        while len(self._completed) > self._completed_scope_limit:
            self._completed.popitem(last=False)
        assert event.end_reason is not None
        return self._result(
            state,
            IngressAcceptanceStatus.COMPLETED,
            (
                IngressReason.CANCELLED
                if event.end_reason is LiveAudioEndReason.CANCELLED
                else IngressReason.ENDED
            ),
            started,
            latency,
            end_reason=event.end_reason,
            queue_depth=0,
        )

    def _scope_is_active_elsewhere(self, event: LiveAudioIngressEvent) -> bool:
        return any(
            state.call_id == event.call_id
            or state.provider_stream_id == event.provider_stream_id
            for state in self._active.values()
        )

    @staticmethod
    def _format_matches(
        state: _StreamState,
        event: LiveAudioIngressEvent,
    ) -> bool:
        return (
            state.tenant_id == event.tenant_id
            and state.call_id == event.call_id
            and state.provider_name == event.provider_name
            and state.provider_stream_id == event.provider_stream_id
            and state.codec_name == event.codec_name
            and state.sample_rate_hz == event.sample_rate_hz
            and state.channel_count == event.channel_count
        )

    def _result(
        self,
        state: _StreamState,
        status: IngressAcceptanceStatus,
        reason: IngressReason,
        started: float,
        latency: float,
        *,
        accepted_chunk: AcceptedLiveAudioChunk | None = None,
        end_reason: LiveAudioEndReason | None = None,
        queue_depth: int | None = None,
    ) -> IngressAcceptance:
        processing = self._processing_duration(started)
        return IngressAcceptance(
            status=status,
            reason=reason,
            metrics=IngressMetrics(
                accepted_frames=state.accepted_frames,
                rejected_frames=state.rejected_frames,
                gap_count=state.gap_count,
                duplicate_count=state.duplicate_count,
                queue_depth=len(state.queue) if queue_depth is None else queue_depth,
                reorder_depth=0,
                processing_duration_seconds=processing,
                capture_to_arrival_latency_seconds=latency,
            ),
            accepted_chunk=accepted_chunk,
            end_reason=end_reason,
        )

    def _stateless_result(
        self,
        status: IngressAcceptanceStatus,
        reason: IngressReason,
        started: float,
        latency: float,
        *,
        event: LiveAudioIngressEvent,
    ) -> IngressAcceptance:
        return IngressAcceptance(
            status=status,
            reason=reason,
            metrics=IngressMetrics(
                accepted_frames=0,
                rejected_frames=int(event.event_type is LiveAudioEventType.AUDIO_CHUNK),
                gap_count=0,
                duplicate_count=0,
                queue_depth=0,
                reorder_depth=0,
                processing_duration_seconds=self._processing_duration(started),
                capture_to_arrival_latency_seconds=latency,
            ),
        )

    def _processing_duration(self, started: float) -> float:
        duration = self._clock() - started
        return duration if isfinite(duration) and duration >= 0 else 0.0


def _fingerprint(event: LiveAudioIngressEvent) -> bytes:
    digest = sha256()
    digest.update(event.sequence_number.to_bytes(8, "big"))
    digest.update(event.codec_name.encode("utf-8"))
    digest.update(event.sample_rate_hz.to_bytes(4, "big"))
    digest.update(event.channel_count.to_bytes(2, "big"))
    digest.update(event.duration_seconds.hex().encode("ascii"))
    digest.update(event.captured_at_utc.isoformat().encode("ascii"))
    digest.update(event.audio_payload)
    return digest.digest()
