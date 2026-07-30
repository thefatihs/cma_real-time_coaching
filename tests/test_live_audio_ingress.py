from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.audio_ingress import (
    IngressAcceptance,
    IngressAcceptanceStatus,
    IngressReason,
    LiveAudioEndReason,
    LiveAudioEventType,
    LiveAudioIngressBoundary,
    LiveAudioIngressEvent,
    LiveAudioProviderAdapterProtocol,
)
from app.audio_ingress.contracts import MAX_AUDIO_PAYLOAD_BYTES
from app.streaming.rolling_buffer import RollingAudioBuffer


CAPTURED = datetime(2026, 7, 30, 9, 0, tzinfo=UTC)
ARRIVED = CAPTURED + timedelta(milliseconds=25)


def event(
    event_type: LiveAudioEventType,
    sequence: int,
    **changes: object,
) -> LiveAudioIngressEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "provider_name": "synthetic_provider",
        "provider_stream_id": "stream_001",
        "event_type": event_type,
        "sequence_number": sequence,
        "codec_name": "provider_codec",
        "sample_rate_hz": 16_000,
        "channel_count": 1,
        "captured_at_utc": CAPTURED + timedelta(milliseconds=sequence * 20),
        "arrived_at_utc": ARRIVED + timedelta(milliseconds=sequence * 20),
        "duration_seconds": 0.02 if event_type is LiveAudioEventType.AUDIO_CHUNK else 0,
        "audio_payload": (
            bytes([sequence % 251]) * 640
            if event_type is LiveAudioEventType.AUDIO_CHUNK
            else b""
        ),
        "end_reason": (
            LiveAudioEndReason.COMPLETED
            if event_type is LiveAudioEventType.END
            else None
        ),
    }
    values.update(changes)
    return LiveAudioIngressEvent.model_validate(values)


def boundary(**changes: object) -> LiveAudioIngressBoundary:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "provider_name": "synthetic_provider",
    }
    values.update(changes)
    return LiveAudioIngressBoundary(**values)  # type: ignore[arg-type]


class SyntheticProviderAdapter:
    """Test-only injectable provider; it performs no I/O."""

    def __init__(self, events: tuple[LiveAudioIngressEvent, ...]) -> None:
        self._events = events

    def events(self) -> Iterable[LiveAudioIngressEvent]:
        return iter(self._events)


def consume(
    adapter: LiveAudioProviderAdapterProtocol,
    subject: LiveAudioIngressBoundary,
) -> tuple[IngressAcceptance, ...]:
    return tuple(subject.accept(item) for item in adapter.events())


def test_synthetic_adapter_drives_complete_lifecycle() -> None:
    subject = boundary()
    results = consume(
        SyntheticProviderAdapter(
            (
                event(LiveAudioEventType.START, 0),
                event(LiveAudioEventType.AUDIO_CHUNK, 1),
                event(LiveAudioEventType.END, 2),
            )
        ),
        subject,
    )

    assert [result.reason for result in results] == [
        IngressReason.STARTED,
        IngressReason.FRAME_ACCEPTED,
        IngressReason.ENDED,
    ]
    assert results[-1].end_reason is LiveAudioEndReason.COMPLETED
    assert subject.active_stream_count == 0
    assert subject.retained_audio_chunk_count == 0


def test_start_is_required_before_audio() -> None:
    result = boundary().accept(event(LiveAudioEventType.AUDIO_CHUNK, 1))

    assert result.status is IngressAcceptanceStatus.REJECTED
    assert result.reason is IngressReason.START_REQUIRED
    assert result.metrics.rejected_frames == 1


def test_accepted_audio_preserves_timing_and_adapts_to_internal_chunk() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 10))
    source = event(LiveAudioEventType.AUDIO_CHUNK, 11)

    result = subject.accept(source)
    accepted = result.accepted_chunk

    assert result.status is IngressAcceptanceStatus.ACCEPTED
    assert accepted is not None
    assert accepted.provider_sequence_number == source.sequence_number
    assert accepted.captured_at_utc == source.captured_at_utc
    assert accepted.arrived_at_utc == source.arrived_at_utc
    assert accepted.duration_seconds == source.duration_seconds
    assert accepted.audio_chunk.tenant_id == source.tenant_id
    assert accepted.audio_chunk.call_id == source.call_id
    assert accepted.audio_chunk.sequence_number == source.sequence_number
    assert accepted.audio_chunk.received_at_utc == source.arrived_at_utc
    assert accepted.audio_chunk.chunk_start_seconds == 0.0
    assert result.metrics.capture_to_arrival_latency_seconds == pytest.approx(0.025)


def test_drained_chunks_remain_compatible_with_rolling_buffer() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))
    subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 1))
    subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 2))

    drained = subject.drain(
        tenant_id="tenant_alpha",
        call_id="call_001",
        provider_name="synthetic_provider",
        provider_stream_id="stream_001",
        limit=2,
    )
    legacy = RollingAudioBuffer()
    for item in drained:
        legacy.append(item.audio_chunk)

    assert legacy.first_sequence == 1
    assert legacy.last_sequence == 2
    assert legacy.duration_seconds == pytest.approx(0.04)


def test_exact_duplicate_is_idempotent_even_with_later_arrival() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))
    original = event(LiveAudioEventType.AUDIO_CHUNK, 1)
    subject.accept(original)
    duplicate = original.model_copy(
        update={"arrived_at_utc": original.arrived_at_utc + timedelta(milliseconds=5)}
    )

    result = subject.accept(duplicate)

    assert result.status is IngressAcceptanceStatus.DUPLICATE
    assert result.reason is IngressReason.EXACT_DUPLICATE
    assert result.metrics.accepted_frames == 1
    assert result.metrics.duplicate_count == 1
    assert result.metrics.queue_depth == 1


def test_conflicting_duplicate_fails_closed() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))
    subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 1))

    result = subject.accept(
        event(LiveAudioEventType.AUDIO_CHUNK, 1, audio_payload=b"different")
    )

    assert result.status is IngressAcceptanceStatus.REJECTED
    assert result.reason is IngressReason.CONFLICTING_DUPLICATE
    assert result.metrics.rejected_frames == 1


def test_sequence_gap_fails_without_advancing_state() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))

    gap = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 2))
    recovered = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 1))

    assert gap.reason is IngressReason.SEQUENCE_GAP
    assert gap.metrics.gap_count == 1
    assert gap.metrics.reorder_depth == 0
    assert recovered.reason is IngressReason.FRAME_ACCEPTED


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tenant_id": "tenant_beta"}, IngressReason.SCOPE_MISMATCH),
        ({"provider_name": "other_provider"}, IngressReason.SCOPE_MISMATCH),
        ({"call_id": "other_call"}, IngressReason.SCOPE_MISMATCH),
        ({"provider_stream_id": "other_stream"}, IngressReason.SCOPE_MISMATCH),
    ],
)
def test_wrong_scope_fails_closed(
    changes: dict[str, object],
    reason: IngressReason,
) -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))

    result = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 1, **changes))

    assert result.status is IngressAcceptanceStatus.REJECTED
    assert result.reason is reason
    assert subject.active_stream_count == 1


def test_format_change_fails_closed() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))

    result = subject.accept(
        event(LiveAudioEventType.AUDIO_CHUNK, 1, sample_rate_hz=8_000)
    )

    assert result.reason is IngressReason.FORMAT_MISMATCH
    assert result.metrics.rejected_frames == 1


def test_end_and_cancellation_remove_state_immediately() -> None:
    for end_reason in LiveAudioEndReason:
        subject = boundary()
        subject.accept(event(LiveAudioEventType.START, 0))
        subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 1))

        result = subject.accept(event(LiveAudioEventType.END, 2, end_reason=end_reason))

        assert result.status is IngressAcceptanceStatus.COMPLETED
        assert result.reason is (
            IngressReason.CANCELLED
            if end_reason is LiveAudioEndReason.CANCELLED
            else IngressReason.ENDED
        )
        assert result.metrics.queue_depth == 0
        assert subject.active_stream_count == 0
        assert subject.retained_audio_chunk_count == 0


def test_chunks_after_end_and_duplicate_end_fail_closed() -> None:
    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))
    subject.accept(event(LiveAudioEventType.END, 1))

    chunk = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 2))
    duplicate_end = subject.accept(event(LiveAudioEventType.END, 1))

    assert chunk.reason is IngressReason.START_REQUIRED
    assert duplicate_end.reason is IngressReason.DUPLICATE_END


def test_queue_backpressure_is_fixed_and_retryable() -> None:
    subject = boundary(max_queue_depth=1)
    subject.accept(event(LiveAudioEventType.START, 0))
    subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 1))

    overloaded = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 2))
    subject.drain(
        tenant_id="tenant_alpha",
        call_id="call_001",
        provider_name="synthetic_provider",
        provider_stream_id="stream_001",
        limit=1,
    )
    retried = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, 2))

    assert overloaded.status is IngressAcceptanceStatus.OVERLOADED
    assert overloaded.reason is IngressReason.QUEUE_FULL
    assert retried.reason is IngressReason.FRAME_ACCEPTED


def test_active_stream_limit_is_bounded() -> None:
    subject = boundary(max_active_streams=1)
    subject.accept(event(LiveAudioEventType.START, 0))

    result = subject.accept(
        event(
            LiveAudioEventType.START,
            0,
            call_id="call_002",
            provider_stream_id="stream_002",
        )
    )

    assert result.status is IngressAcceptanceStatus.OVERLOADED
    assert result.reason is IngressReason.ACTIVE_STREAM_LIMIT
    assert result.metrics.rejected_frames == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"channel_count": 2},
        {"sample_rate_hz": 0},
        {"codec_name": ""},
        {"duration_seconds": float("nan")},
        {"audio_payload": b""},
        {"audio_payload": b"x" * (MAX_AUDIO_PAYLOAD_BYTES + 1)},
        {"captured_at_utc": datetime(2026, 7, 30, 9, 0)},
        {"arrived_at_utc": CAPTURED - timedelta(seconds=1)},
    ],
)
def test_metadata_and_payload_bounds_are_enforced(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        event(LiveAudioEventType.AUDIO_CHUNK, 1, **changes)


def test_contracts_are_immutable_and_hide_audio() -> None:
    source = event(
        LiveAudioEventType.AUDIO_CHUNK,
        1,
        audio_payload=b"secret-synthetic-audio",
    )

    assert "secret-synthetic-audio" not in repr(source)
    with pytest.raises(ValidationError):
        source.call_id = "changed"

    subject = boundary()
    subject.accept(event(LiveAudioEventType.START, 0))
    result = subject.accept(source)
    assert "secret-synthetic-audio" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "reason", IngressReason.SEQUENCE_GAP)


def test_long_call_retention_is_constant_and_not_full_recording() -> None:
    subject = boundary(max_queue_depth=1, duplicate_history_limit=4)
    subject.accept(event(LiveAudioEventType.START, 0))
    last_processing_duration = 0.0
    for sequence in range(1, 501):
        result = subject.accept(event(LiveAudioEventType.AUDIO_CHUNK, sequence))
        assert result.status is IngressAcceptanceStatus.ACCEPTED
        last_processing_duration = result.metrics.processing_duration_seconds
        subject.drain(
            tenant_id="tenant_alpha",
            call_id="call_001",
            provider_name="synthetic_provider",
            provider_stream_id="stream_001",
            limit=1,
        )

    assert subject.retained_audio_chunk_count == 0
    assert subject.retained_duplicate_fingerprint_count == 4
    assert last_processing_duration < 1.0
