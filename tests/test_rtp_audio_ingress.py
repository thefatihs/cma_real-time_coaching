from datetime import UTC, datetime, timedelta
from struct import unpack

import pytest
from pydantic import ValidationError

from app.audio_ingress import (
    OUTPUT_SAMPLE_RATE_HZ,
    RTPCodec,
    RTPIngressReason,
    RTPIngressStatus,
    RTPMediaEnd,
    RTPMediaIngressAdapter,
    RTPMediaPacket,
    RTPMediaStart,
    RTPPayloadFormat,
    decode_g711_payload,
    upsample_pcm16_8khz_to_16khz,
)
from app.events.models import AudioChunkEvent


_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = "tenant-a"
_CALL = "call-a"


def _format(
    codec: RTPCodec = RTPCodec.PCMU,
    *,
    payload_type: int = 0,
) -> RTPPayloadFormat:
    return RTPPayloadFormat(codec=codec, payload_type=payload_type)


def _start(
    *,
    generation: int = 1,
    sequence: int = 0,
    timestamp: int = 0,
    codec: RTPCodec = RTPCodec.PCMU,
    payload_type: int = 0,
    tenant_id: str = _TENANT,
    call_id: str = _CALL,
    ssrc: int | None = 1234,
) -> RTPMediaStart:
    return RTPMediaStart(
        tenant_id=tenant_id,
        call_id=call_id,
        source_generation=generation,
        arrival_monotonic_seconds=0.0,
        arrived_at_utc=_NOW,
        media_format=_format(codec, payload_type=payload_type),
        initial_sequence_number=sequence,
        initial_rtp_timestamp=timestamp,
        ssrc=ssrc,
    )


def _packet(
    sequence: int,
    *,
    generation: int = 1,
    timestamp: int | None = None,
    arrival: float | None = None,
    value: int = 0xFF,
    codec: RTPCodec = RTPCodec.PCMU,
    payload_type: int = 0,
    payload: bytes | None = None,
    tenant_id: str = _TENANT,
    call_id: str = _CALL,
    ssrc: int | None = 1234,
) -> RTPMediaPacket:
    resolved_arrival = sequence * 0.02 if arrival is None else arrival
    return RTPMediaPacket(
        tenant_id=tenant_id,
        call_id=call_id,
        source_generation=generation,
        arrival_monotonic_seconds=resolved_arrival,
        arrived_at_utc=_NOW + timedelta(seconds=resolved_arrival),
        sequence_number=sequence,
        rtp_timestamp=sequence * 160 if timestamp is None else timestamp,
        payload_type=payload_type,
        codec=codec,
        payload=bytes([value]) * 160 if payload is None else payload,
        ssrc=ssrc,
    )


def _end(*, generation: int = 1, arrival: float = 3.0) -> RTPMediaEnd:
    return RTPMediaEnd(
        tenant_id=_TENANT,
        call_id=_CALL,
        source_generation=generation,
        arrival_monotonic_seconds=arrival,
        arrived_at_utc=_NOW + timedelta(seconds=arrival),
    )


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"\xd5\x55\xaa\x2a", (8, -8, 32256, -32256)),
    ],
)
def test_pcma_decodes_known_samples(encoded: bytes, expected: tuple[int, ...]) -> None:
    assert unpack("<4h", decode_g711_payload(RTPCodec.PCMA, encoded)) == expected


def test_pcmu_decodes_known_samples() -> None:
    decoded = decode_g711_payload(RTPCodec.PCMU, b"\xff\x7f\x80\x00")

    assert unpack("<4h", decoded) == (0, 0, 32124, -32124)


def test_pcm16_upsampling_is_deterministic_mono_duplication() -> None:
    source = bytes.fromhex("0100feff")

    assert upsample_pcm16_8khz_to_16khz(source) == bytes.fromhex("01000100fefffeff")


def test_ordered_packets_emit_monotonic_existing_audio_chunks() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    assert adapter.start(_start()).status is RTPIngressStatus.STARTED

    for sequence in range(200):
        result = adapter.accept_packet(_packet(sequence))
        assert result.status is RTPIngressStatus.ACCEPTED

    chunks = adapter.drain_audio_chunks(
        tenant_id=_TENANT,
        call_id=_CALL,
        source_generation=1,
        limit=3,
    )
    assert len(chunks) == 2
    assert [chunk.chunk_start_seconds for chunk in chunks] == [0.0, 2.0]
    assert [chunk.sequence_number for chunk in chunks] == [0, 1]
    chunk = chunks[0]
    assert isinstance(chunk, AudioChunkEvent)
    assert chunk.tenant_id == _TENANT
    assert chunk.call_id == _CALL
    assert chunk.sequence_number == 0
    assert chunk.chunk_start_seconds == 0.0
    assert chunk.chunk_duration_seconds == 2.0
    assert chunk.sample_rate_hz == OUTPUT_SAMPLE_RATE_HZ
    assert chunk.channel_count == 1
    assert chunk.codec_name == "pcm_s16le"
    assert len(chunk.audio_bytes) == 64_000


def test_bounded_out_of_order_packets_are_reordered() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())

    buffered = adapter.accept_packet(
        _packet(1, arrival=0.01, value=0x80),
    )
    accepted = adapter.accept_packet(
        _packet(0, arrival=0.02, value=0xFF),
    )
    adapter.end(_end(arrival=0.03))
    chunk = adapter.drain_audio_chunks(
        tenant_id=_TENANT,
        call_id=_CALL,
        source_generation=1,
        limit=1,
    )[0]

    assert buffered.status is RTPIngressStatus.BUFFERED
    assert accepted.status is RTPIngressStatus.ACCEPTED
    assert adapter.diagnostics.reordered_packet_count == 1
    assert chunk.audio_bytes[:4] == b"\0\0\0\0"
    assert chunk.audio_bytes[640:644] == bytes.fromhex("7c7d7c7d")


def test_exact_duplicate_is_idempotent_and_conflict_fails_closed() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())
    packet = _packet(0)

    assert adapter.accept_packet(packet).status is RTPIngressStatus.ACCEPTED
    duplicate = adapter.accept_packet(packet)
    conflict = adapter.accept_packet(_packet(0, value=0x80, arrival=0.01))

    assert duplicate.status is RTPIngressStatus.DUPLICATE
    assert duplicate.reason is RTPIngressReason.EXACT_DUPLICATE
    assert conflict.status is RTPIngressStatus.REJECTED
    assert conflict.reason is RTPIngressReason.CONFLICTING_DUPLICATE
    assert adapter.diagnostics.accepted_packet_count == 1
    assert adapter.diagnostics.duplicate_packet_count == 1


def test_sequence_gap_wait_is_bounded_and_updates_loss_counter() -> None:
    adapter = RTPMediaIngressAdapter(
        tenant_id=_TENANT,
        call_id=_CALL,
        max_jitter_wait_seconds=0.05,
    )
    adapter.start(_start())
    adapter.accept_packet(_packet(0))
    buffered = adapter.accept_packet(_packet(2, arrival=0.04))

    result = adapter.advance(0.10)

    assert buffered.status is RTPIngressStatus.BUFFERED
    assert result.status is RTPIngressStatus.ACCEPTED
    assert result.diagnostics.missing_sequence_count == 1
    assert result.diagnostics.packet_buffer_depth == 0
    assert result.diagnostics.decoded_pcm_duration_seconds == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("packet", "reason"),
    [
        (_packet(0, payload_type=8), RTPIngressReason.FORMAT_MISMATCH),
        (_packet(0, codec=RTPCodec.PCMA), RTPIngressReason.FORMAT_MISMATCH),
        (_packet(0, payload=b"\xff" * 159), RTPIngressReason.PAYLOAD_SIZE_MISMATCH),
        (_packet(0, tenant_id="other"), RTPIngressReason.SCOPE_MISMATCH),
        (_packet(0, call_id="other"), RTPIngressReason.SCOPE_MISMATCH),
        (
            _packet(0, generation=2),
            RTPIngressReason.SOURCE_GENERATION_MISMATCH,
        ),
        (_packet(0, ssrc=999), RTPIngressReason.SSRC_MISMATCH),
    ],
)
def test_mismatched_packets_fail_closed(
    packet: RTPMediaPacket,
    reason: RTPIngressReason,
) -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())

    result = adapter.accept_packet(packet)

    assert result.status is RTPIngressStatus.REJECTED
    assert result.reason is reason
    assert result.diagnostics.accepted_packet_count == 0


def test_malformed_packet_fails_closed_without_payload_exposure() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())

    result = adapter.accept_packet(object())

    assert result.reason is RTPIngressReason.MALFORMED_PACKET
    assert "payload=" not in repr(_packet(0))
    with pytest.raises(ValidationError):
        _packet(0, payload=b"")


def test_replacement_revokes_old_generation_and_releases_old_audio() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())
    adapter.accept_packet(_packet(0))

    replacement = adapter.replace_source(
        _start(generation=2, sequence=10, timestamp=1_000),
    )
    stale = adapter.accept_packet(_packet(1, generation=1, arrival=0.01))

    assert replacement.status is RTPIngressStatus.STARTED
    assert stale.reason is RTPIngressReason.STALE_SOURCE_GENERATION
    assert stale.diagnostics.decoded_pcm_duration_seconds == 0.0
    assert (
        adapter.drain_audio_chunks(
            tenant_id=_TENANT,
            call_id=_CALL,
            source_generation=1,
            limit=1,
        )
        == ()
    )


def test_end_flushes_final_partial_once_and_duplicate_end_is_idempotent() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())
    adapter.accept_packet(_packet(0))

    completed = adapter.end(_end())
    duplicate = adapter.end(_end(arrival=4.0))
    chunks = adapter.drain_audio_chunks(
        tenant_id=_TENANT,
        call_id=_CALL,
        source_generation=1,
        limit=2,
    )
    drained_again = adapter.drain_audio_chunks(
        tenant_id=_TENANT,
        call_id=_CALL,
        source_generation=1,
        limit=2,
    )

    assert completed.status is RTPIngressStatus.COMPLETED
    assert duplicate.status is RTPIngressStatus.DUPLICATE
    assert duplicate.reason is RTPIngressReason.DUPLICATE_END
    assert len(chunks) == 1
    assert chunks[0].chunk_duration_seconds == pytest.approx(0.02)
    assert drained_again == ()


def test_reset_clears_packet_pcm_and_output_state() -> None:
    adapter = RTPMediaIngressAdapter(tenant_id=_TENANT, call_id=_CALL)
    adapter.start(_start())
    adapter.accept_packet(_packet(1, arrival=0.01))

    reset = adapter.reset()

    assert reset.status is RTPIngressStatus.RESET
    assert reset.diagnostics.packet_buffer_depth == 0
    assert reset.diagnostics.output_queue_depth == 0
    assert reset.diagnostics.decoded_pcm_duration_seconds == 0.0
    assert adapter.accept_packet(_packet(0, arrival=0.02)).reason is (
        RTPIngressReason.START_REQUIRED
    )


def test_output_queue_overload_fails_closed_and_releases_audio() -> None:
    adapter = RTPMediaIngressAdapter(
        tenant_id=_TENANT,
        call_id=_CALL,
        max_output_chunks=1,
    )
    adapter.start(_start())

    for sequence in range(199):
        result = adapter.accept_packet(_packet(sequence))
        assert result.status is RTPIngressStatus.ACCEPTED
    overloaded = adapter.accept_packet(_packet(199))

    assert overloaded.status is RTPIngressStatus.OVERLOADED
    assert overloaded.reason is RTPIngressReason.OUTPUT_QUEUE_FULL
    assert overloaded.diagnostics.packet_buffer_depth == 0
    assert overloaded.diagnostics.output_queue_depth == 0
