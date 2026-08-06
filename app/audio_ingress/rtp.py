"""Provider-independent, in-memory RTP media adaptation for mono telephony."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from math import isfinite
from struct import pack
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.events.models import AudioChunkEvent


RTP_SEQUENCE_MODULUS = 1 << 16
RTP_TIMESTAMP_MODULUS = 1 << 32
TELEPHONY_SAMPLE_RATE_HZ = 8_000
OUTPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_CHUNK_DURATION_SECONDS = 2.0
OUTPUT_CHUNK_BYTES = int(OUTPUT_SAMPLE_RATE_HZ * OUTPUT_CHUNK_DURATION_SECONDS * 2)
DEFAULT_PACKET_SAMPLES = 160
MAX_RTP_PAYLOAD_BYTES = 4_096
MAX_SCOPE_ID_LENGTH = 128


class RTPCodec(str, Enum):
    PCMA = "PCMA"
    PCMU = "PCMU"


class RTPIngressStatus(str, Enum):
    STARTED = "started"
    ACCEPTED = "accepted"
    BUFFERED = "buffered"
    DUPLICATE = "duplicate"
    COMPLETED = "completed"
    RESET = "reset"
    REJECTED = "rejected"
    OVERLOADED = "overloaded"


class RTPIngressReason(str, Enum):
    STARTED = "started"
    PACKET_ACCEPTED = "packet_accepted"
    PACKET_BUFFERED = "packet_buffered"
    EXACT_DUPLICATE = "exact_duplicate"
    ENDED = "ended"
    DUPLICATE_END = "duplicate_end"
    RESET = "reset"
    START_REQUIRED = "start_required"
    ACTIVE_SOURCE_EXISTS = "active_source_exists"
    STALE_SOURCE_GENERATION = "stale_source_generation"
    SOURCE_GENERATION_MISMATCH = "source_generation_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    FORMAT_MISMATCH = "format_mismatch"
    SSRC_MISMATCH = "ssrc_mismatch"
    MALFORMED_PACKET = "malformed_packet"
    PAYLOAD_SIZE_MISMATCH = "payload_size_mismatch"
    ARRIVAL_TIME_REGRESSION = "arrival_time_regression"
    STALE_SEQUENCE = "stale_sequence"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    REORDER_WINDOW_EXCEEDED = "reorder_window_exceeded"
    RTP_TIMESTAMP_MISMATCH = "rtp_timestamp_mismatch"
    OUTPUT_QUEUE_FULL = "output_queue_full"


class RTPPayloadFormat(BaseModel):
    """Explicit mono G.711 payload mapping for one RTP source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    codec: RTPCodec
    payload_type: int = Field(ge=0, le=127)
    sample_rate_hz: int = TELEPHONY_SAMPLE_RATE_HZ
    channel_count: int = 1
    samples_per_packet: int = Field(default=DEFAULT_PACKET_SAMPLES, ge=1, le=960)

    @model_validator(mode="after")
    def validate_telephony_format(self) -> Self:
        if self.sample_rate_hz != TELEPHONY_SAMPLE_RATE_HZ or self.channel_count != 1:
            raise ValueError("unsupported_rtp_media_format")
        return self


class _ScopedRTPControl(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(max_length=MAX_SCOPE_ID_LENGTH)
    call_id: str = Field(max_length=MAX_SCOPE_ID_LENGTH)
    source_generation: int = Field(ge=0)
    arrival_monotonic_seconds: float = Field(ge=0.0)
    arrived_at_utc: datetime

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("invalid_scope")
        return cleaned

    @field_validator("arrival_monotonic_seconds")
    @classmethod
    def validate_monotonic_time(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("invalid_arrival_time")
        return value

    @field_validator("arrived_at_utc")
    @classmethod
    def validate_utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone_required")
        return value


class RTPMediaStart(_ScopedRTPControl):
    """START control for one exact RTP source generation."""

    media_format: RTPPayloadFormat
    initial_sequence_number: int = Field(ge=0, lt=RTP_SEQUENCE_MODULUS)
    initial_rtp_timestamp: int = Field(ge=0, lt=RTP_TIMESTAMP_MODULUS)
    ssrc: int | None = Field(default=None, ge=0, lt=RTP_TIMESTAMP_MODULUS)


class RTPMediaPacket(_ScopedRTPControl):
    """Immutable RTP media packet with hidden, bounded payload."""

    sequence_number: int = Field(ge=0, lt=RTP_SEQUENCE_MODULUS)
    rtp_timestamp: int = Field(ge=0, lt=RTP_TIMESTAMP_MODULUS)
    payload_type: int = Field(ge=0, le=127)
    codec: RTPCodec
    payload: bytes = Field(min_length=1, max_length=MAX_RTP_PAYLOAD_BYTES, repr=False)
    ssrc: int | None = Field(default=None, ge=0, lt=RTP_TIMESTAMP_MODULUS)


class RTPMediaEnd(_ScopedRTPControl):
    """END control for one exact RTP source generation."""


@dataclass(frozen=True, slots=True)
class RTPIngressDiagnostics:
    accepted_packet_count: int
    duplicate_packet_count: int
    reordered_packet_count: int
    missing_sequence_count: int
    rejected_packet_count: int
    latest_rejection_reason: RTPIngressReason | None
    decoded_pcm_duration_seconds: float
    emitted_chunk_count: int
    packet_buffer_depth: int
    output_queue_depth: int


@dataclass(frozen=True, slots=True)
class RTPIngressAcceptance:
    status: RTPIngressStatus
    reason: RTPIngressReason
    diagnostics: RTPIngressDiagnostics


@dataclass(slots=True)
class _ActiveRTPSource:
    start: RTPMediaStart
    expected_sequence_number: int
    expected_rtp_timestamp: int
    last_arrival_monotonic_seconds: float
    latest_arrived_at_utc: datetime
    packet_buffer: dict[int, RTPMediaPacket] = field(default_factory=dict)
    packet_fingerprints: OrderedDict[int, bytes] = field(default_factory=OrderedDict)
    pcm_buffer: bytearray = field(default_factory=bytearray, repr=False)
    output_queue: deque[AudioChunkEvent] = field(default_factory=deque, repr=False)
    gap_started_monotonic_seconds: float | None = None
    accepted_packet_count: int = 0
    duplicate_packet_count: int = 0
    reordered_packet_count: int = 0
    missing_sequence_count: int = 0
    rejected_packet_count: int = 0
    latest_rejection_reason: RTPIngressReason | None = None
    decoded_input_sample_count: int = 0
    emitted_chunk_count: int = 0
    emitted_sample_count: int = 0


class RTPMediaIngressAdapter:
    """Adapt one bounded RTP source into existing mono PCM16 AudioChunkEvents."""

    def __init__(
        self,
        *,
        tenant_id: str,
        call_id: str,
        max_reorder_packets: int = 8,
        max_jitter_wait_seconds: float = 0.1,
        max_output_chunks: int = 8,
        duplicate_history_limit: int = 512,
    ) -> None:
        if not tenant_id.strip() or not call_id.strip():
            raise ValueError("invalid_rtp_adapter_scope")
        if (
            max_reorder_packets <= 0
            or max_reorder_packets >= RTP_SEQUENCE_MODULUS // 2
            or not isfinite(max_jitter_wait_seconds)
            or max_jitter_wait_seconds < 0
            or max_output_chunks <= 0
            or duplicate_history_limit <= 0
        ):
            raise ValueError("invalid_rtp_adapter_limit")
        self._tenant_id = tenant_id.strip()
        self._call_id = call_id.strip()
        self._max_reorder_packets = max_reorder_packets
        self._max_jitter_wait_seconds = max_jitter_wait_seconds
        self._max_output_chunks = max_output_chunks
        self._duplicate_history_limit = duplicate_history_limit
        self._active: _ActiveRTPSource | None = None
        self._completed: _ActiveRTPSource | None = None
        self._highest_source_generation: int | None = None
        self._completed_source_generation: int | None = None

    @property
    def diagnostics(self) -> RTPIngressDiagnostics:
        return self._diagnostics(self._active or self._completed)

    def start(self, control: RTPMediaStart) -> RTPIngressAcceptance:
        scope_reason = self._scope_reason(control)
        if scope_reason is not None:
            return self._stateless_rejection(scope_reason)
        if self._active is not None:
            if control == self._active.start:
                return self._result(
                    RTPIngressStatus.DUPLICATE,
                    RTPIngressReason.ACTIVE_SOURCE_EXISTS,
                )
            return self._rejection(RTPIngressReason.ACTIVE_SOURCE_EXISTS)
        if (
            self._highest_source_generation is not None
            and control.source_generation <= self._highest_source_generation
        ):
            return self._stateless_rejection(RTPIngressReason.STALE_SOURCE_GENERATION)
        self._initialize(control)
        return self._result(RTPIngressStatus.STARTED, RTPIngressReason.STARTED)

    def replace_source(self, control: RTPMediaStart) -> RTPIngressAcceptance:
        scope_reason = self._scope_reason(control)
        if scope_reason is not None:
            return self._rejection(scope_reason)
        if (
            self._highest_source_generation is not None
            and control.source_generation <= self._highest_source_generation
        ):
            return self._rejection(RTPIngressReason.STALE_SOURCE_GENERATION)
        self._release_audio_state()
        self._initialize(control)
        return self._result(RTPIngressStatus.STARTED, RTPIngressReason.STARTED)

    def accept_packet(self, packet: object) -> RTPIngressAcceptance:
        if not isinstance(packet, RTPMediaPacket):
            return self._rejection(RTPIngressReason.MALFORMED_PACKET)
        state = self._active
        if state is None:
            return self._stateless_rejection(RTPIngressReason.START_REQUIRED)
        rejection = self._validate_packet(state, packet)
        if rejection is not None:
            return self._rejection(rejection)

        fingerprint = _packet_fingerprint(packet)
        buffered = state.packet_buffer.get(packet.sequence_number)
        previous = state.packet_fingerprints.get(packet.sequence_number)
        if buffered is not None or previous is not None:
            known = _packet_fingerprint(buffered) if buffered is not None else previous
            if known == fingerprint:
                state.duplicate_packet_count += 1
                return self._result(
                    RTPIngressStatus.DUPLICATE,
                    RTPIngressReason.EXACT_DUPLICATE,
                )
            return self._rejection(RTPIngressReason.CONFLICTING_DUPLICATE)

        distance = _sequence_distance(
            packet.sequence_number,
            state.expected_sequence_number,
        )
        if distance >= RTP_SEQUENCE_MODULUS // 2:
            return self._rejection(RTPIngressReason.STALE_SEQUENCE)
        if distance > self._max_reorder_packets:
            return self._rejection(RTPIngressReason.REORDER_WINDOW_EXCEEDED)

        state.packet_buffer[packet.sequence_number] = packet
        state.packet_fingerprints[packet.sequence_number] = fingerprint
        self._trim_fingerprint_history(state)
        state.accepted_packet_count += 1
        state.latest_arrived_at_utc = packet.arrived_at_utc
        state.last_arrival_monotonic_seconds = packet.arrival_monotonic_seconds
        if distance > 0:
            state.reordered_packet_count += 1
            if state.gap_started_monotonic_seconds is None:
                state.gap_started_monotonic_seconds = packet.arrival_monotonic_seconds

        failure = self._drain_contiguous(state)
        if failure is None:
            failure = self._expire_gap(
                state,
                packet.arrival_monotonic_seconds,
                force=False,
            )
        if failure is not None:
            return self._fail(failure)
        if packet.sequence_number in state.packet_buffer:
            return self._result(
                RTPIngressStatus.BUFFERED,
                RTPIngressReason.PACKET_BUFFERED,
            )
        return self._result(
            RTPIngressStatus.ACCEPTED,
            RTPIngressReason.PACKET_ACCEPTED,
        )

    def advance(self, arrival_monotonic_seconds: float) -> RTPIngressAcceptance:
        state = self._active
        if state is None:
            return self._stateless_rejection(RTPIngressReason.START_REQUIRED)
        if (
            not isfinite(arrival_monotonic_seconds)
            or arrival_monotonic_seconds < state.last_arrival_monotonic_seconds
        ):
            return self._rejection(RTPIngressReason.ARRIVAL_TIME_REGRESSION)
        state.last_arrival_monotonic_seconds = arrival_monotonic_seconds
        failure = self._expire_gap(state, arrival_monotonic_seconds, force=False)
        if failure is not None:
            return self._fail(failure)
        return self._result(
            RTPIngressStatus.ACCEPTED,
            RTPIngressReason.PACKET_ACCEPTED,
        )

    def end(self, control: RTPMediaEnd) -> RTPIngressAcceptance:
        scope_reason = self._scope_reason(control)
        if scope_reason is not None:
            return self._rejection(scope_reason)
        state = self._active
        if state is None:
            if control.source_generation == self._completed_source_generation:
                return RTPIngressAcceptance(
                    status=RTPIngressStatus.DUPLICATE,
                    reason=RTPIngressReason.DUPLICATE_END,
                    diagnostics=self._diagnostics(self._completed),
                )
            return self._stateless_rejection(RTPIngressReason.START_REQUIRED)
        if control.source_generation != state.start.source_generation:
            return self._rejection(self._generation_reason(control.source_generation))
        if control.arrival_monotonic_seconds < state.last_arrival_monotonic_seconds:
            return self._rejection(RTPIngressReason.ARRIVAL_TIME_REGRESSION)

        state.latest_arrived_at_utc = control.arrived_at_utc
        failure = self._expire_gap(
            state,
            control.arrival_monotonic_seconds,
            force=True,
        )
        if failure is None:
            failure = self._flush_final_chunk(state)
        if failure is not None:
            return self._fail(failure)
        state.packet_fingerprints.clear()
        self._completed_source_generation = state.start.source_generation
        self._completed = state
        self._active = None
        return RTPIngressAcceptance(
            status=RTPIngressStatus.COMPLETED,
            reason=RTPIngressReason.ENDED,
            diagnostics=self._diagnostics(state),
        )

    def reset(self) -> RTPIngressAcceptance:
        self._release_audio_state()
        self._completed_source_generation = None
        return RTPIngressAcceptance(
            status=RTPIngressStatus.RESET,
            reason=RTPIngressReason.RESET,
            diagnostics=self._diagnostics(None),
        )

    def drain_audio_chunks(
        self,
        *,
        tenant_id: str,
        call_id: str,
        source_generation: int,
        limit: int,
    ) -> tuple[AudioChunkEvent, ...]:
        if limit <= 0:
            raise ValueError("invalid_drain_limit")
        state = self._active or self._completed
        if (
            state is None
            or tenant_id != self._tenant_id
            or call_id != self._call_id
            or source_generation != state.start.source_generation
        ):
            return ()
        chunks: list[AudioChunkEvent] = []
        while state.output_queue and len(chunks) < limit:
            chunks.append(state.output_queue.popleft())
        return tuple(chunks)

    def _initialize(self, control: RTPMediaStart) -> None:
        self._active = _ActiveRTPSource(
            start=control,
            expected_sequence_number=control.initial_sequence_number,
            expected_rtp_timestamp=control.initial_rtp_timestamp,
            last_arrival_monotonic_seconds=control.arrival_monotonic_seconds,
            latest_arrived_at_utc=control.arrived_at_utc,
        )
        self._highest_source_generation = control.source_generation
        self._completed = None
        self._completed_source_generation = None

    def _validate_packet(
        self,
        state: _ActiveRTPSource,
        packet: RTPMediaPacket,
    ) -> RTPIngressReason | None:
        scope_reason = self._scope_reason(packet)
        if scope_reason is not None:
            return scope_reason
        if packet.source_generation != state.start.source_generation:
            return self._generation_reason(packet.source_generation)
        media_format = state.start.media_format
        if (
            packet.payload_type != media_format.payload_type
            or packet.codec is not media_format.codec
        ):
            return RTPIngressReason.FORMAT_MISMATCH
        if packet.ssrc != state.start.ssrc:
            return RTPIngressReason.SSRC_MISMATCH
        if len(packet.payload) != media_format.samples_per_packet:
            return RTPIngressReason.PAYLOAD_SIZE_MISMATCH
        if packet.arrival_monotonic_seconds < state.last_arrival_monotonic_seconds:
            return RTPIngressReason.ARRIVAL_TIME_REGRESSION
        return None

    def _drain_contiguous(
        self,
        state: _ActiveRTPSource,
    ) -> RTPIngressReason | None:
        while True:
            packet = state.packet_buffer.pop(state.expected_sequence_number, None)
            if packet is None:
                break
            if packet.rtp_timestamp != state.expected_rtp_timestamp:
                return RTPIngressReason.RTP_TIMESTAMP_MISMATCH
            decoded = decode_g711_payload(packet.codec, packet.payload)
            state.pcm_buffer.extend(upsample_pcm16_8khz_to_16khz(decoded))
            state.decoded_input_sample_count += len(packet.payload)
            state.expected_sequence_number = _advance_sequence(
                state.expected_sequence_number,
                1,
            )
            state.expected_rtp_timestamp = _advance_timestamp(
                state.expected_rtp_timestamp,
                len(packet.payload),
            )
            failure = self._emit_complete_chunks(state)
            if failure is not None:
                return failure
        if state.packet_buffer:
            if state.gap_started_monotonic_seconds is None:
                state.gap_started_monotonic_seconds = (
                    state.last_arrival_monotonic_seconds
                )
        else:
            state.gap_started_monotonic_seconds = None
        return None

    def _expire_gap(
        self,
        state: _ActiveRTPSource,
        now: float,
        *,
        force: bool,
    ) -> RTPIngressReason | None:
        while state.packet_buffer:
            if state.expected_sequence_number in state.packet_buffer:
                failure = self._drain_contiguous(state)
                if failure is not None:
                    return failure
                continue
            if not force and (
                state.gap_started_monotonic_seconds is None
                or now - state.gap_started_monotonic_seconds
                < self._max_jitter_wait_seconds
            ):
                return None
            nearest = min(
                state.packet_buffer,
                key=lambda sequence: _sequence_distance(
                    sequence,
                    state.expected_sequence_number,
                ),
            )
            missing = _sequence_distance(nearest, state.expected_sequence_number)
            if missing <= 0 or missing > self._max_reorder_packets:
                return RTPIngressReason.REORDER_WINDOW_EXCEEDED
            state.missing_sequence_count += missing
            silence_samples = missing * state.start.media_format.samples_per_packet * 2
            state.pcm_buffer.extend(b"\0\0" * silence_samples)
            state.expected_sequence_number = _advance_sequence(
                state.expected_sequence_number,
                missing,
            )
            state.expected_rtp_timestamp = _advance_timestamp(
                state.expected_rtp_timestamp,
                missing * state.start.media_format.samples_per_packet,
            )
            failure = self._emit_complete_chunks(state)
            if failure is not None:
                return failure
            state.gap_started_monotonic_seconds = now
        return None

    def _emit_complete_chunks(
        self,
        state: _ActiveRTPSource,
    ) -> RTPIngressReason | None:
        while len(state.pcm_buffer) >= OUTPUT_CHUNK_BYTES:
            payload = bytes(state.pcm_buffer[:OUTPUT_CHUNK_BYTES])
            del state.pcm_buffer[:OUTPUT_CHUNK_BYTES]
            failure = self._queue_chunk(state, payload)
            if failure is not None:
                return failure
        return None

    def _flush_final_chunk(
        self,
        state: _ActiveRTPSource,
    ) -> RTPIngressReason | None:
        if not state.pcm_buffer:
            return None
        payload = bytes(state.pcm_buffer)
        state.pcm_buffer.clear()
        return self._queue_chunk(state, payload)

    def _queue_chunk(
        self,
        state: _ActiveRTPSource,
        payload: bytes,
    ) -> RTPIngressReason | None:
        if len(state.output_queue) >= self._max_output_chunks:
            return RTPIngressReason.OUTPUT_QUEUE_FULL
        sample_count = len(payload) // 2
        start_seconds = state.emitted_sample_count / OUTPUT_SAMPLE_RATE_HZ
        duration_seconds = sample_count / OUTPUT_SAMPLE_RATE_HZ
        state.output_queue.append(
            AudioChunkEvent(
                tenant_id=self._tenant_id,
                call_id=self._call_id,
                sequence_number=state.emitted_chunk_count,
                received_at_utc=state.latest_arrived_at_utc,
                chunk_start_seconds=start_seconds,
                chunk_duration_seconds=duration_seconds,
                sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ,
                channel_count=1,
                codec_name="pcm_s16le",
                audio_bytes=payload,
            )
        )
        state.emitted_sample_count += sample_count
        state.emitted_chunk_count += 1
        return None

    def _scope_reason(
        self,
        control: _ScopedRTPControl,
    ) -> RTPIngressReason | None:
        if control.tenant_id != self._tenant_id or control.call_id != self._call_id:
            return RTPIngressReason.SCOPE_MISMATCH
        return None

    def _generation_reason(self, generation: int) -> RTPIngressReason:
        state = self._active
        if state is not None and generation < state.start.source_generation:
            return RTPIngressReason.STALE_SOURCE_GENERATION
        return RTPIngressReason.SOURCE_GENERATION_MISMATCH

    def _rejection(self, reason: RTPIngressReason) -> RTPIngressAcceptance:
        state = self._active
        if state is not None:
            state.rejected_packet_count += 1
            state.latest_rejection_reason = reason
        return RTPIngressAcceptance(
            status=RTPIngressStatus.REJECTED,
            reason=reason,
            diagnostics=self._diagnostics(state),
        )

    def _stateless_rejection(self, reason: RTPIngressReason) -> RTPIngressAcceptance:
        return RTPIngressAcceptance(
            status=RTPIngressStatus.REJECTED,
            reason=reason,
            diagnostics=self._diagnostics(None),
        )

    def _fail(self, reason: RTPIngressReason) -> RTPIngressAcceptance:
        state = self._active
        if state is not None:
            state.rejected_packet_count += 1
            state.latest_rejection_reason = reason
            state.packet_buffer.clear()
            state.packet_fingerprints.clear()
            state.pcm_buffer.clear()
            state.output_queue.clear()
            self._completed = state
            self._active = None
        return RTPIngressAcceptance(
            status=(
                RTPIngressStatus.OVERLOADED
                if reason is RTPIngressReason.OUTPUT_QUEUE_FULL
                else RTPIngressStatus.REJECTED
            ),
            reason=reason,
            diagnostics=self._diagnostics(state),
        )

    def _result(
        self,
        status: RTPIngressStatus,
        reason: RTPIngressReason,
    ) -> RTPIngressAcceptance:
        return RTPIngressAcceptance(
            status=status,
            reason=reason,
            diagnostics=self._diagnostics(self._active),
        )

    @staticmethod
    def _diagnostics(state: _ActiveRTPSource | None) -> RTPIngressDiagnostics:
        if state is None:
            return RTPIngressDiagnostics(
                accepted_packet_count=0,
                duplicate_packet_count=0,
                reordered_packet_count=0,
                missing_sequence_count=0,
                rejected_packet_count=0,
                latest_rejection_reason=None,
                decoded_pcm_duration_seconds=0.0,
                emitted_chunk_count=0,
                packet_buffer_depth=0,
                output_queue_depth=0,
            )
        return RTPIngressDiagnostics(
            accepted_packet_count=state.accepted_packet_count,
            duplicate_packet_count=state.duplicate_packet_count,
            reordered_packet_count=state.reordered_packet_count,
            missing_sequence_count=state.missing_sequence_count,
            rejected_packet_count=state.rejected_packet_count,
            latest_rejection_reason=state.latest_rejection_reason,
            decoded_pcm_duration_seconds=(
                state.decoded_input_sample_count / TELEPHONY_SAMPLE_RATE_HZ
            ),
            emitted_chunk_count=state.emitted_chunk_count,
            packet_buffer_depth=len(state.packet_buffer),
            output_queue_depth=len(state.output_queue),
        )

    def _trim_fingerprint_history(self, state: _ActiveRTPSource) -> None:
        while len(state.packet_fingerprints) > self._duplicate_history_limit:
            removable = next(
                (
                    sequence
                    for sequence in state.packet_fingerprints
                    if sequence not in state.packet_buffer
                ),
                None,
            )
            if removable is None:
                break
            del state.packet_fingerprints[removable]

    def _release_audio_state(self) -> None:
        if self._active is not None:
            self._active.packet_buffer.clear()
            self._active.packet_fingerprints.clear()
            self._active.pcm_buffer.clear()
            self._active.output_queue.clear()
        if self._completed is not None:
            self._completed.packet_buffer.clear()
            self._completed.packet_fingerprints.clear()
            self._completed.pcm_buffer.clear()
            self._completed.output_queue.clear()
        self._active = None
        self._completed = None


def decode_g711_payload(codec: RTPCodec, payload: bytes) -> bytes:
    """Decode one bounded PCMA/PCMU payload into little-endian signed PCM16."""
    if not payload or len(payload) > MAX_RTP_PAYLOAD_BYTES:
        raise ValueError("invalid_g711_payload")
    if codec is RTPCodec.PCMA:
        samples = tuple(_decode_alaw(value) for value in payload)
    elif codec is RTPCodec.PCMU:
        samples = tuple(_decode_mulaw(value) for value in payload)
    else:
        raise ValueError("unsupported_g711_codec")
    return pack(f"<{len(samples)}h", *samples)


def upsample_pcm16_8khz_to_16khz(pcm_s16le: bytes) -> bytes:
    """Deterministically upsample mono PCM16 by duplicating each source sample."""
    if not pcm_s16le or len(pcm_s16le) % 2:
        raise ValueError("invalid_pcm16_payload")
    output = bytearray(len(pcm_s16le) * 2)
    output_index = 0
    for index in range(0, len(pcm_s16le), 2):
        sample = pcm_s16le[index : index + 2]
        output[output_index : output_index + 2] = sample
        output[output_index + 2 : output_index + 4] = sample
        output_index += 4
    return bytes(output)


def _decode_mulaw(value: int) -> int:
    decoded = (~value) & 0xFF
    sign = decoded & 0x80
    exponent = (decoded >> 4) & 0x07
    mantissa = decoded & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


def _decode_alaw(value: int) -> int:
    decoded = value ^ 0x55
    sign = decoded & 0x80
    exponent = (decoded >> 4) & 0x07
    mantissa = decoded & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    elif exponent == 1:
        sample = (mantissa << 4) + 0x108
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    return sample if sign else -sample


def _packet_fingerprint(packet: RTPMediaPacket) -> bytes:
    digest = sha256()
    digest.update(packet.source_generation.to_bytes(8, "big"))
    digest.update(packet.sequence_number.to_bytes(2, "big"))
    digest.update(packet.rtp_timestamp.to_bytes(4, "big"))
    digest.update(packet.payload_type.to_bytes(1, "big"))
    digest.update(packet.codec.value.encode("ascii"))
    digest.update((packet.ssrc or 0).to_bytes(4, "big"))
    digest.update(packet.payload)
    return digest.digest()


def _sequence_distance(sequence: int, expected: int) -> int:
    return (sequence - expected) % RTP_SEQUENCE_MODULUS


def _advance_sequence(sequence: int, count: int) -> int:
    return (sequence + count) % RTP_SEQUENCE_MODULUS


def _advance_timestamp(timestamp: int, samples: int) -> int:
    return (timestamp + samples) % RTP_TIMESTAMP_MODULUS
