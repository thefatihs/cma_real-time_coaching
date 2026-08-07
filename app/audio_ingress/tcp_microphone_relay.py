"""Bounded protocol primitives for a development-only TCP microphone relay."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from hashlib import sha256
import hmac
import json
from math import isfinite
import socket
from struct import Struct
from threading import Lock
from typing import Final, TypeAlias

from app.audio_ingress.local_microphone import (
    LocalMicrophoneIngressSession,
    LocalMicrophoneTerminalReason,
    LocalMicTestCapability,
)


RELAY_MAGIC: Final = b"CMRL"
RELAY_VERSION: Final = 1
RELAY_HEADER: Final = Struct("!4sBBII")
RELAY_HEADER_BYTES: Final = RELAY_HEADER.size
MAX_RELAY_METADATA_BYTES: Final = 2_048
MAX_RELAY_AUDIO_PAYLOAD_BYTES: Final = 64_000
MAX_RELAY_RECORD_BYTES: Final = (
    RELAY_HEADER_BYTES + MAX_RELAY_METADATA_BYTES + MAX_RELAY_AUDIO_PAYLOAD_BYTES
)
MAX_RELAY_IDENTIFIER_LENGTH: Final = 128
MIN_RELAY_TOKEN_LENGTH: Final = 16
MAX_RELAY_TOKEN_LENGTH: Final = 256
MAX_RELAY_DUPLICATE_HISTORY: Final = 64
RELAY_CODEC_NAME: Final = "pcm_s16le"
RELAY_SAMPLE_RATE_HZ: Final = 16_000
RELAY_CHANNEL_COUNT: Final = 1
RELAY_LOOPBACK_HOST: Final = "127.0.0.1"
RELAY_BACKLOG: Final = 1
RELAY_IO_TIMEOUT_SECONDS: Final = 5.0
RELAY_RECV_BYTES: Final = 4_096


class RelayMessageType(IntEnum):
    START = 1
    AUDIO = 2
    PAUSE = 3
    RESUME = 4
    END = 5
    ACK = 6
    ERROR = 7


class RelaySessionState(str, Enum):
    AWAIT_START = "await_start"
    STREAMING = "streaming"
    PAUSED = "paused"
    ENDED = "ended"
    FAILED = "failed"


class RelayAcceptanceStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class RelayEndReason(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RelayReason(str, Enum):
    STARTED = "started"
    AUDIO_ACCEPTED = "audio_accepted"
    PAUSED = "paused"
    RESUMED = "resumed"
    ENDED = "ended"
    EXACT_DUPLICATE = "exact_duplicate"
    BAD_MAGIC = "bad_magic"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNKNOWN_MESSAGE_TYPE = "unknown_message_type"
    METADATA_TOO_LARGE = "metadata_too_large"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNEXPECTED_PAYLOAD = "unexpected_payload"
    AUDIO_PAYLOAD_REQUIRED = "audio_payload_required"
    ODD_AUDIO_PAYLOAD = "odd_audio_payload"
    INVALID_UTF8 = "invalid_utf8"
    MALFORMED_JSON = "malformed_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    NON_FINITE_NUMBER = "non_finite_number"
    UNKNOWN_METADATA_FIELD = "unknown_metadata_field"
    MISSING_METADATA_FIELD = "missing_metadata_field"
    INVALID_METADATA = "invalid_metadata"
    INVALID_IDENTIFIER = "invalid_identifier"
    INVALID_TOKEN = "invalid_token"
    INVALID_FORMAT = "invalid_format"
    INVALID_SAMPLE_COUNT = "invalid_sample_count"
    AUTHENTICATION_FAILED = "authentication_failed"
    SCOPE_MISMATCH = "scope_mismatch"
    UNEXPECTED_MESSAGE_ORDER = "unexpected_message_order"
    SEQUENCE_GAP = "sequence_gap"
    STALE_SEQUENCE = "stale_sequence"
    STALE_GENERATION = "stale_generation"
    GENERATION_MISMATCH = "generation_mismatch"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    TERMINAL_STATE = "terminal_state"
    BUFFER_LIMIT = "buffer_limit"
    RECEIVER_DISABLED = "receiver_disabled"
    INVALID_BIND = "invalid_bind"
    CLIENT_ACTIVE = "client_active"
    IO_TIMEOUT = "io_timeout"
    CONNECTION_CLOSED = "connection_closed"
    SESSION_REJECTED = "session_rejected"


class MicrophoneRelayProtocolError(ValueError):
    """Sanitized protocol error containing a fixed reason code only."""

    def __init__(self, reason: RelayReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class RelayStartMetadata:
    token: str = field(repr=False)
    tenant_id: str
    call_id: str
    stream_id: str
    sequence_number: int
    generation: int
    codec_name: str
    sample_rate_hz: int
    channel_count: int


@dataclass(frozen=True, slots=True)
class RelayAudioMetadata:
    sequence_number: int
    generation: int
    sample_count: int
    captured_at_utc: datetime


@dataclass(frozen=True, slots=True)
class RelayControlMetadata:
    sequence_number: int
    generation: int


@dataclass(frozen=True, slots=True)
class RelayEndMetadata:
    sequence_number: int
    generation: int
    end_reason: RelayEndReason


@dataclass(frozen=True, slots=True)
class RelayResponseMetadata:
    sequence_number: int
    reason: RelayReason


RelayMetadata: TypeAlias = (
    RelayStartMetadata
    | RelayAudioMetadata
    | RelayControlMetadata
    | RelayEndMetadata
    | RelayResponseMetadata
)


@dataclass(frozen=True, slots=True)
class MicrophoneRelayRecord:
    message_type: RelayMessageType
    metadata: RelayMetadata = field(repr=False)
    payload: bytes = field(default=b"", repr=False)
    fingerprint: bytes = field(default=b"", repr=False)

    @property
    def sequence_number(self) -> int:
        return self.metadata.sequence_number


@dataclass(frozen=True, slots=True)
class RelayAcceptance:
    status: RelayAcceptanceStatus
    reason: RelayReason
    state: RelaySessionState
    record: MicrophoneRelayRecord | None = field(default=None, repr=False)


def relay_tokens_match(expected: str, received: str) -> bool:
    """Compare bounded opaque tokens without exposing either value."""
    if not _valid_token(expected) or not _valid_token(received):
        return False
    return hmac.compare_digest(
        expected.encode("utf-8"),
        received.encode("utf-8"),
    )


def encode_relay_record(
    message_type: RelayMessageType,
    metadata: Mapping[str, object],
    payload: bytes = b"",
) -> bytes:
    """Encode one validated relay record without retaining its content."""
    if not isinstance(message_type, RelayMessageType):
        raise MicrophoneRelayProtocolError(RelayReason.UNKNOWN_MESSAGE_TYPE)
    if not isinstance(payload, bytes):
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA)
    try:
        metadata_bytes = json.dumps(
            dict(metadata),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise MicrophoneRelayProtocolError(RelayReason.NON_FINITE_NUMBER) from None
    _validate_declared_lengths(
        message_type,
        metadata_length=len(metadata_bytes),
        payload_length=len(payload),
    )
    _decode_record(message_type, metadata_bytes, payload)
    return (
        RELAY_HEADER.pack(
            RELAY_MAGIC,
            RELAY_VERSION,
            message_type.value,
            len(metadata_bytes),
            len(payload),
        )
        + metadata_bytes
        + payload
    )


class MicrophoneRelayRecordParser:
    """Incrementally parse bounded records from arbitrarily fragmented reads."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._expected_record_bytes: int | None = None
        self._message_type: RelayMessageType | None = None
        self._metadata_length = 0
        self._payload_length = 0
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(
        self,
        data: bytes | bytearray | memoryview,
    ) -> tuple[MicrophoneRelayRecord, ...]:
        if self._failed:
            raise MicrophoneRelayProtocolError(RelayReason.TERMINAL_STATE)
        try:
            view = memoryview(data).cast("B")
        except (TypeError, ValueError):
            self._fail()
            raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA) from None
        records: list[MicrophoneRelayRecord] = []
        offset = 0
        try:
            while offset < len(view):
                if self._expected_record_bytes is None:
                    needed = RELAY_HEADER_BYTES - len(self._buffer)
                    take = min(needed, len(view) - offset)
                    self._buffer.extend(view[offset : offset + take])
                    offset += take
                    if len(self._buffer) < RELAY_HEADER_BYTES:
                        break
                    self._read_header()
                assert self._expected_record_bytes is not None
                remaining = self._expected_record_bytes - len(self._buffer)
                take = min(remaining, len(view) - offset)
                self._buffer.extend(view[offset : offset + take])
                offset += take
                if len(self._buffer) > MAX_RELAY_RECORD_BYTES:
                    raise MicrophoneRelayProtocolError(RelayReason.BUFFER_LIMIT)
                if len(self._buffer) == self._expected_record_bytes:
                    records.append(self._finish_record())
            return tuple(records)
        except MicrophoneRelayProtocolError:
            self._fail()
            raise

    def _read_header(self) -> None:
        magic, version, raw_type, metadata_length, payload_length = RELAY_HEADER.unpack(
            self._buffer
        )
        if magic != RELAY_MAGIC:
            raise MicrophoneRelayProtocolError(RelayReason.BAD_MAGIC)
        if version != RELAY_VERSION:
            raise MicrophoneRelayProtocolError(RelayReason.UNSUPPORTED_VERSION)
        try:
            message_type = RelayMessageType(raw_type)
        except ValueError:
            raise MicrophoneRelayProtocolError(
                RelayReason.UNKNOWN_MESSAGE_TYPE
            ) from None
        _validate_declared_lengths(
            message_type,
            metadata_length=metadata_length,
            payload_length=payload_length,
        )
        expected = RELAY_HEADER_BYTES + metadata_length + payload_length
        if expected > MAX_RELAY_RECORD_BYTES:
            raise MicrophoneRelayProtocolError(RelayReason.BUFFER_LIMIT)
        self._message_type = message_type
        self._metadata_length = metadata_length
        self._payload_length = payload_length
        self._expected_record_bytes = expected

    def _finish_record(self) -> MicrophoneRelayRecord:
        assert self._message_type is not None
        metadata_start = RELAY_HEADER_BYTES
        payload_start = metadata_start + self._metadata_length
        metadata_bytes = bytes(self._buffer[metadata_start:payload_start])
        payload = bytes(self._buffer[payload_start:])
        record = _decode_record(self._message_type, metadata_bytes, payload)
        self._buffer.clear()
        self._expected_record_bytes = None
        self._message_type = None
        self._metadata_length = 0
        self._payload_length = 0
        return record

    def _fail(self) -> None:
        self._failed = True
        self._buffer.clear()
        self._expected_record_bytes = None
        self._message_type = None
        self._metadata_length = 0
        self._payload_length = 0


class MicrophoneRelayStateMachine:
    """Validate exact-scope relay ordering without performing any I/O."""

    def __init__(
        self,
        *,
        expected_token: str,
        tenant_id: str,
        call_id: str,
        stream_id: str,
        duplicate_history_limit: int = MAX_RELAY_DUPLICATE_HISTORY,
    ) -> None:
        if not _valid_token(expected_token):
            raise ValueError(RelayReason.INVALID_TOKEN.value)
        for identifier in (tenant_id, call_id, stream_id):
            if not _valid_identifier(identifier):
                raise ValueError(RelayReason.INVALID_IDENTIFIER.value)
        if (
            type(duplicate_history_limit) is not int
            or duplicate_history_limit <= 0
            or duplicate_history_limit > MAX_RELAY_DUPLICATE_HISTORY
        ):
            raise ValueError(RelayReason.BUFFER_LIMIT.value)
        self._expected_token = expected_token
        self._tenant_id = tenant_id
        self._call_id = call_id
        self._stream_id = stream_id
        self._duplicate_history_limit = duplicate_history_limit
        self._state = RelaySessionState.AWAIT_START
        self._generation: int | None = None
        self._next_sequence = 0
        self._fingerprints: OrderedDict[int, bytes] = OrderedDict()

    @property
    def state(self) -> RelaySessionState:
        return self._state

    @property
    def generation(self) -> int | None:
        return self._generation

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def duplicate_history_size(self) -> int:
        return len(self._fingerprints)

    def accept(self, record: MicrophoneRelayRecord) -> RelayAcceptance:
        if not isinstance(record, MicrophoneRelayRecord):
            return self._reject(RelayReason.INVALID_METADATA)
        if self._state in {RelaySessionState.ENDED, RelaySessionState.FAILED}:
            return RelayAcceptance(
                status=RelayAcceptanceStatus.REJECTED,
                reason=RelayReason.TERMINAL_STATE,
                state=self._state,
            )
        sequence = record.sequence_number
        if sequence < self._next_sequence:
            fingerprint = self._fingerprints.get(sequence)
            if fingerprint is None:
                return self._reject(RelayReason.STALE_SEQUENCE)
            if fingerprint != record.fingerprint:
                return self._reject(RelayReason.CONFLICTING_DUPLICATE)
            return RelayAcceptance(
                status=RelayAcceptanceStatus.DUPLICATE,
                reason=RelayReason.EXACT_DUPLICATE,
                state=self._state,
                record=record,
            )
        if sequence > self._next_sequence:
            return self._reject(RelayReason.SEQUENCE_GAP)
        if self._state is RelaySessionState.AWAIT_START:
            return self._accept_start(record)
        if record.message_type is RelayMessageType.START:
            return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
        if record.message_type in {RelayMessageType.ACK, RelayMessageType.ERROR}:
            return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
        metadata = record.metadata
        if not isinstance(
            metadata,
            (RelayAudioMetadata, RelayControlMetadata, RelayEndMetadata),
        ):
            return self._reject(RelayReason.INVALID_METADATA)
        generation = metadata.generation
        if record.message_type is RelayMessageType.RESUME:
            return self._accept_resume(record, generation)
        if generation != self._generation:
            return self._reject(
                RelayReason.STALE_GENERATION
                if self._generation is not None and generation < self._generation
                else RelayReason.GENERATION_MISMATCH
            )
        if record.message_type is RelayMessageType.AUDIO:
            if self._state is not RelaySessionState.STREAMING:
                return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
            return self._accepted(record, RelayReason.AUDIO_ACCEPTED)
        if record.message_type is RelayMessageType.PAUSE:
            if self._state is not RelaySessionState.STREAMING:
                return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
            self._state = RelaySessionState.PAUSED
            return self._accepted(record, RelayReason.PAUSED)
        if record.message_type is RelayMessageType.END:
            if self._state not in {
                RelaySessionState.STREAMING,
                RelaySessionState.PAUSED,
            }:
                return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
            self._state = RelaySessionState.ENDED
            return self._accepted(record, RelayReason.ENDED)
        return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)

    def fail(self, reason: RelayReason) -> None:
        if self._state is not RelaySessionState.ENDED:
            self._state = RelaySessionState.FAILED
        del reason

    def _accept_start(self, record: MicrophoneRelayRecord) -> RelayAcceptance:
        if record.message_type is not RelayMessageType.START:
            return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
        metadata = record.metadata
        if not isinstance(metadata, RelayStartMetadata):
            return self._reject(RelayReason.INVALID_METADATA)
        if not relay_tokens_match(self._expected_token, metadata.token):
            return self._reject(RelayReason.AUTHENTICATION_FAILED)
        if (
            metadata.tenant_id != self._tenant_id
            or metadata.call_id != self._call_id
            or metadata.stream_id != self._stream_id
        ):
            return self._reject(RelayReason.SCOPE_MISMATCH)
        if metadata.sequence_number != 0:
            return self._reject(RelayReason.SEQUENCE_GAP)
        self._generation = metadata.generation
        self._state = RelaySessionState.STREAMING
        return self._accepted(record, RelayReason.STARTED)

    def _accept_resume(
        self,
        record: MicrophoneRelayRecord,
        generation: int,
    ) -> RelayAcceptance:
        if self._state is not RelaySessionState.PAUSED:
            return self._reject(RelayReason.UNEXPECTED_MESSAGE_ORDER)
        assert self._generation is not None
        if generation <= self._generation:
            return self._reject(RelayReason.STALE_GENERATION)
        if generation != self._generation + 1:
            return self._reject(RelayReason.GENERATION_MISMATCH)
        self._generation = generation
        self._state = RelaySessionState.STREAMING
        return self._accepted(record, RelayReason.RESUMED)

    def _accepted(
        self,
        record: MicrophoneRelayRecord,
        reason: RelayReason,
    ) -> RelayAcceptance:
        self._fingerprints[record.sequence_number] = record.fingerprint
        while len(self._fingerprints) > self._duplicate_history_limit:
            self._fingerprints.popitem(last=False)
        self._next_sequence += 1
        return RelayAcceptance(
            status=RelayAcceptanceStatus.ACCEPTED,
            reason=reason,
            state=self._state,
            record=record,
        )

    def _reject(self, reason: RelayReason) -> RelayAcceptance:
        self._state = RelaySessionState.FAILED
        return RelayAcceptance(
            status=RelayAcceptanceStatus.REJECTED,
            reason=reason,
            state=self._state,
        )


class BoundedMicrophoneRelayProtocol:
    """Compose framing and lifecycle validation into one fail-closed boundary."""

    def __init__(
        self,
        *,
        expected_token: str,
        tenant_id: str,
        call_id: str,
        stream_id: str,
        duplicate_history_limit: int = MAX_RELAY_DUPLICATE_HISTORY,
    ) -> None:
        self._parser = MicrophoneRelayRecordParser()
        self._state_machine = MicrophoneRelayStateMachine(
            expected_token=expected_token,
            tenant_id=tenant_id,
            call_id=call_id,
            stream_id=stream_id,
            duplicate_history_limit=duplicate_history_limit,
        )

    @property
    def state(self) -> RelaySessionState:
        return self._state_machine.state

    @property
    def buffered_bytes(self) -> int:
        return self._parser.buffered_bytes

    @property
    def duplicate_history_size(self) -> int:
        return self._state_machine.duplicate_history_size

    def feed(
        self,
        data: bytes | bytearray | memoryview,
    ) -> tuple[RelayAcceptance, ...]:
        if self.state in {RelaySessionState.ENDED, RelaySessionState.FAILED}:
            return (
                RelayAcceptance(
                    status=RelayAcceptanceStatus.REJECTED,
                    reason=RelayReason.TERMINAL_STATE,
                    state=self.state,
                ),
            )
        try:
            records = self._parser.feed(data)
        except MicrophoneRelayProtocolError as error:
            self._state_machine.fail(error.reason)
            raise
        return tuple(self._state_machine.accept(record) for record in records)

    def fail(self, reason: RelayReason) -> None:
        self._state_machine.fail(reason)


class LocalhostMicrophoneRelayReceiver:
    """Own one loopback listener/client and adapt validated PCM to local ingress."""

    def __init__(
        self,
        *,
        session: LocalMicrophoneIngressSession,
        resource: object,
        expected_token: str,
        tenant_id: str,
        call_id: str,
        stream_id: str,
        resume_capability_factory: Callable[[], LocalMicTestCapability],
        enabled: bool = False,
        bind_host: str = RELAY_LOOPBACK_HOST,
        port: int = 0,
        io_timeout_seconds: float = RELAY_IO_TIMEOUT_SECONDS,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        if bind_host != RELAY_LOOPBACK_HOST:
            raise ValueError(RelayReason.INVALID_BIND.value)
        if type(port) is not int or not 0 <= port <= 65_535:
            raise ValueError(RelayReason.INVALID_BIND.value)
        if (
            not isfinite(io_timeout_seconds)
            or io_timeout_seconds <= 0
            or io_timeout_seconds > RELAY_IO_TIMEOUT_SECONDS
        ):
            raise ValueError(RelayReason.IO_TIMEOUT.value)
        if (
            session.capability.tenant_id != tenant_id
            or session.capability.call_id != call_id
        ):
            raise ValueError(RelayReason.SCOPE_MISMATCH.value)
        self._session = session
        self._resource = resource
        self._resume_capability_factory = resume_capability_factory
        self._enabled = enabled
        self._bind_host = bind_host
        self._port = port
        self._io_timeout_seconds = io_timeout_seconds
        self._socket_factory = socket_factory
        self._protocol = BoundedMicrophoneRelayProtocol(
            expected_token=expected_token,
            tenant_id=tenant_id,
            call_id=call_id,
            stream_id=stream_id,
        )
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None
        self._client_lock = Lock()
        self._client_active = False
        self._closed = False
        self._session_released = False

    @property
    def state(self) -> RelaySessionState:
        return self._protocol.state

    @property
    def listening_address(self) -> tuple[str, int] | None:
        listener = self._listener
        if listener is None:
            return None
        host, port = listener.getsockname()[:2]
        return str(host), int(port)

    @property
    def client_active(self) -> bool:
        with self._client_lock:
            return self._client_active

    def start(self) -> tuple[str, int]:
        if not self._enabled:
            raise PermissionError(RelayReason.RECEIVER_DISABLED.value)
        if self._closed:
            raise RuntimeError(RelayReason.TERMINAL_STATE.value)
        if self._listener is not None:
            address = self.listening_address
            assert address is not None
            return address
        listener = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.settimeout(self._io_timeout_seconds)
            listener.bind((self._bind_host, self._port))
            listener.listen(RELAY_BACKLOG)
        except Exception:
            listener.close()
            raise RuntimeError(RelayReason.INVALID_BIND.value) from None
        self._listener = listener
        address = self.listening_address
        assert address is not None
        return address

    def serve_once(self) -> None:
        listener = self._listener
        if listener is None:
            raise RuntimeError(RelayReason.RECEIVER_DISABLED.value)
        try:
            client, _address = listener.accept()
        except TimeoutError:
            self._fail(RelayReason.IO_TIMEOUT)
            return
        self.serve_connected_socket(client)

    def serve_connected_socket(self, client: socket.socket) -> None:
        if not self._enabled or self._closed:
            client.close()
            raise PermissionError(RelayReason.RECEIVER_DISABLED.value)
        with self._client_lock:
            if self._client_active:
                self._send_error(client, RelayReason.CLIENT_ACTIVE)
                client.close()
                return
            self._client_active = True
            self._client = client
        try:
            client.settimeout(self._io_timeout_seconds)
            while self.state not in {
                RelaySessionState.ENDED,
                RelaySessionState.FAILED,
            }:
                try:
                    data = client.recv(RELAY_RECV_BYTES)
                except TimeoutError:
                    self._fail(RelayReason.IO_TIMEOUT)
                    self._send_error(client, RelayReason.IO_TIMEOUT)
                    return
                if not data:
                    self._fail(RelayReason.CONNECTION_CLOSED)
                    return
                for response in self.process_bytes(data):
                    client.sendall(response)
        except OSError:
            self._fail(RelayReason.CONNECTION_CLOSED)
        finally:
            client.close()
            with self._client_lock:
                if self._client is client:
                    self._client = None
                self._client_active = False

    def process_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        arrived_at_utc: datetime | None = None,
    ) -> tuple[bytes, ...]:
        if self._closed:
            return (self._error_record(0, RelayReason.TERMINAL_STATE),)
        now = arrived_at_utc or datetime.now(UTC)
        try:
            acceptances = self._protocol.feed(data)
        except MicrophoneRelayProtocolError as error:
            self._fail(error.reason)
            return (self._error_record(0, error.reason),)
        responses: list[bytes] = []
        for acceptance in acceptances:
            sequence = (
                acceptance.record.sequence_number
                if acceptance.record is not None
                else 0
            )
            if acceptance.status is RelayAcceptanceStatus.REJECTED:
                self._fail(acceptance.reason)
                responses.append(self._error_record(sequence, acceptance.reason))
                break
            if acceptance.status is RelayAcceptanceStatus.DUPLICATE:
                responses.append(self._ack_record(sequence, acceptance.reason))
                continue
            record = acceptance.record
            assert record is not None
            reason = self._admit(record, arrived_at_utc=now)
            if reason is not None:
                self._fail(reason)
                responses.append(self._error_record(sequence, reason))
                break
            responses.append(self._ack_record(sequence, acceptance.reason))
        return tuple(responses)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        listener, self._listener = self._listener, None
        client, self._client = self._client, None
        if listener is not None:
            listener.close()
        if client is not None:
            client.close()
        self._release_session(LocalMicrophoneTerminalReason.RESOURCE_CLOSED)

    def _admit(
        self,
        record: MicrophoneRelayRecord,
        *,
        arrived_at_utc: datetime,
    ) -> RelayReason | None:
        try:
            if record.message_type is RelayMessageType.START:
                metadata = record.metadata
                assert isinstance(metadata, RelayStartMetadata)
                if metadata.generation != self._session.diagnostics.capture_generation:
                    return RelayReason.GENERATION_MISMATCH
                self._session.start(arrived_at_utc=arrived_at_utc)
            elif record.message_type is RelayMessageType.AUDIO:
                metadata = record.metadata
                assert isinstance(metadata, RelayAudioMetadata)
                self._session.accept_validated_pcm(
                    record.payload,
                    captured_at_utc=metadata.captured_at_utc,
                    arrived_at_utc=arrived_at_utc,
                    capture_generation=metadata.generation,
                )
            elif record.message_type is RelayMessageType.PAUSE:
                if not self._session.pause_capture(
                    resource=self._resource,
                    arrived_at_utc=arrived_at_utc,
                ):
                    return RelayReason.SESSION_REJECTED
                if not self._session.wait_until_paused(
                    resource=self._resource,
                    timeout_seconds=self._io_timeout_seconds,
                ):
                    return RelayReason.IO_TIMEOUT
            elif record.message_type is RelayMessageType.RESUME:
                capability = self._resume_capability_factory()
                if not self._session.resume_capture(
                    capability,
                    resource=self._resource,
                ):
                    capability.revoke()
                    return RelayReason.SESSION_REJECTED
            elif record.message_type is RelayMessageType.END:
                if not self._session.finish_call(
                    resource=self._resource,
                    arrived_at_utc=arrived_at_utc,
                ):
                    return RelayReason.SESSION_REJECTED
            else:
                return RelayReason.UNEXPECTED_MESSAGE_ORDER
        except Exception:
            return RelayReason.SESSION_REJECTED
        return None

    def _fail(self, reason: RelayReason) -> None:
        if self.state is RelaySessionState.ENDED:
            return
        self._protocol.fail(reason)
        self._release_session(
            (
                LocalMicrophoneTerminalReason.DISCONNECTED
                if reason is RelayReason.CONNECTION_CLOSED
                else LocalMicrophoneTerminalReason.FAILED
            )
        )

    def _release_session(self, reason: LocalMicrophoneTerminalReason) -> None:
        if self._session_released:
            return
        self._session_released = True
        self._session.close(reason)

    @staticmethod
    def _ack_record(sequence: int, reason: RelayReason) -> bytes:
        return encode_relay_record(
            RelayMessageType.ACK,
            {"sequence_number": sequence, "reason": reason.value},
        )

    @staticmethod
    def _error_record(sequence: int, reason: RelayReason) -> bytes:
        return encode_relay_record(
            RelayMessageType.ERROR,
            {"sequence_number": sequence, "reason": reason.value},
        )

    @classmethod
    def _send_error(cls, client: socket.socket, reason: RelayReason) -> None:
        try:
            client.settimeout(RELAY_IO_TIMEOUT_SECONDS)
            client.sendall(cls._error_record(0, reason))
        except OSError:
            return


def _validate_declared_lengths(
    message_type: RelayMessageType,
    *,
    metadata_length: int,
    payload_length: int,
) -> None:
    if metadata_length > MAX_RELAY_METADATA_BYTES:
        raise MicrophoneRelayProtocolError(RelayReason.METADATA_TOO_LARGE)
    if message_type is RelayMessageType.AUDIO:
        if payload_length == 0:
            raise MicrophoneRelayProtocolError(RelayReason.AUDIO_PAYLOAD_REQUIRED)
        if payload_length > MAX_RELAY_AUDIO_PAYLOAD_BYTES:
            raise MicrophoneRelayProtocolError(RelayReason.PAYLOAD_TOO_LARGE)
        if payload_length % 2:
            raise MicrophoneRelayProtocolError(RelayReason.ODD_AUDIO_PAYLOAD)
    elif payload_length:
        raise MicrophoneRelayProtocolError(RelayReason.UNEXPECTED_PAYLOAD)


def _decode_record(
    message_type: RelayMessageType,
    metadata_bytes: bytes,
    payload: bytes,
) -> MicrophoneRelayRecord:
    metadata_object = _decode_metadata_json(metadata_bytes)
    metadata = _validate_metadata(message_type, metadata_object, payload)
    digest = sha256()
    digest.update(bytes((message_type.value,)))
    digest.update(metadata_bytes)
    digest.update(payload)
    return MicrophoneRelayRecord(
        message_type=message_type,
        metadata=metadata,
        payload=payload,
        fingerprint=digest.digest(),
    )


def _decode_metadata_json(metadata_bytes: bytes) -> dict[str, object]:
    try:
        text = metadata_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_UTF8) from None

    def reject_duplicate_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MicrophoneRelayProtocolError(RelayReason.DUPLICATE_JSON_KEY)
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise MicrophoneRelayProtocolError(RelayReason.NON_FINITE_NUMBER)

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except MicrophoneRelayProtocolError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError):
        raise MicrophoneRelayProtocolError(RelayReason.MALFORMED_JSON) from None
    if not isinstance(value, dict):
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA)
    return value


def _validate_metadata(
    message_type: RelayMessageType,
    values: dict[str, object],
    payload: bytes,
) -> RelayMetadata:
    if message_type is RelayMessageType.START:
        required = {
            "token",
            "tenant_id",
            "call_id",
            "stream_id",
            "sequence_number",
            "generation",
            "codec_name",
            "sample_rate_hz",
            "channel_count",
        }
        _require_exact_fields(values, required)
        token = _required_token(values["token"])
        tenant_id = _required_identifier(values["tenant_id"])
        call_id = _required_identifier(values["call_id"])
        stream_id = _required_identifier(values["stream_id"])
        sequence = _required_non_negative_int(values["sequence_number"])
        generation = _required_positive_int(values["generation"])
        codec = _required_text(values["codec_name"])
        sample_rate = _required_positive_int(values["sample_rate_hz"])
        channels = _required_positive_int(values["channel_count"])
        if (
            codec != RELAY_CODEC_NAME
            or sample_rate != RELAY_SAMPLE_RATE_HZ
            or channels != RELAY_CHANNEL_COUNT
        ):
            raise MicrophoneRelayProtocolError(RelayReason.INVALID_FORMAT)
        return RelayStartMetadata(
            token=token,
            tenant_id=tenant_id,
            call_id=call_id,
            stream_id=stream_id,
            sequence_number=sequence,
            generation=generation,
            codec_name=codec,
            sample_rate_hz=sample_rate,
            channel_count=channels,
        )
    if message_type is RelayMessageType.AUDIO:
        required = {
            "sequence_number",
            "generation",
            "sample_count",
            "captured_at_utc",
        }
        _require_exact_fields(values, required)
        sequence = _required_non_negative_int(values["sequence_number"])
        generation = _required_positive_int(values["generation"])
        sample_count = _required_positive_int(values["sample_count"])
        captured = _required_aware_datetime(values["captured_at_utc"])
        if sample_count != len(payload) // 2:
            raise MicrophoneRelayProtocolError(RelayReason.INVALID_SAMPLE_COUNT)
        return RelayAudioMetadata(
            sequence_number=sequence,
            generation=generation,
            sample_count=sample_count,
            captured_at_utc=captured,
        )
    if message_type in {RelayMessageType.PAUSE, RelayMessageType.RESUME}:
        _require_exact_fields(values, {"sequence_number", "generation"})
        return RelayControlMetadata(
            sequence_number=_required_non_negative_int(values["sequence_number"]),
            generation=_required_positive_int(values["generation"]),
        )
    if message_type is RelayMessageType.END:
        _require_exact_fields(
            values,
            {"sequence_number", "generation", "end_reason"},
        )
        raw_reason = _required_text(values["end_reason"])
        try:
            end_reason = RelayEndReason(raw_reason)
        except ValueError:
            raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA) from None
        return RelayEndMetadata(
            sequence_number=_required_non_negative_int(values["sequence_number"]),
            generation=_required_positive_int(values["generation"]),
            end_reason=end_reason,
        )
    _require_exact_fields(values, {"sequence_number", "reason"})
    raw_reason = _required_text(values["reason"])
    try:
        reason = RelayReason(raw_reason)
    except ValueError:
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA) from None
    return RelayResponseMetadata(
        sequence_number=_required_non_negative_int(values["sequence_number"]),
        reason=reason,
    )


def _require_exact_fields(values: dict[str, object], required: set[str]) -> None:
    keys = set(values)
    if keys - required:
        raise MicrophoneRelayProtocolError(RelayReason.UNKNOWN_METADATA_FIELD)
    if required - keys:
        raise MicrophoneRelayProtocolError(RelayReason.MISSING_METADATA_FIELD)


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA)
    return value


def _required_identifier(value: object) -> str:
    text = _required_text(value)
    if not _valid_identifier(text):
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_IDENTIFIER)
    return text


def _required_token(value: object) -> str:
    text = _required_text(value)
    if not _valid_token(text):
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_TOKEN)
    return text


def _required_non_negative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA)
    return value


def _required_positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA)
    return value


def _required_aware_datetime(value: object) -> datetime:
    text = _required_text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MicrophoneRelayProtocolError(RelayReason.INVALID_METADATA)
    if not isfinite(parsed.timestamp()):
        raise MicrophoneRelayProtocolError(RelayReason.NON_FINITE_NUMBER)
    return parsed


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and len(value) <= MAX_RELAY_IDENTIFIER_LENGTH
    )


def _valid_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and MIN_RELAY_TOKEN_LENGTH <= len(value) <= MAX_RELAY_TOKEN_LENGTH
    )
