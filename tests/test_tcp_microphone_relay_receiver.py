from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Event, Thread
import socket

import pytest

from app.audio_ingress.local_microphone import (
    LOCAL_MIC_CHUNK_BYTES,
    LOCAL_MIC_GATE_ENVIRONMENT_KEY,
    LocalMicrophoneASRReadiness,
    LocalMicrophoneIngressSession,
    LocalMicrophoneStatus,
    LocalMicrophoneTerminalReason,
    LocalMicTestCapability,
    create_local_mic_test_capability,
)
from app.audio_ingress.tcp_microphone_relay import (
    RELAY_IO_TIMEOUT_SECONDS,
    RELAY_LOOPBACK_HOST,
    LocalhostMicrophoneRelayReceiver,
    MicrophoneRelayRecordParser,
    RelayMessageType,
    RelayReason,
    RelayResponseMetadata,
    RelaySessionState,
    encode_relay_record,
)
from app.events.models import AudioChunkEvent


NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
TOKEN = "synthetic-relay-token-00000001"
ENVIRONMENT = {LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"}


def capability(resource: object) -> LocalMicTestCapability:
    return create_local_mic_test_capability(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=resource,
        server_address=RELAY_LOOPBACK_HOST,
        environment=ENVIRONMENT,
    )


def session_and_receiver(
    *,
    max_queue_depth: int = 8,
    enabled: bool = True,
    bind_host: str = RELAY_LOOPBACK_HOST,
    timeout: float = RELAY_IO_TIMEOUT_SECONDS,
) -> tuple[
    object,
    LocalMicrophoneIngressSession,
    LocalhostMicrophoneRelayReceiver,
]:
    resource = object()
    session = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        provider_stream_id="relay-stream",
        max_queue_depth=max_queue_depth,
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )
    receiver = LocalhostMicrophoneRelayReceiver(
        session=session,
        resource=resource,
        expected_token=TOKEN,
        tenant_id="tenant_alpha",
        call_id="call_001",
        stream_id="relay-stream",
        resume_capability_factory=lambda: capability(resource),
        enabled=enabled,
        bind_host=bind_host,
        io_timeout_seconds=timeout,
    )
    return resource, session, receiver


def start_metadata(
    *,
    token: str = TOKEN,
    tenant_id: str = "tenant_alpha",
    call_id: str = "call_001",
    stream_id: str = "relay-stream",
    generation: int = 1,
) -> dict[str, object]:
    return {
        "token": token,
        "tenant_id": tenant_id,
        "call_id": call_id,
        "stream_id": stream_id,
        "sequence_number": 0,
        "generation": generation,
        "codec_name": "pcm_s16le",
        "sample_rate_hz": 16_000,
        "channel_count": 1,
    }


def audio_record(
    sequence: int,
    *,
    generation: int = 1,
    payload: bytes = b"\1\0",
    captured_at_utc: datetime = NOW,
) -> bytes:
    return encode_relay_record(
        RelayMessageType.AUDIO,
        {
            "sequence_number": sequence,
            "generation": generation,
            "sample_count": len(payload) // 2,
            "captured_at_utc": captured_at_utc.isoformat(),
        },
        payload,
    )


def control_record(
    message_type: RelayMessageType,
    sequence: int,
    *,
    generation: int = 1,
) -> bytes:
    metadata: dict[str, object] = {
        "sequence_number": sequence,
        "generation": generation,
    }
    if message_type is RelayMessageType.END:
        metadata["end_reason"] = "completed"
    return encode_relay_record(message_type, metadata)


def start_record(**changes: object) -> bytes:
    metadata = start_metadata()
    metadata.update(changes)
    return encode_relay_record(RelayMessageType.START, metadata)


def response_reason(response: bytes) -> tuple[RelayMessageType, RelayReason]:
    record = MicrophoneRelayRecordParser().feed(response)[0]
    assert isinstance(record.metadata, RelayResponseMetadata)
    return record.message_type, record.metadata.reason


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "localhost", "192.0.2.1"])
def test_receiver_rejects_every_non_literal_loopback_bind(host: str) -> None:
    with pytest.raises(ValueError, match=RelayReason.INVALID_BIND.value):
        session_and_receiver(bind_host=host)


def test_receiver_binds_only_literal_ipv4_loopback_with_backlog_one() -> None:
    _resource, _session, receiver = session_and_receiver()

    host, port = receiver.start()

    assert host == RELAY_LOOPBACK_HOST
    assert 0 < port <= 65_535
    receiver.close()
    receiver.close()


def test_receiver_is_default_off() -> None:
    _resource, _session, receiver = session_and_receiver(enabled=False)

    with pytest.raises(PermissionError, match=RelayReason.RECEIVER_DISABLED.value):
        receiver.start()


def test_start_audio_uses_existing_audio_chunk_event_without_renormalizing() -> None:
    resource, session, receiver = session_and_receiver()
    payload = b"\x01\x80" * (LOCAL_MIC_CHUNK_BYTES // 2)

    assert response_reason(receiver.process_bytes(start_record())[0]) == (
        RelayMessageType.ACK,
        RelayReason.STARTED,
    )
    assert response_reason(
        receiver.process_bytes(
            audio_record(1, payload=payload),
            arrived_at_utc=NOW + timedelta(milliseconds=50),
        )[0]
    ) == (RelayMessageType.ACK, RelayReason.AUDIO_ACCEPTED)

    chunks = iter(session.iter_audio_chunks(cancellation=Event()))
    chunk = next(chunks)
    assert type(chunk) is AudioChunkEvent
    assert chunk.tenant_id == "tenant_alpha"
    assert chunk.call_id == "call_001"
    assert chunk.sequence_number == 1
    assert chunk.audio_bytes == payload
    assert chunk.chunk_duration_seconds == 2.0
    session.acknowledge_processed_chunk(resource=resource)

    receiver.process_bytes(control_record(RelayMessageType.END, 2))
    with pytest.raises(StopIteration):
        next(chunks)


def test_audio_before_start_fails_closed_and_releases_session() -> None:
    _resource, session, receiver = session_and_receiver()

    response = receiver.process_bytes(audio_record(0))[0]

    assert response_reason(response) == (
        RelayMessageType.ERROR,
        RelayReason.UNEXPECTED_MESSAGE_ORDER,
    )
    assert receiver.state is RelaySessionState.FAILED
    assert session.diagnostics.status is LocalMicrophoneStatus.FAILED
    assert not session.capability.active


def test_wrong_token_and_scope_fail_without_secret_leakage() -> None:
    for record, expected in (
        (
            start_record(token="synthetic-relay-token-99999999"),
            RelayReason.AUTHENTICATION_FAILED,
        ),
        (start_record(call_id="other_call"), RelayReason.SCOPE_MISMATCH),
        (start_record(tenant_id="other_tenant"), RelayReason.SCOPE_MISMATCH),
        (start_record(stream_id="other_stream"), RelayReason.SCOPE_MISMATCH),
    ):
        _resource, _session, receiver = session_and_receiver()
        response = receiver.process_bytes(record)[0]
        assert response_reason(response) == (RelayMessageType.ERROR, expected)
        assert TOKEN not in repr(receiver)
        assert TOKEN not in response.decode("latin1")


def test_pause_resume_preserves_call_and_requires_next_generation() -> None:
    _resource, session, receiver = session_and_receiver()
    receiver.process_bytes(start_record())

    paused = receiver.process_bytes(
        control_record(RelayMessageType.PAUSE, 1),
    )[0]
    resumed = receiver.process_bytes(
        control_record(RelayMessageType.RESUME, 2, generation=2),
    )[0]
    accepted = receiver.process_bytes(
        audio_record(3, generation=2, payload=b"\2\0"),
    )[0]

    assert response_reason(paused) == (RelayMessageType.ACK, RelayReason.PAUSED)
    assert response_reason(resumed) == (RelayMessageType.ACK, RelayReason.RESUMED)
    assert response_reason(accepted) == (
        RelayMessageType.ACK,
        RelayReason.AUDIO_ACCEPTED,
    )
    assert session.diagnostics.capture_generation == 2
    assert session.diagnostics.status is LocalMicrophoneStatus.STREAMING


@pytest.mark.parametrize("generation", [1, 3])
def test_resume_stale_or_gapped_generation_fails_closed(generation: int) -> None:
    _resource, _session, receiver = session_and_receiver()
    receiver.process_bytes(start_record())
    receiver.process_bytes(control_record(RelayMessageType.PAUSE, 1))

    response = receiver.process_bytes(
        control_record(RelayMessageType.RESUME, 2, generation=generation)
    )[0]

    assert response_reason(response)[0] is RelayMessageType.ERROR
    assert receiver.state is RelaySessionState.FAILED


def test_stale_audio_generation_fails_closed() -> None:
    _resource, _session, receiver = session_and_receiver()
    receiver.process_bytes(start_record())
    receiver.process_bytes(control_record(RelayMessageType.PAUSE, 1))
    receiver.process_bytes(control_record(RelayMessageType.RESUME, 2, generation=2))

    response = receiver.process_bytes(audio_record(3, generation=1))[0]

    assert response_reason(response) == (
        RelayMessageType.ERROR,
        RelayReason.STALE_GENERATION,
    )


def test_end_drains_already_admitted_audio_and_is_terminal() -> None:
    resource, session, receiver = session_and_receiver()
    payload = b"\1\0" * (LOCAL_MIC_CHUNK_BYTES // 2)
    receiver.process_bytes(start_record())
    receiver.process_bytes(audio_record(1, payload=payload))

    ended = receiver.process_bytes(control_record(RelayMessageType.END, 2))[0]

    assert response_reason(ended) == (RelayMessageType.ACK, RelayReason.ENDED)
    chunks = iter(session.iter_audio_chunks(cancellation=Event()))
    assert next(chunks).audio_bytes == payload
    session.acknowledge_processed_chunk(resource=resource)
    with pytest.raises(StopIteration):
        next(chunks)
    assert session.diagnostics.status is LocalMicrophoneStatus.COMPLETED
    terminal = receiver.process_bytes(audio_record(3))[0]
    assert response_reason(terminal) == (
        RelayMessageType.ERROR,
        RelayReason.TERMINAL_STATE,
    )
    assert session.diagnostics.status is LocalMicrophoneStatus.COMPLETED


def test_queue_overload_fails_closed_without_accepting_more_audio() -> None:
    _resource, session, receiver = session_and_receiver(max_queue_depth=1)
    payload = b"\1\0" * (LOCAL_MIC_CHUNK_BYTES // 2)
    receiver.process_bytes(start_record())
    receiver.process_bytes(audio_record(1, payload=payload))

    response = receiver.process_bytes(audio_record(2, payload=payload))[0]

    assert response_reason(response) == (
        RelayMessageType.ERROR,
        RelayReason.SESSION_REJECTED,
    )
    assert receiver.state is RelaySessionState.FAILED
    assert session.diagnostics.status is LocalMicrophoneStatus.OVERLOADED
    assert session.diagnostics.queue_depth == 0
    assert response_reason(receiver.process_bytes(audio_record(3))[0])[1] is (
        RelayReason.TERMINAL_STATE
    )


def test_malformed_record_releases_session_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resource, session, receiver = session_and_receiver()
    calls = 0
    original = session.close

    def close(
        reason: LocalMicrophoneTerminalReason = (
            LocalMicrophoneTerminalReason.RESOURCE_CLOSED
        ),
    ) -> None:
        nonlocal calls
        calls += 1
        original(reason)

    monkeypatch.setattr(session, "close", close)
    malformed = b"FAIL" + b"\0" * 10

    receiver.process_bytes(malformed)
    receiver.close()
    receiver.close()

    assert calls == 1
    assert receiver.state is RelaySessionState.FAILED


class TimeoutSocket:
    def __init__(self) -> None:
        self.closed = 0
        self.sent: list[bytes] = []

    def settimeout(self, timeout: float) -> None:
        assert 0 < timeout <= RELAY_IO_TIMEOUT_SECONDS

    def recv(self, size: int) -> bytes:
        assert size <= 4_096
        raise TimeoutError

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed += 1


def test_timeout_fails_closed_and_closes_connected_socket_once() -> None:
    _resource, session, receiver = session_and_receiver()
    client = TimeoutSocket()

    receiver.serve_connected_socket(client)  # type: ignore[arg-type]

    assert client.closed == 1
    assert response_reason(client.sent[0]) == (
        RelayMessageType.ERROR,
        RelayReason.IO_TIMEOUT,
    )
    assert receiver.state is RelaySessionState.FAILED
    assert session.diagnostics.status is LocalMicrophoneStatus.FAILED


def test_disconnect_before_end_releases_resources() -> None:
    _resource, session, receiver = session_and_receiver()
    server, client = socket.socketpair()
    client.close()

    receiver.serve_connected_socket(server)

    assert receiver.state is RelaySessionState.FAILED
    assert not receiver.client_active
    assert session.diagnostics.status is LocalMicrophoneStatus.DISCONNECTED


def test_second_simultaneous_client_is_rejected_without_worker_leak() -> None:
    _resource, _session, receiver = session_and_receiver()
    first_server, first_client = socket.socketpair()
    second_server, second_client = socket.socketpair()
    worker = Thread(
        target=receiver.serve_connected_socket,
        args=(first_server,),
        daemon=False,
    )
    worker.start()
    first_client.sendall(start_record())
    first_client.recv(4_096)

    receiver.serve_connected_socket(second_server)

    response = second_client.recv(4_096)
    assert response_reason(response) == (
        RelayMessageType.ERROR,
        RelayReason.CLIENT_ACTIVE,
    )
    first_client.sendall(control_record(RelayMessageType.END, 1))
    first_client.recv(4_096)
    first_client.close()
    second_client.close()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert not receiver.client_active


def test_invalid_initial_generation_fails_before_local_session_start() -> None:
    _resource, session, receiver = session_and_receiver()

    response = receiver.process_bytes(start_record(generation=2))[0]

    assert response_reason(response) == (
        RelayMessageType.ERROR,
        RelayReason.GENERATION_MISMATCH,
    )
    assert session.diagnostics.received_chunk_count == 0
    assert not session.capability.active


def test_receiver_repr_contains_no_token_scope_or_audio() -> None:
    _resource, _session, receiver = session_and_receiver()

    rendered = repr(receiver)

    assert TOKEN not in rendered
    assert "tenant_alpha" not in rendered
    assert "call_001" not in rendered
    assert "relay-stream" not in rendered
