from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from collections import deque
from threading import Condition, Event, Thread, get_ident
import socket
import sys
from typing import Any, cast

import pytest
from scripts import run_local_microphone_relay_client as relay_client_app

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
    MAX_RELAY_PROGRESS_TRANSITIONS,
    RELAY_FIRST_AUDIO_GRACE_TIMEOUT_SECONDS,
    RELAY_INITIAL_CLIENT_WAIT_TIMEOUT_SECONDS,
    RELAY_IO_TIMEOUT_SECONDS,
    RELAY_LOOPBACK_HOST,
    LocalhostMicrophoneRelayReceiver,
    MicrophoneRelayRecordParser,
    RelayClientProgressStage,
    RelayMessageType,
    RelayProgressHistory,
    RelayReason,
    RelayReceiverProgressStage,
    RelayResponseMetadata,
    RelaySessionState,
    encode_relay_record,
)
from app.events.models import AudioChunkEvent
from scripts.run_local_microphone_relay_client import (
    BoundedRelaySender,
    NativeRelayCapture,
    RelayClientConfig,
    RelayClientSession,
    RelayClientStatus,
    _relay_failure_message,
    retained_relay_client_session,
    reset_terminal_relay_client_session,
)
from live_dashboard.demo_data import tenant_demos
from live_dashboard.runtime_wiring import (
    DashboardExecutionIdentity,
    DashboardExecutionResource,
)
from live_dashboard.view_models import (
    DashboardExecutionMode,
    DashboardExecutionSnapshot,
    DashboardExecutionStage,
    DashboardExecutionStatus,
    create_local_execution,
    execution_snapshot,
)


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
    assert (
        RelayReceiverProgressStage.FIRST_AUDIO_RECEIVED not in receiver.progress_stages
    )
    assert response_reason(
        receiver.process_bytes(
            audio_record(1, payload=payload),
            arrived_at_utc=NOW + timedelta(milliseconds=50),
        )[0]
    ) == (RelayMessageType.ACK, RelayReason.AUDIO_ACCEPTED)
    assert receiver.progress_stages == (
        RelayReceiverProgressStage.RELAY_SESSION_CREATED,
        RelayReceiverProgressStage.START_RECEIVED,
        RelayReceiverProgressStage.START_VALIDATED,
        RelayReceiverProgressStage.FIRST_AUDIO_RECEIVED,
        RelayReceiverProgressStage.AUDIO_STREAMING,
    )

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
        assert receiver.last_failure_reason is expected
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


class PreStartSocket:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = deque(chunks)
        self.closed = 0
        self.sent: list[bytes] = []
        self.timeouts: list[float] = []

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def recv(self, size: int) -> bytes:
        assert size <= 4_096
        if self._chunks:
            return self._chunks.popleft()
        raise TimeoutError

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed += 1


def test_pre_start_timeout_records_recv_call_with_zero_received_bytes() -> None:
    _resource, _session, receiver = session_and_receiver()
    client = PreStartSocket(())

    receiver.serve_connected_socket(client)  # type: ignore[arg-type]

    diagnostics = receiver.pre_start_diagnostics
    assert diagnostics.recv_call_count_before_start == 1
    assert diagnostics.received_byte_count_before_start == 0
    assert diagnostics.parsed_record_count_before_start == 0
    assert receiver.last_failure_reason is RelayReason.IO_TIMEOUT
    assert client.timeouts == [RELAY_IO_TIMEOUT_SECONDS, RELAY_IO_TIMEOUT_SECONDS]


def test_partial_start_records_bytes_without_parsed_record_before_timeout() -> None:
    _resource, _session, receiver = session_and_receiver()
    partial_start = start_record()[:7]
    client = PreStartSocket((partial_start,))

    receiver.serve_connected_socket(client)  # type: ignore[arg-type]

    diagnostics = receiver.pre_start_diagnostics
    assert diagnostics.recv_call_count_before_start == 2
    assert diagnostics.received_byte_count_before_start == len(partial_start)
    assert diagnostics.parsed_record_count_before_start == 0
    assert receiver.last_failure_reason is RelayReason.IO_TIMEOUT
    assert client.timeouts == [RELAY_IO_TIMEOUT_SECONDS, RELAY_IO_TIMEOUT_SECONDS]


def test_complete_start_counts_one_record_then_stops_pre_start_counters() -> None:
    _resource, _session, receiver = session_and_receiver()
    complete_start = start_record()
    client = PreStartSocket((complete_start,))

    receiver.serve_connected_socket(client)  # type: ignore[arg-type]

    diagnostics = receiver.pre_start_diagnostics
    assert diagnostics.recv_call_count_before_start == 1
    assert diagnostics.received_byte_count_before_start == len(complete_start)
    assert diagnostics.parsed_record_count_before_start == 1
    assert receiver.start_validated
    assert response_reason(client.sent[0]) == (
        RelayMessageType.ACK,
        RelayReason.STARTED,
    )
    assert client.timeouts == [
        RELAY_IO_TIMEOUT_SECONDS,
        RELAY_FIRST_AUDIO_GRACE_TIMEOUT_SECONDS,
        RELAY_IO_TIMEOUT_SECONDS,
    ]
    assert TOKEN not in repr(diagnostics)


class ListenerTimeoutSocket:
    def __init__(self) -> None:
        self.timeout: float | None = None
        self.closed = 0

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def bind(self, address: tuple[str, int]) -> None:
        assert address == (RELAY_LOOPBACK_HOST, 0)

    def listen(self, backlog: int) -> None:
        assert backlog == 1

    def getsockname(self) -> tuple[str, int]:
        return RELAY_LOOPBACK_HOST, 18_765

    def close(self) -> None:
        self.closed += 1


def test_initial_client_wait_timeout_is_separate_from_connected_io_timeout() -> None:
    resource, session, _receiver = session_and_receiver()
    listener = ListenerTimeoutSocket()
    receiver = LocalhostMicrophoneRelayReceiver(
        session=session,
        resource=resource,
        expected_token=TOKEN,
        tenant_id="tenant_alpha",
        call_id="call_001",
        stream_id="relay-stream",
        resume_capability_factory=lambda: capability(resource),
        enabled=True,
        io_timeout_seconds=1.0,
        socket_factory=lambda *_args: listener,  # type: ignore[arg-type]
    )

    assert receiver.start() == (RELAY_LOOPBACK_HOST, 18_765)
    assert listener.timeout == RELAY_INITIAL_CLIENT_WAIT_TIMEOUT_SECONDS
    assert listener.timeout is not None
    assert listener.timeout > RELAY_IO_TIMEOUT_SECONDS
    receiver.close()


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
    assert receiver.last_failure_reason is RelayReason.IO_TIMEOUT
    assert session.diagnostics.status is LocalMicrophoneStatus.FAILED


class FirstAudioSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.received = deque(
            (
                start_record(),
                audio_record(1),
                control_record(RelayMessageType.END, 2),
            )
        )

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def recv(self, size: int) -> bytes:
        assert size <= 4_096
        return self.received.popleft()

    def sendall(self, data: bytes) -> None:
        assert response_reason(data)[0] is RelayMessageType.ACK

    def close(self) -> None:
        return None


class TimeoutThenClientListener:
    def __init__(self, client: FirstAudioSocket) -> None:
        self.client = client
        self.accept_count = 0
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        assert timeout == RELAY_INITIAL_CLIENT_WAIT_TIMEOUT_SECONDS

    def bind(self, address: tuple[str, int]) -> None:
        assert address == (RELAY_LOOPBACK_HOST, 0)

    def listen(self, backlog: int) -> None:
        assert backlog == 1

    def getsockname(self) -> tuple[str, int]:
        return RELAY_LOOPBACK_HOST, 18_765

    def accept(self) -> tuple[FirstAudioSocket, tuple[str, int]]:
        self.accept_count += 1
        if self.accept_count == 1:
            raise TimeoutError
        return self.client, (RELAY_LOOPBACK_HOST, 50_000)

    def close(self) -> None:
        self.closed = True


class CloseInterruptibleListener:
    def __init__(self) -> None:
        self.accept_entered = Event()
        self.closed = Event()

    def settimeout(self, timeout: float) -> None:
        assert timeout == RELAY_INITIAL_CLIENT_WAIT_TIMEOUT_SECONDS

    def bind(self, address: tuple[str, int]) -> None:
        assert address == (RELAY_LOOPBACK_HOST, 0)

    def listen(self, backlog: int) -> None:
        assert backlog == 1

    def getsockname(self) -> tuple[str, int]:
        return RELAY_LOOPBACK_HOST, 18_765

    def accept(self) -> tuple[FirstAudioSocket, tuple[str, int]]:
        self.accept_entered.set()
        assert self.closed.wait(timeout=1.0)
        raise OSError

    def close(self) -> None:
        self.closed.set()


def test_initial_accept_timeout_keeps_waiting_for_later_valid_client() -> None:
    resource = object()
    session = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        provider_stream_id="relay-stream",
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )
    client = FirstAudioSocket()
    listener = TimeoutThenClientListener(client)
    receiver = LocalhostMicrophoneRelayReceiver(
        session=session,
        resource=resource,
        expected_token=TOKEN,
        tenant_id="tenant_alpha",
        call_id="call_001",
        stream_id="relay-stream",
        resume_capability_factory=lambda: capability(resource),
        enabled=True,
        socket_factory=lambda *_args: listener,  # type: ignore[arg-type]
    )

    receiver.start_background()
    for _attempt in range(1_000):
        if not receiver.worker_active:
            break
        Event().wait(0.001)

    assert listener.accept_count == 2
    assert receiver.state is RelaySessionState.ENDED
    assert receiver.last_failure_reason is None
    assert session.diagnostics.received_chunk_count == 1


def test_close_interrupts_initial_accept_wait_without_worker_leak() -> None:
    resource = object()
    session = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        provider_stream_id="relay-stream",
    )
    listener = CloseInterruptibleListener()
    receiver = LocalhostMicrophoneRelayReceiver(
        session=session,
        resource=resource,
        expected_token=TOKEN,
        tenant_id="tenant_alpha",
        call_id="call_001",
        stream_id="relay-stream",
        resume_capability_factory=lambda: capability(resource),
        enabled=True,
        socket_factory=lambda *_args: listener,  # type: ignore[arg-type]
    )

    receiver.start_background()
    assert listener.accept_entered.wait(timeout=1.0)
    receiver.close()

    assert not receiver.worker_active
    assert receiver.state is RelaySessionState.AWAIT_START
    assert receiver.last_failure_reason is None


def test_first_audio_grace_is_separate_then_restores_normal_io_timeout() -> None:
    _resource, _session, receiver = session_and_receiver()
    client = FirstAudioSocket()

    receiver.serve_connected_socket(client)  # type: ignore[arg-type]

    assert RELAY_IO_TIMEOUT_SECONDS == 5.0
    assert client.timeouts == [
        RELAY_IO_TIMEOUT_SECONDS,
        RELAY_FIRST_AUDIO_GRACE_TIMEOUT_SECONDS,
        RELAY_IO_TIMEOUT_SECONDS,
    ]


def test_disconnect_before_end_releases_resources() -> None:
    _resource, session, receiver = session_and_receiver()
    server, client = socket.socketpair()
    client.close()

    receiver.serve_connected_socket(server)

    assert receiver.state is RelaySessionState.FAILED
    assert receiver.last_failure_reason is RelayReason.CONNECTION_CLOSED
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


def test_relay_audio_publishes_pre_end_dashboard_snapshot() -> None:
    resource = DashboardExecutionResource(
        DashboardExecutionIdentity("tenant_alpha", "call_001"),
        integration=None,
    )
    session = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        provider_stream_id="relay-stream",
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    session.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )
    resource.attach_microphone_session(session)
    receiver = LocalhostMicrophoneRelayReceiver(
        session=session,
        resource=resource,
        expected_token=TOKEN,
        tenant_id="tenant_alpha",
        call_id="call_001",
        stream_id="relay-stream",
        resume_capability_factory=lambda: capability(resource),
        enabled=True,
    )
    state = create_local_execution(tenant_demos()["tenant_alpha"], "call_001")
    initial = execution_snapshot(
        state,
        revision=0,
        lifecycle_status=DashboardExecutionStatus.RUNNING,
        execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
        execution_stage=DashboardExecutionStage.READY_TO_CAPTURE,
    )
    published_before_end = Event()

    def consume(
        cancellation: Event,
        publish: Callable[[DashboardExecutionSnapshot], None],
    ) -> DashboardExecutionSnapshot:
        chunks = iter(session.iter_audio_chunks(cancellation=cancellation))
        next(chunks)
        session.acknowledge_processed_chunk(resource=resource)
        state.current_chunk = 1
        partial = execution_snapshot(
            state,
            revision=1,
            lifecycle_status=DashboardExecutionStatus.RUNNING,
            execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
            execution_stage=DashboardExecutionStage.TRANSCRIPT_UPDATING,
        )
        publish(partial)
        published_before_end.set()
        with pytest.raises(StopIteration):
            next(chunks)
        return execution_snapshot(
            state,
            revision=2,
            lifecycle_status=DashboardExecutionStatus.COMPLETED,
            execution_mode=DashboardExecutionMode.LOCAL_MIC_TEST,
            execution_stage=DashboardExecutionStage.COMPLETED,
        )

    assert resource.start_worker(initial, consume)
    payload = b"\1\0" * (LOCAL_MIC_CHUNK_BYTES // 2)
    receiver.process_bytes(start_record())
    receiver.process_bytes(audio_record(1, payload=payload))

    assert published_before_end.wait(timeout=1.0)
    assert resource.latest_snapshot is not None
    assert resource.latest_snapshot.revision == 1
    assert resource.latest_snapshot.processed_chunks == 1
    assert receiver.state is RelaySessionState.STREAMING

    receiver.process_bytes(control_record(RelayMessageType.END, 2))
    resource.join_worker()
    assert resource.latest_snapshot is not None
    assert (
        resource.latest_snapshot.lifecycle_status is DashboardExecutionStatus.COMPLETED
    )
    resource.close()


class AckSocket:
    def __init__(
        self,
        *,
        error_type: RelayMessageType | None = None,
        disconnect_type: RelayMessageType | None = None,
        block_type: RelayMessageType | None = None,
    ) -> None:
        self._condition = Condition()
        self._responses: deque[bytes] = deque()
        self._release = Event()
        self.error_type = error_type
        self.disconnect_type = disconnect_type
        self.block_type = block_type
        self.records: list[object] = []
        self.send_threads: list[int] = []
        self.closed = 0
        self.connected_to: tuple[str, int] | None = None

    def settimeout(self, timeout: float) -> None:
        assert 0 < timeout <= RELAY_IO_TIMEOUT_SECONDS

    def connect(self, address: tuple[str, int]) -> None:
        self.connected_to = address

    def sendall(self, data: bytes) -> None:
        record = MicrophoneRelayRecordParser().feed(data)[0]
        with self._condition:
            self.records.append(record)
            self.send_threads.append(get_ident())
            response_type = (
                RelayMessageType.ERROR
                if record.message_type is self.error_type
                else RelayMessageType.ACK
            )
            reason = (
                RelayReason.SESSION_REJECTED
                if response_type is RelayMessageType.ERROR
                else RelayReason.AUDIO_ACCEPTED
            )
            self._responses.append(
                encode_relay_record(
                    response_type,
                    {
                        "sequence_number": record.sequence_number,
                        "reason": reason.value,
                    },
                )
            )
            self._condition.notify_all()

    def recv(self, size: int) -> bytes:
        assert size <= 4_096
        with self._condition:
            latest = self.records[-1]
        message_type = getattr(latest, "message_type")
        if message_type is self.disconnect_type:
            return b""
        if message_type is self.block_type:
            assert self._release.wait(timeout=1.0)
        with self._condition:
            return self._responses.popleft()

    def close(self) -> None:
        self.closed += 1
        self._release.set()

    def wait_for_records(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: len(self.records) >= count,
                timeout=1.0,
            )

    def release(self) -> None:
        self._release.set()


class FakeNativeStream:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


def native_capture(sender: BoundedRelaySender) -> NativeRelayCapture:
    return NativeRelayCapture(
        sender, stream_factory=lambda _callback: FakeNativeStream()
    )


def client_config() -> RelayClientConfig:
    return RelayClientConfig(
        tenant_id="tenant_alpha",
        call_id="call_001",
        stream_id="relay-stream",
        token=TOKEN,
    )


def wait_for_client_status(
    sender: BoundedRelaySender,
    status: RelayClientStatus,
) -> None:
    for _attempt in range(1_000):
        if sender.diagnostics.status is status:
            return
        Event().wait(0.001)
    pytest.fail(f"client status did not reach {status.value}")


def test_local_sender_encodes_start_audio_pause_resume_and_drained_end() -> None:
    connection = AckSocket()
    sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )

    assert sender.start()
    connection.wait_for_records(1)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert sender.enqueue_audio(b"\1\0", captured_at_utc=NOW)
    assert sender.pause()
    connection.wait_for_records(3)
    wait_for_client_status(sender, RelayClientStatus.PAUSED)
    assert sender.resume()
    connection.wait_for_records(4)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert sender.enqueue_audio(b"\2\0", captured_at_utc=NOW)
    assert sender.end()
    connection.wait_for_records(6)
    wait_for_client_status(sender, RelayClientStatus.ENDED)

    message_types = [getattr(record, "message_type") for record in connection.records]
    sequences = [getattr(record, "sequence_number") for record in connection.records]
    assert message_types == [
        RelayMessageType.START,
        RelayMessageType.AUDIO,
        RelayMessageType.PAUSE,
        RelayMessageType.RESUME,
        RelayMessageType.AUDIO,
        RelayMessageType.END,
    ]
    assert sequences == list(range(6))
    assert sender.diagnostics.generation == 2
    assert sender.diagnostics.sent_chunk_count == 2
    assert sender.diagnostics.acknowledged_chunk_count == 2
    sender.close()
    sender.close()
    assert connection.closed == 1
    assert sender.progress_stages == (
        RelayClientProgressStage.TCP_CONNECTING,
        RelayClientProgressStage.TCP_CONNECTED,
        RelayClientProgressStage.START_SENT,
        RelayClientProgressStage.START_ACKNOWLEDGED,
        RelayClientProgressStage.FIRST_AUDIO_CHUNK_SENT,
        RelayClientProgressStage.FIRST_AUDIO_CHUNK_ACKNOWLEDGED,
        RelayClientProgressStage.STREAMING,
        RelayClientProgressStage.PAUSED,
        RelayClientProgressStage.RESUMED,
        RelayClientProgressStage.ENDED,
    )


def test_progress_history_is_deduplicated_bounded_and_fixed() -> None:
    history = RelayProgressHistory()
    stages = tuple(RelayClientProgressStage) + tuple(RelayReceiverProgressStage)
    history.record(RelayClientProgressStage.TCP_CONNECTING)
    history.record(RelayClientProgressStage.TCP_CONNECTING)
    for stage in stages:
        history.record(stage)

    assert len(history.stages) == MAX_RELAY_PROGRESS_TRANSITIONS
    assert history.stages == stages[-MAX_RELAY_PROGRESS_TRANSITIONS:]


def test_local_sender_queue_overload_fails_closed_without_dropping() -> None:
    connection = AckSocket(block_type=RelayMessageType.AUDIO)
    sender = BoundedRelaySender(
        client_config(),
        queue_depth=1,
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )
    sender.start()
    connection.wait_for_records(1)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)

    assert sender.enqueue_audio(b"\1\0", captured_at_utc=NOW)
    connection.wait_for_records(2)
    assert sender.enqueue_audio(b"\2\0", captured_at_utc=NOW)
    assert not sender.enqueue_audio(b"\3\0", captured_at_utc=NOW)

    assert sender.diagnostics.status is RelayClientStatus.FAILED
    assert sender.diagnostics.failure_reason is RelayReason.BUFFER_LIMIT
    assert connection.closed == 0
    connection.release()
    sender.close()
    assert not sender.worker_active


def test_remote_error_and_tunnel_disconnect_are_fixed_failures() -> None:
    for connection, expected in (
        (
            AckSocket(error_type=RelayMessageType.AUDIO),
            RelayReason.SESSION_REJECTED,
        ),
        (
            AckSocket(disconnect_type=RelayMessageType.AUDIO),
            RelayReason.CONNECTION_CLOSED,
        ),
    ):
        sender = BoundedRelaySender(
            client_config(),
            socket_factory=lambda *_args, selected=connection: selected,  # type: ignore[arg-type]
        )
        sender.start()
        connection.wait_for_records(1)
        wait_for_client_status(sender, RelayClientStatus.STREAMING)
        sender.enqueue_audio(b"\1\0", captured_at_utc=NOW)
        connection.wait_for_records(2)
        wait_for_client_status(sender, RelayClientStatus.FAILED)
        assert sender.diagnostics.failure_reason is expected
        sender.close()


def test_terminal_client_session_can_reset_and_retry_without_restart() -> None:
    failed_sender = BoundedRelaySender(client_config())
    failed_sender.fail(RelayReason.CONNECTION_CLOSED)
    failed_session = RelayClientSession(
        sender=failed_sender,
        capture=native_capture(failed_sender),
    )
    session_state: dict[str, object] = {
        "relay_client_session": failed_session,
    }

    assert reset_terminal_relay_client_session(session_state)
    assert "relay_client_session" not in session_state

    retry_connection = AckSocket()
    retry_sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: retry_connection,  # type: ignore[arg-type]
    )
    retry_session = RelayClientSession(
        sender=retry_sender,
        capture=native_capture(retry_sender),
    )
    session_state["relay_client_session"] = retry_session

    assert retry_sender.start()
    retry_connection.wait_for_records(1)
    wait_for_client_status(retry_sender, RelayClientStatus.STREAMING)
    assert not reset_terminal_relay_client_session(session_state)
    assert retry_sender.end()
    retry_connection.wait_for_records(2)
    wait_for_client_status(retry_sender, RelayClientStatus.ENDED)
    assert reset_terminal_relay_client_session(session_state)
    assert "relay_client_session" not in session_state


def test_streamlit_rerun_recognizes_session_and_renders_without_second_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = AckSocket(block_type=RelayMessageType.START)
    sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )
    stream = FakeNativeStream()
    capture = NativeRelayCapture(sender, stream_factory=lambda _callback: stream)
    retained = RelayClientSession(
        sender=sender,
        capture=capture,
    )
    assert sender.start()
    connection.wait_for_records(1)
    assert sender.diagnostics.status is RelayClientStatus.CONNECTING
    sender.record_progress(RelayClientProgressStage.CONFIGURATION_VALIDATED)

    class RerunRelayClientSession(relay_client_app.RelayClientSessionHandle):
        pass

    assert not isinstance(retained, RerunRelayClientSession)
    monkeypatch.setattr(
        relay_client_app,
        "RelayClientSession",
        RerunRelayClientSession,
    )
    assert retained_relay_client_session(retained) is retained

    class Column:
        def button(self, _label: str, **_kwargs: object) -> bool:
            return False

    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {"relay_client_session": retained}
            self.metrics: list[tuple[str, str]] = []
            self.infos: list[str] = []
            self.captions: list[str] = []
            self.connect_disabled: bool | None = None
            self.start_microphone_clicks = 0
            self.reruns = 0

        def set_page_config(self, **_kwargs: object) -> None:
            return None

        def get_option(self, _name: str) -> str:
            return RELAY_LOOPBACK_HOST

        def title(self, _value: str) -> None:
            return None

        def caption(self, value: str) -> None:
            self.captions.append(value)

        def text_input(self, _label: str, **_kwargs: object) -> str:
            return ""

        def number_input(self, _label: str, *_args: int) -> int:
            return 18_765

        def button(self, label: str, **kwargs: object) -> bool:
            if label == "Connect / Start":
                self.connect_disabled = bool(kwargs["disabled"])
            if (
                label == "Mikrofonu Başlat"
                and not bool(kwargs["disabled"])
                and self.start_microphone_clicks == 0
            ):
                self.start_microphone_clicks += 1
                return True
            return False

        def rerun(self) -> None:
            self.reruns += 1

        def info(self, value: str) -> None:
            self.infos.append(value)

        def metric(self, label: str, value: object) -> None:
            self.metrics.append((label, str(value)))

        def error(self, _value: str) -> None:
            return None

        def columns(self, count: int) -> list[Column]:
            return [Column() for _index in range(count)]

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)  # type: ignore[arg-type]

    relay_client_app.render()
    relay_client_app.render()

    assert fake_streamlit.connect_disabled is True
    assert not fake_streamlit.infos
    assert ("Durum", RelayClientStatus.CONNECTING.value) in fake_streamlit.metrics
    assert any("Yapılandırma doğrulandı" in value for value in fake_streamlit.captions)
    assert len(connection.records) == 1
    connection.release()
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    relay_client_app.render()
    relay_client_app.render()
    for _attempt in range(1_000):
        if stream.started == 1:
            break
        Event().wait(0.001)
    assert stream.started == 1
    assert fake_streamlit.start_microphone_clicks == 1
    assert len(connection.records) == 1
    retained.close()


def test_native_capture_opens_after_start_ack_and_callback_queues_audio() -> None:
    connection = AckSocket(block_type=RelayMessageType.START)
    sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )
    stream = FakeNativeStream()
    callbacks: list[Callable[[object, int, object, object], None]] = []
    capture = NativeRelayCapture(
        sender,
        stream_factory=lambda callback: callbacks.append(callback) or stream,
    )
    assert sender.start()
    connection.wait_for_records(1)
    assert sender.diagnostics.status is RelayClientStatus.CONNECTING
    assert stream.started == 0
    assert not callbacks
    assert len(connection.records) == 1

    connection.release()
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert not capture.started
    assert capture.start()
    for _attempt in range(1_000):
        if callbacks and stream.started == 1:
            break
        Event().wait(0.001)
    assert RelayClientProgressStage.START_ACKNOWLEDGED in sender.progress_stages
    assert capture.opened
    assert capture.device_label is not None
    assert RelayClientProgressStage.NATIVE_MICROPHONE_OPENED in sender.progress_stages
    assert (
        sender.progress_stages.count(RelayClientProgressStage.NATIVE_MICROPHONE_OPENED)
        == 1
    )
    assert (
        RelayClientProgressStage.FIRST_NATIVE_AUDIO_BLOCK_RECEIVED
        not in sender.progress_stages
    )
    callback_thread = get_ident()
    callbacks[0](b"\1\0" * 32_000, 32_000, object(), object())
    connection.wait_for_records(2)
    assert callback_thread not in connection.send_threads
    assert getattr(connection.records[1], "message_type") is RelayMessageType.AUDIO
    assert len(getattr(connection.records[1], "payload")) == LOCAL_MIC_CHUNK_BYTES
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert sender.diagnostics.sent_chunk_count == 1
    assert sender.diagnostics.acknowledged_chunk_count == 1
    assert (
        RelayClientProgressStage.FIRST_NATIVE_AUDIO_BLOCK_RECEIVED
        in sender.progress_stages
    )
    capture.close()
    assert not capture.opened
    sender.close()


@pytest.mark.parametrize(
    ("factory", "reason"),
    [
        (
            lambda _callback: (_ for _ in ()).throw(LookupError()),
            RelayReason.MICROPHONE_UNAVAILABLE,
        ),
        (
            lambda _callback: (_ for _ in ()).throw(
                RuntimeError("raw PortAudio device failure")
            ),
            RelayReason.MICROPHONE_OPEN_FAILED,
        ),
    ],
)
def test_native_microphone_open_failures_are_sanitized(
    factory: Callable[[Callable[[object, int, object, object], None]], object],
    reason: RelayReason,
) -> None:
    connection = AckSocket()
    sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )
    capture = NativeRelayCapture(sender, stream_factory=cast(Any, factory))

    sender.start()
    connection.wait_for_records(1)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert capture.start()
    wait_for_client_status(sender, RelayClientStatus.FAILED)

    assert sender.diagnostics.failure_reason is reason
    assert TOKEN not in repr(capture)
    assert "raw PortAudio device failure" not in repr(sender.diagnostics)
    rendered = _relay_failure_message(reason)
    assert rendered == f"Relay başarısız: {reason.value}"
    assert "raw PortAudio device failure" not in rendered
    capture.close()


def test_native_capture_pause_resume_end_and_cleanup_use_existing_generation() -> None:
    connection = AckSocket()
    sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )
    stream = FakeNativeStream()
    callbacks: list[Callable[[object, int, object, object], None]] = []
    capture = NativeRelayCapture(
        sender,
        stream_factory=lambda callback: callbacks.append(callback) or stream,
    )
    sender.set_resume_callback(capture.resume)
    session = RelayClientSession(sender=sender, capture=capture)

    sender.start()
    connection.wait_for_records(1)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert capture.start()
    for _attempt in range(1_000):
        if callbacks and stream.started == 1:
            break
        Event().wait(0.001)
    callbacks[0](b"\1\0" * 32_000, 32_000, object(), object())
    connection.wait_for_records(2)
    assert session.pause()
    connection.wait_for_records(3)
    wait_for_client_status(sender, RelayClientStatus.PAUSED)
    assert stream.stopped == 1
    assert session.resume()
    connection.wait_for_records(4)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert sender.diagnostics.generation == 2
    assert stream.started == 2
    assert (
        sender.progress_stages.count(RelayClientProgressStage.NATIVE_MICROPHONE_OPENED)
        == 1
    )
    callbacks[0](b"\2\0" * 800, 800, object(), object())
    Event().wait(0.05)
    assert session.end()
    assert not capture.opened
    assert stream.closed == 1
    connection.wait_for_records(6)
    wait_for_client_status(sender, RelayClientStatus.ENDED)
    session.close()

    assert stream.closed == 1
    assert stream.stopped == 2
    assert not capture.worker_active
    assert getattr(connection.records[4], "message_type") is RelayMessageType.AUDIO
    assert len(getattr(connection.records[4], "payload")) == 1_600


def test_native_callback_queue_overflow_fails_safely() -> None:
    connection = AckSocket(block_type=RelayMessageType.AUDIO)
    sender = BoundedRelaySender(
        client_config(),
        socket_factory=lambda *_args: connection,  # type: ignore[arg-type]
    )
    stream = FakeNativeStream()
    callbacks: list[Callable[[object, int, object, object], None]] = []
    capture = NativeRelayCapture(
        sender,
        stream_factory=lambda callback: callbacks.append(callback) or stream,
        queue_depth=1,
    )
    sender.start()
    connection.wait_for_records(1)
    wait_for_client_status(sender, RelayClientStatus.STREAMING)
    assert capture.start()
    for _attempt in range(1_000):
        if callbacks and stream.started == 1:
            break
        Event().wait(0.001)

    for _block in range(100):
        callbacks[0](b"\1\0" * 16_000, 16_000, object(), object())
    wait_for_client_status(sender, RelayClientStatus.FAILED)

    assert sender.diagnostics.failure_reason is RelayReason.BUFFER_LIMIT
    connection.release()
    capture.close()
    sender.close()


def test_local_client_configuration_and_repr_never_expose_token() -> None:
    config = client_config()
    sender = BoundedRelaySender(config)

    assert TOKEN not in repr(config)
    assert TOKEN not in repr(sender)
    with pytest.raises(ValueError, match=RelayReason.INVALID_BIND.value):
        RelayClientConfig(
            tenant_id="tenant_alpha",
            call_id="call_001",
            stream_id="relay-stream",
            token=TOKEN,
            host="0.0.0.0",
        )
