"""Localhost-only Streamlit capture client for the SSH microphone relay.

Start with:

    uv run streamlit run scripts/run_local_microphone_relay_client.py \
      --server.address 127.0.0.1 \
      --server.port 8503
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
import socket
import sys
from threading import Lock, Thread, current_thread
from typing import Callable, Final

import av


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio_ingress.local_microphone import (  # noqa: E402
    LOCAL_MIC_CHUNK_BYTES,
    PyAVLocalMicrophoneNormalizer,
)
from app.audio_ingress.tcp_microphone_relay import (  # noqa: E402
    MAX_RELAY_RECORD_BYTES,
    RELAY_IO_TIMEOUT_SECONDS,
    RELAY_LOOPBACK_HOST,
    MicrophoneRelayRecordParser,
    RelayEndReason,
    RelayMessageType,
    RelayReason,
    RelayResponseMetadata,
    encode_relay_record,
)
from live_dashboard.local_microphone import microphone_webrtc_streamer  # noqa: E402


DEFAULT_RELAY_PORT: Final = 18_765
LOCAL_RELAY_SENDER_QUEUE_DEPTH: Final = 4
CLIENT_START_COMMAND: Final = (
    "uv run streamlit run scripts/run_local_microphone_relay_client.py "
    "--server.address 127.0.0.1 --server.port 8503"
)


class RelayClientStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    ENDED = "ended"
    FAILED = "failed"


class _RelayClientError(RuntimeError):
    def __init__(self, reason: RelayReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class RelayClientConfig:
    tenant_id: str
    call_id: str
    stream_id: str
    token: str = field(repr=False)
    host: str = RELAY_LOOPBACK_HOST
    port: int = DEFAULT_RELAY_PORT

    def __post_init__(self) -> None:
        if self.host != RELAY_LOOPBACK_HOST:
            raise ValueError(RelayReason.INVALID_BIND.value)
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ValueError(RelayReason.INVALID_BIND.value)
        encode_relay_record(
            RelayMessageType.START,
            {
                "token": self.token,
                "tenant_id": self.tenant_id,
                "call_id": self.call_id,
                "stream_id": self.stream_id,
                "sequence_number": 0,
                "generation": 1,
                "codec_name": "pcm_s16le",
                "sample_rate_hz": 16_000,
                "channel_count": 1,
            },
        )


@dataclass(frozen=True, slots=True)
class RelayClientDiagnostics:
    status: RelayClientStatus
    generation: int
    sent_chunk_count: int
    acknowledged_chunk_count: int
    queue_depth: int
    failure_reason: RelayReason | None


@dataclass(frozen=True, slots=True)
class _OutboundRecord:
    message_type: RelayMessageType
    sequence_number: int
    generation: int
    captured_at_utc: datetime | None = None
    payload: bytes = field(default=b"", repr=False)


class BoundedRelaySender:
    """Serialize one exact-scope relay session on a bounded worker queue."""

    def __init__(
        self,
        config: RelayClientConfig,
        *,
        queue_depth: int = LOCAL_RELAY_SENDER_QUEUE_DEPTH,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        if type(queue_depth) is not int or queue_depth <= 0:
            raise ValueError(RelayReason.BUFFER_LIMIT.value)
        self._config = config
        self._queue: Queue[_OutboundRecord] = Queue(maxsize=queue_depth)
        self._socket_factory = socket_factory
        self._socket: socket.socket | None = None
        self._worker: Thread | None = None
        self._lock = Lock()
        self._status = RelayClientStatus.DISCONNECTED
        self._generation = 1
        self._next_sequence = 1
        self._sent_chunks = 0
        self._acknowledged_chunks = 0
        self._failure_reason: RelayReason | None = None
        self._closed = False

    @property
    def diagnostics(self) -> RelayClientDiagnostics:
        with self._lock:
            return RelayClientDiagnostics(
                status=self._status,
                generation=self._generation,
                sent_chunk_count=self._sent_chunks,
                acknowledged_chunk_count=self._acknowledged_chunks,
                queue_depth=self._queue.qsize(),
                failure_reason=self._failure_reason,
            )

    @property
    def worker_active(self) -> bool:
        worker = self._worker
        return worker is not None and worker.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self._worker is not None or self._closed:
                return False
            self._status = RelayClientStatus.CONNECTING
            worker = Thread(
                target=self._run,
                name="local-microphone-relay-sender",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return True

    def enqueue_audio(self, payload: bytes, *, captured_at_utc: datetime) -> bool:
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > LOCAL_MIC_CHUNK_BYTES
            or len(payload) % 2
        ):
            self._fail(RelayReason.INVALID_FORMAT, close_connection=False)
            return False
        with self._lock:
            if self._status is not RelayClientStatus.STREAMING:
                return False
            record = _OutboundRecord(
                message_type=RelayMessageType.AUDIO,
                sequence_number=self._next_sequence,
                generation=self._generation,
                captured_at_utc=captured_at_utc,
                payload=payload,
            )
            self._next_sequence += 1
        return self._enqueue(record)

    def pause(self) -> bool:
        with self._lock:
            if self._status is not RelayClientStatus.STREAMING:
                return False
            record = _OutboundRecord(
                RelayMessageType.PAUSE,
                self._next_sequence,
                self._generation,
            )
            self._next_sequence += 1
        return self._enqueue(record)

    def resume(self) -> bool:
        with self._lock:
            if self._status is not RelayClientStatus.PAUSED:
                return False
            generation = self._generation + 1
            record = _OutboundRecord(
                RelayMessageType.RESUME,
                self._next_sequence,
                generation,
            )
            self._next_sequence += 1
        return self._enqueue(record)

    def end(self) -> bool:
        with self._lock:
            if self._status not in {
                RelayClientStatus.STREAMING,
                RelayClientStatus.PAUSED,
            }:
                return False
            record = _OutboundRecord(
                RelayMessageType.END,
                self._next_sequence,
                self._generation,
            )
            self._next_sequence += 1
        return self._enqueue(record)

    def close(self) -> None:
        with self._lock:
            connection, self._socket = self._socket, None
            if not self._closed:
                self._closed = True
                if self._status not in {
                    RelayClientStatus.ENDED,
                    RelayClientStatus.FAILED,
                }:
                    self._status = RelayClientStatus.DISCONNECTED
        if connection is not None:
            connection.close()
        worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join(timeout=RELAY_IO_TIMEOUT_SECONDS)

    def _enqueue(self, record: _OutboundRecord) -> bool:
        try:
            self._queue.put_nowait(record)
        except Full:
            self._fail(RelayReason.BUFFER_LIMIT, close_connection=False)
            return False
        return True

    def _run(self) -> None:
        connection = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            connection.settimeout(RELAY_IO_TIMEOUT_SECONDS)
            connection.connect((self._config.host, self._config.port))
            with self._lock:
                if self._closed:
                    return
                self._socket = connection
            self._exchange(
                connection,
                RelayMessageType.START,
                {
                    "token": self._config.token,
                    "tenant_id": self._config.tenant_id,
                    "call_id": self._config.call_id,
                    "stream_id": self._config.stream_id,
                    "sequence_number": 0,
                    "generation": 1,
                    "codec_name": "pcm_s16le",
                    "sample_rate_hz": 16_000,
                    "channel_count": 1,
                },
            )
            with self._lock:
                self._status = RelayClientStatus.STREAMING
            while True:
                try:
                    outbound = self._queue.get(timeout=0.25)
                except Empty:
                    with self._lock:
                        if self._closed:
                            return
                    continue
                with self._lock:
                    if self._closed:
                        return
                self._send_outbound(connection, outbound)
                if outbound.message_type is RelayMessageType.END:
                    with self._lock:
                        self._status = RelayClientStatus.ENDED
                    return
        except _RelayClientError as error:
            self._fail(error.reason)
        except Exception:
            self._fail(RelayReason.CONNECTION_CLOSED)
        finally:
            connection.close()
            with self._lock:
                if self._socket is connection:
                    self._socket = None

    def _send_outbound(
        self,
        connection: socket.socket,
        outbound: _OutboundRecord,
    ) -> None:
        metadata: dict[str, object] = {
            "sequence_number": outbound.sequence_number,
            "generation": outbound.generation,
        }
        if outbound.message_type is RelayMessageType.AUDIO:
            assert outbound.captured_at_utc is not None
            metadata.update(
                {
                    "sample_count": len(outbound.payload) // 2,
                    "captured_at_utc": outbound.captured_at_utc.isoformat(),
                }
            )
        elif outbound.message_type is RelayMessageType.END:
            metadata["end_reason"] = RelayEndReason.COMPLETED.value
        self._exchange(
            connection,
            outbound.message_type,
            metadata,
            outbound.payload,
        )
        with self._lock:
            if outbound.message_type is RelayMessageType.AUDIO:
                self._sent_chunks += 1
                self._acknowledged_chunks += 1
            elif outbound.message_type is RelayMessageType.PAUSE:
                self._status = RelayClientStatus.PAUSED
            elif outbound.message_type is RelayMessageType.RESUME:
                self._generation = outbound.generation
                self._status = RelayClientStatus.STREAMING

    @staticmethod
    def _exchange(
        connection: socket.socket,
        message_type: RelayMessageType,
        metadata: dict[str, object],
        payload: bytes = b"",
    ) -> None:
        raw_sequence = metadata["sequence_number"]
        if type(raw_sequence) is not int:
            raise _RelayClientError(RelayReason.INVALID_METADATA)
        sequence = raw_sequence
        connection.sendall(encode_relay_record(message_type, metadata, payload))
        parser = MicrophoneRelayRecordParser()
        received = 0
        while received <= MAX_RELAY_RECORD_BYTES:
            data = connection.recv(4_096)
            if not data:
                raise _RelayClientError(RelayReason.CONNECTION_CLOSED)
            received += len(data)
            records = parser.feed(data)
            if not records:
                continue
            if len(records) != 1:
                raise _RelayClientError(RelayReason.INVALID_METADATA)
            response = records[0]
            response_metadata = response.metadata
            if (
                not isinstance(response_metadata, RelayResponseMetadata)
                or response_metadata.sequence_number != sequence
            ):
                raise _RelayClientError(RelayReason.SCOPE_MISMATCH)
            if response.message_type is RelayMessageType.ERROR:
                raise _RelayClientError(response_metadata.reason)
            if response.message_type is not RelayMessageType.ACK:
                raise _RelayClientError(RelayReason.UNEXPECTED_MESSAGE_ORDER)
            return
        raise _RelayClientError(RelayReason.BUFFER_LIMIT)

    def fail(self, reason: RelayReason) -> None:
        self._fail(reason)

    def _fail(
        self,
        reason: RelayReason,
        *,
        close_connection: bool = True,
    ) -> None:
        with self._lock:
            if self._status in {RelayClientStatus.ENDED, RelayClientStatus.FAILED}:
                return
            self._status = RelayClientStatus.FAILED
            self._failure_reason = reason
            self._closed = True
            connection = self._socket if close_connection else None
            if close_connection:
                self._socket = None
        if connection is not None:
            connection.close()


@dataclass(frozen=True, slots=True)
class _CaptureDiagnostics:
    capture_generation: int


class RelayCaptureSession:
    """WebRTC callback adapter restricted to normalization and bounded enqueue."""

    def __init__(self, sender: BoundedRelaySender) -> None:
        self._sender = sender
        self._normalizer = PyAVLocalMicrophoneNormalizer()
        self._buffer = bytearray()
        self._lock = Lock()

    @property
    def diagnostics(self) -> _CaptureDiagnostics:
        return _CaptureDiagnostics(self._sender.diagnostics.generation)

    def accept_frame(
        self,
        frame: av.AudioFrame,
        *,
        capture_generation: int | None = None,
    ) -> av.AudioFrame:
        with self._lock:
            if (
                capture_generation is not None
                and capture_generation != self._sender.diagnostics.generation
            ):
                raise PermissionError(RelayReason.STALE_GENERATION.value)
            for item in self._normalizer.normalize(frame):
                self._append(item.pcm_s16le)
        return frame

    def mark_reconnecting(self, *, capture_generation: int | None = None) -> bool:
        del capture_generation
        self._sender.fail(RelayReason.CONNECTION_CLOSED)
        return True

    def flush(self) -> bool:
        with self._lock:
            for item in self._normalizer.flush():
                self._append(item.pcm_s16le)
            if not self._buffer:
                return True
            payload = bytes(self._buffer)
            self._buffer.clear()
            return self._sender.enqueue_audio(
                payload,
                captured_at_utc=datetime.now(UTC),
            )

    def _append(self, pcm_s16le: bytes) -> None:
        offset = 0
        while offset < len(pcm_s16le):
            remaining = LOCAL_MIC_CHUNK_BYTES - len(self._buffer)
            take = min(remaining, len(pcm_s16le) - offset)
            self._buffer.extend(pcm_s16le[offset : offset + take])
            offset += take
            if len(self._buffer) == LOCAL_MIC_CHUNK_BYTES:
                payload = bytes(self._buffer)
                self._buffer.clear()
                if not self._sender.enqueue_audio(
                    payload,
                    captured_at_utc=datetime.now(UTC),
                ):
                    raise RuntimeError(RelayReason.BUFFER_LIMIT.value)


@dataclass(slots=True)
class RelayClientSession:
    sender: BoundedRelaySender
    capture: RelayCaptureSession

    def pause(self) -> bool:
        return self.capture.flush() and self.sender.pause()

    def resume(self) -> bool:
        return self.sender.resume()

    def end(self) -> bool:
        return self.capture.flush() and self.sender.end()

    def close(self) -> None:
        self.sender.close()


def create_relay_client_session(config: RelayClientConfig) -> RelayClientSession:
    sender = BoundedRelaySender(config)
    return RelayClientSession(sender=sender, capture=RelayCaptureSession(sender))


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="SSH Mikrofon Relay", page_icon="🎙️")
    configured_address = st.get_option("server.address")
    if configured_address != RELAY_LOOPBACK_HOST:
        st.error("Bu geliştirme aracı yalnızca 127.0.0.1 üzerinde çalışır.")
        st.stop()
    st.title("SSH Mikrofon Relay — Geliştirme Testi")
    st.caption(CLIENT_START_COMMAND)
    st.text_input("Relay host", value=RELAY_LOOPBACK_HOST, disabled=True)
    port = int(st.number_input("Relay port", 1, 65_535, DEFAULT_RELAY_PORT))
    tenant_id = st.text_input("tenant_id")
    call_id = st.text_input("call_id")
    stream_id = st.text_input("stream_id")
    token = st.text_input("Ephemeral token", type="password")
    session = st.session_state.get("relay_client_session")
    if st.button("Connect / Start", disabled=session is not None):
        try:
            config = RelayClientConfig(
                tenant_id=tenant_id,
                call_id=call_id,
                stream_id=stream_id,
                token=token,
                port=port,
            )
            session = create_relay_client_session(config)
            st.session_state.relay_client_session = session
            session.sender.start()
            st.rerun()
        except Exception:
            st.error("Relay oturumu güvenli biçimde başlatılamadı.")
    if not isinstance(session, RelayClientSession):
        st.info("Bağlantı bilgilerini girip Connect / Start seçin.")
        return
    diagnostics = session.sender.diagnostics
    st.metric("Durum", diagnostics.status.value)
    st.metric("Generation", diagnostics.generation)
    st.metric("Gönderilen parça", diagnostics.sent_chunk_count)
    st.metric("Onaylanan parça", diagnostics.acknowledged_chunk_count)
    st.metric("Kuyruk", diagnostics.queue_depth)
    if diagnostics.failure_reason is not None:
        st.error(f"Relay başarısız: {diagnostics.failure_reason.value}")
    if diagnostics.status is RelayClientStatus.STREAMING:
        microphone_webrtc_streamer(
            session=session.capture,  # type: ignore[arg-type]
            key="ssh-microphone-relay-capture",
            desired_playing_state=True,
        )
    pause, resume, end = st.columns(3)
    if pause.button(
        "Pause",
        disabled=diagnostics.status is not RelayClientStatus.STREAMING,
    ):
        session.pause()
        st.rerun()
    if resume.button(
        "Resume",
        disabled=diagnostics.status is not RelayClientStatus.PAUSED,
    ):
        session.resume()
        st.rerun()
    if end.button(
        "End",
        disabled=diagnostics.status
        not in {RelayClientStatus.STREAMING, RelayClientStatus.PAUSED},
    ):
        session.end()
        st.rerun()


if __name__ == "__main__":
    render()
