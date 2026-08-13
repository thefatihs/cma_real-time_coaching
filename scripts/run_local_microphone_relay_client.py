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
from threading import Event, Lock, Thread, current_thread
from typing import Any, Callable, Final, MutableMapping, Protocol, cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio_ingress.local_microphone import LOCAL_MIC_CHUNK_BYTES  # noqa: E402
from app.audio_ingress.tcp_microphone_relay import (  # noqa: E402
    MAX_RELAY_RECORD_BYTES,
    RELAY_IO_TIMEOUT_SECONDS,
    RELAY_LOOPBACK_HOST,
    MicrophoneRelayRecordParser,
    RelayClientProgressStage,
    RelayEndReason,
    RelayClientSessionHandle,
    RelayMessageType,
    RelayReason,
    RelayResponseMetadata,
    RelayProgressHistory,
    RelayProgressStage,
    encode_relay_record,
)

DEFAULT_RELAY_PORT: Final = 18_765
LOCAL_RELAY_SENDER_QUEUE_DEPTH: Final = 4
LOCAL_NATIVE_CAPTURE_QUEUE_DEPTH: Final = 8
CLIENT_START_COMMAND: Final = (
    "uv run streamlit run scripts/run_local_microphone_relay_client.py "
    "--server.address 127.0.0.1 --server.port 8503"
)
_CLIENT_PROGRESS_LABELS: Final = {
    RelayClientProgressStage.CONFIGURATION_VALIDATED: "Yapılandırma doğrulandı",
    RelayClientProgressStage.TCP_CONNECTING: "TCP bağlantısı kuruluyor",
    RelayClientProgressStage.TCP_CONNECTED: "TCP bağlantısı kuruldu",
    RelayClientProgressStage.START_SENT: "START gönderildi",
    RelayClientProgressStage.START_ACKNOWLEDGED: "START GPU tarafından onaylandı",
    RelayClientProgressStage.NATIVE_MICROPHONE_OPENING: "Yerel mikrofon açılıyor",
    RelayClientProgressStage.NATIVE_MICROPHONE_OPENED: "Yerel mikrofon açıldı",
    RelayClientProgressStage.FIRST_NATIVE_AUDIO_BLOCK_RECEIVED: (
        "İlk yerel mikrofon ses bloğu alındı"
    ),
    RelayClientProgressStage.FIRST_PCM_CHUNK_ENQUEUED: "İlk PCM parçası kuyruğa alındı",
    RelayClientProgressStage.FIRST_AUDIO_CHUNK_SENT: "İlk ses parçası gönderildi",
    RelayClientProgressStage.FIRST_AUDIO_CHUNK_ACKNOWLEDGED: (
        "İlk ses parçası GPU tarafından onaylandı"
    ),
    RelayClientProgressStage.STREAMING: "Ses aktarımı etkin",
    RelayClientProgressStage.PAUSED: "Ses aktarımı duraklatıldı",
    RelayClientProgressStage.RESUMED: "Ses aktarımı sürdürüldü",
    RelayClientProgressStage.ENDED: "Relay oturumu tamamlandı",
    RelayClientProgressStage.FAILED: "Relay oturumu başarısız",
}
_CLIENT_WAITING_STAGES: Final = {
    RelayClientProgressStage.TCP_CONNECTING,
    RelayClientProgressStage.NATIVE_MICROPHONE_OPENING,
    RelayClientProgressStage.STREAMING,
}


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
        self._progress = RelayProgressHistory()
        self._resume_callback: Callable[[], bool] | None = None

    @property
    def progress_stages(self) -> tuple[RelayProgressStage, ...]:
        return self._progress.stages

    def record_progress(self, stage: RelayClientProgressStage) -> None:
        self._progress.record(stage)

    def set_resume_callback(self, callback: Callable[[], bool]) -> None:
        self._resume_callback = callback

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
            self.record_progress(RelayClientProgressStage.TCP_CONNECTING)
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
            self.record_progress(RelayClientProgressStage.TCP_CONNECTED)
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
                sent_callback=lambda: self.record_progress(
                    RelayClientProgressStage.START_SENT
                ),
            )
            self.record_progress(RelayClientProgressStage.START_ACKNOWLEDGED)
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
                    self.record_progress(RelayClientProgressStage.ENDED)
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
            sent_callback=(
                lambda: (
                    self.record_progress(
                        RelayClientProgressStage.FIRST_AUDIO_CHUNK_SENT
                    )
                    if outbound.message_type is RelayMessageType.AUDIO
                    else None
                )
            ),
        )
        resume_callback: Callable[[], bool] | None = None
        with self._lock:
            if outbound.message_type is RelayMessageType.AUDIO:
                self._sent_chunks += 1
                self._acknowledged_chunks += 1
                self.record_progress(
                    RelayClientProgressStage.FIRST_AUDIO_CHUNK_ACKNOWLEDGED
                )
                self.record_progress(RelayClientProgressStage.STREAMING)
            elif outbound.message_type is RelayMessageType.PAUSE:
                self._status = RelayClientStatus.PAUSED
                self.record_progress(RelayClientProgressStage.PAUSED)
            elif outbound.message_type is RelayMessageType.RESUME:
                self._generation = outbound.generation
                self._status = RelayClientStatus.STREAMING
                self.record_progress(RelayClientProgressStage.RESUMED)
                resume_callback = self._resume_callback
        if resume_callback is not None and not resume_callback():
            raise _RelayClientError(RelayReason.MICROPHONE_OPEN_FAILED)

    @staticmethod
    def _exchange(
        connection: socket.socket,
        message_type: RelayMessageType,
        metadata: dict[str, object],
        payload: bytes = b"",
        sent_callback: Callable[[], None] | None = None,
    ) -> None:
        raw_sequence = metadata["sequence_number"]
        if type(raw_sequence) is not int:
            raise _RelayClientError(RelayReason.INVALID_METADATA)
        sequence = raw_sequence
        connection.sendall(encode_relay_record(message_type, metadata, payload))
        if sent_callback is not None:
            sent_callback()
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
            self.record_progress(RelayClientProgressStage.FAILED)
            self._failure_reason = reason
            self._closed = True
            connection = self._socket if close_connection else None
            if close_connection:
                self._socket = None
        if connection is not None:
            connection.close()


class NativeInputStream(Protocol):
    def start(self) -> object: ...
    def stop(self) -> object: ...
    def close(self) -> object: ...


NativeStreamFactory = Callable[
    [Callable[[object, int, object, object], None]], NativeInputStream
]


def _sounddevice_stream_factory(
    callback: Callable[[object, int, object, object], None],
) -> NativeInputStream:
    import sounddevice  # type: ignore[import-untyped]

    devices = sounddevice.query_devices(kind="input")
    if not devices or int(devices.get("max_input_channels", 0)) < 1:
        raise LookupError
    return cast(
        NativeInputStream,
        sounddevice.RawInputStream(
            samplerate=16_000,
            channels=1,
            dtype="int16",
            callback=callback,
        ),
    )


class NativeRelayCapture:
    """Bounded Windows native PCM capture feeding the existing relay sender."""

    def __init__(
        self,
        sender: BoundedRelaySender,
        *,
        stream_factory: NativeStreamFactory = _sounddevice_stream_factory,
        queue_depth: int = LOCAL_NATIVE_CAPTURE_QUEUE_DEPTH,
    ) -> None:
        self._sender = sender
        self._stream_factory = stream_factory
        self._queue: Queue[bytes] = Queue(maxsize=queue_depth)
        self._stop = Event()
        self._overflow = Event()
        self._admitting = Event()
        self._drained = Event()
        self._drained.set()
        self._worker: Thread | None = None
        self._stream: NativeInputStream | None = None
        self._buffer = bytearray()
        self._lock = Lock()
        self._opened = Event()

    @property
    def worker_active(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def started(self) -> bool:
        return self._worker is not None

    @property
    def opened(self) -> bool:
        return self._opened.is_set()

    @property
    def device_label(self) -> str | None:
        return "Windows varsayılan giriş aygıtı" if self.opened else None

    def start(self) -> bool:
        if self._sender.diagnostics.status is not RelayClientStatus.STREAMING:
            return False
        with self._lock:
            if self._worker is not None:
                return False
            self._sender.record_progress(
                RelayClientProgressStage.NATIVE_MICROPHONE_OPENING
            )
            self._worker = Thread(
                target=self._run, name="native-relay-capture", daemon=True
            )
            self._worker.start()
            return True

    def audio_callback(
        self,
        input_data: object,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info, status
        if not self._admitting.is_set():
            return
        pcm = bytes(cast(Any, input_data))
        if not pcm or len(pcm) > LOCAL_MIC_CHUNK_BYTES or len(pcm) % 2:
            self._overflow.set()
            return
        try:
            self._drained.clear()
            self._queue.put_nowait(pcm)
        except Full:
            self._overflow.set()
            if self._queue.empty():
                self._drained.set()

    def pause(self) -> bool:
        self._admitting.clear()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                self._sender.fail(RelayReason.MICROPHONE_OPEN_FAILED)
                return False
        if not self._drained.wait(timeout=RELAY_IO_TIMEOUT_SECONDS):
            self._sender.fail(RelayReason.BUFFER_LIMIT)
            return False
        return self.flush()

    def resume(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        try:
            stream.start()
        except Exception:
            self._sender.fail(RelayReason.MICROPHONE_OPEN_FAILED)
            return False
        self._admitting.set()
        self._opened.set()
        return True

    def flush(self) -> bool:
        with self._lock:
            if not self._buffer:
                return True
            payload = bytes(self._buffer)
            self._buffer.clear()
        return self._enqueue(payload)

    def close(self) -> None:
        was_admitting = self._admitting.is_set()
        self._admitting.clear()
        self._stop.set()
        with self._lock:
            stream, self._stream = self._stream, None
            self._opened.clear()
        if stream is not None:
            if was_admitting:
                try:
                    stream.stop()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception:
                pass
        worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join(timeout=RELAY_IO_TIMEOUT_SECONDS)

    def _run(self) -> None:
        try:
            stream = self._stream_factory(self.audio_callback)
            with self._lock:
                if self._stop.is_set():
                    stream.close()
                    return
                self._stream = stream
            stream.start()
        except LookupError:
            self._sender.fail(RelayReason.MICROPHONE_UNAVAILABLE)
            return
        except Exception:
            self._sender.fail(RelayReason.MICROPHONE_OPEN_FAILED)
            return
        with self._lock:
            if self._stop.is_set():
                if self._stream is stream:
                    self._stream = None
                    stream.close()
                return
            self._opened.set()
            self._admitting.set()
        self._sender.record_progress(RelayClientProgressStage.NATIVE_MICROPHONE_OPENED)
        while not self._stop.is_set():
            if self._overflow.is_set():
                self._sender.fail(RelayReason.BUFFER_LIMIT)
                self._stop.set()
                break
            try:
                pcm = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._sender.record_progress(
                    RelayClientProgressStage.FIRST_NATIVE_AUDIO_BLOCK_RECEIVED
                )
                self._append(pcm)
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._drained.set()

    def _append(self, pcm: bytes) -> None:
        offset = 0
        while offset < len(pcm):
            with self._lock:
                take = min(LOCAL_MIC_CHUNK_BYTES - len(self._buffer), len(pcm) - offset)
                self._buffer.extend(pcm[offset : offset + take])
                offset += take
                if len(self._buffer) < LOCAL_MIC_CHUNK_BYTES:
                    continue
                payload = bytes(self._buffer)
                self._buffer.clear()
            if not self._enqueue(payload):
                self._stop.set()
                return

    def _enqueue(self, payload: bytes) -> bool:
        accepted = self._sender.enqueue_audio(
            payload, captured_at_utc=datetime.now(UTC)
        )
        if accepted:
            self._sender.record_progress(
                RelayClientProgressStage.FIRST_PCM_CHUNK_ENQUEUED
            )
        return accepted


@dataclass(slots=True)
class RelayClientSession(RelayClientSessionHandle):
    sender: BoundedRelaySender
    capture: NativeRelayCapture

    def pause(self) -> bool:
        return self.capture.pause() and self.sender.pause()

    def resume(self) -> bool:
        return self.sender.resume()

    def end(self) -> bool:
        if not self.capture.pause() or not self.sender.end():
            return False
        self.capture.close()
        return True

    def close(self) -> None:
        self.capture.close()
        self.sender.close()


def create_relay_client_session(
    config: RelayClientConfig,
    *,
    stream_factory: NativeStreamFactory = _sounddevice_stream_factory,
) -> RelayClientSession:
    sender = BoundedRelaySender(config)
    capture = NativeRelayCapture(sender, stream_factory=stream_factory)
    sender.set_resume_callback(capture.resume)
    sender.record_progress(RelayClientProgressStage.CONFIGURATION_VALIDATED)
    return RelayClientSession(sender=sender, capture=capture)


def retained_relay_client_session(value: object) -> RelayClientSession | None:
    if not isinstance(value, RelayClientSessionHandle):
        return None
    return cast(RelayClientSession, value)


def reset_terminal_relay_client_session(
    session_state: MutableMapping[str, object],
) -> bool:
    session = retained_relay_client_session(session_state.get("relay_client_session"))
    if session is None or session.sender.diagnostics.status not in {
        RelayClientStatus.FAILED,
        RelayClientStatus.ENDED,
    }:
        return False
    session.close()
    session_state.pop("relay_client_session", None)
    return True


def _render_relay_progress(st: object, sender: BoundedRelaySender) -> None:
    stages = sender.progress_stages
    for index, stage in enumerate(stages):
        assert isinstance(stage, RelayClientProgressStage)
        current = index == len(stages) - 1
        prefix = "…" if current and stage in _CLIENT_WAITING_STAGES else "✓"
        getattr(st, "caption")(f"{prefix} {_CLIENT_PROGRESS_LABELS[stage]}")


def _relay_failure_message(reason: RelayReason) -> str:
    return f"Relay başarısız: {reason.value}"


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
    session = retained_relay_client_session(
        st.session_state.get("relay_client_session")
    )
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
            if not session.sender.start():
                raise RuntimeError(RelayReason.TERMINAL_STATE.value)
            st.rerun()
        except Exception:
            retained = retained_relay_client_session(
                st.session_state.pop("relay_client_session", None)
            )
            if retained is not None:
                retained.close()
            st.error("Relay oturumu güvenli biçimde başlatılamadı.")
    if session is None:
        st.info("Bağlantı bilgilerini girip Connect / Start seçin.")
        return
    diagnostics = session.sender.diagnostics
    st.metric("Durum", diagnostics.status.value)
    st.metric("Generation", diagnostics.generation)
    st.metric("Gönderilen parça", diagnostics.sent_chunk_count)
    st.metric("Onaylanan parça", diagnostics.acknowledged_chunk_count)
    st.metric("Kuyruk", diagnostics.queue_depth)
    if diagnostics.failure_reason is not None:
        st.error(_relay_failure_message(diagnostics.failure_reason))
    if diagnostics.status in {RelayClientStatus.FAILED, RelayClientStatus.ENDED}:
        if st.button("Reset / Reconnect"):
            reset_terminal_relay_client_session(
                cast(MutableMapping[str, object], st.session_state)
            )
            st.rerun()
        return
    _render_relay_progress(st, session.sender)
    if session.capture.device_label is not None:
        st.caption(f"Giriş aygıtı: {session.capture.device_label}")
    if st.button(
        "Mikrofonu Başlat",
        disabled=(
            diagnostics.status is not RelayClientStatus.STREAMING
            or session.capture.started
        ),
    ):
        session.capture.start()
        st.rerun()
    pause, resume, end = st.columns(3)
    if pause.button(
        "Pause",
        disabled=(
            diagnostics.status is not RelayClientStatus.STREAMING
            or not session.capture.opened
        ),
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
