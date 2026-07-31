"""Ephemeral, localhost-only microphone capture for synthetic development tests."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from ipaddress import ip_address
from math import isfinite
from threading import Condition, Event, RLock
from typing import Never, Protocol, runtime_checkable
from uuid import uuid4

import av

from app.audio_ingress.boundary import (
    IngressAcceptanceStatus,
    LiveAudioIngressBoundary,
)
from app.audio_ingress.contracts import (
    LiveAudioEndReason,
    LiveAudioEventType,
    LiveAudioIngressEvent,
)
from app.events.models import AudioChunkEvent


LOCAL_MIC_PROVIDER_NAME = "LOCAL_MIC_TEST"
LOCAL_MIC_GATE_ENVIRONMENT_KEY = "CALLMETRIC_DASHBOARD_LOCAL_MIC_TEST"
LOCAL_MIC_SAMPLE_RATE_HZ = 16_000
LOCAL_MIC_CHANNEL_COUNT = 1
LOCAL_MIC_CODEC_NAME = "pcm_s16le"
LOCAL_MIC_CHUNK_SECONDS = 2.0
LOCAL_MIC_CHUNK_BYTES = 64_000
LOCAL_MIC_MAX_QUEUE_DEPTH = 8


class LocalMicrophoneStatus(str, Enum):
    PERMISSION_PENDING = "PERMISSION_PENDING"
    READY = "READY"
    STREAMING = "STREAMING"
    STOP_REQUESTED = "STOP_REQUESTED"
    COMPLETED = "COMPLETED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    OVERLOADED = "OVERLOADED"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"
    REPLACED = "REPLACED"


class LocalMicrophoneASRReadiness(str, Enum):
    PREPARING_MODEL = "PREPARING_MODEL"
    WARMING_UP = "WARMING_UP"
    READY_TO_CAPTURE = "READY_TO_CAPTURE"
    STREAMING = "STREAMING"
    FAILED = "FAILED"


class LocalMicrophoneTerminalReason(str, Enum):
    COMPLETED = "COMPLETED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    OVERLOADED = "OVERLOADED"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"
    RESET = "RESET"
    REPLACED = "REPLACED"
    RESOURCE_CLOSED = "RESOURCE_CLOSED"


@dataclass(frozen=True, slots=True)
class LocalMicrophoneASRTimings:
    engine_construction_seconds: float | None = None
    model_loading_seconds: float | None = None
    warmup_seconds: float | None = None
    first_audio_preparation_seconds: float | None = None
    first_inference_seconds: float | None = None
    latest_audio_preparation_seconds: float | None = None
    latest_inference_seconds: float | None = None
    inference_count: int = 0


@dataclass(frozen=True, slots=True)
class LocalMicrophoneDiagnostics:
    status: LocalMicrophoneStatus
    asr_readiness: LocalMicrophoneASRReadiness
    asr_timings: LocalMicrophoneASRTimings
    received_chunk_count: int
    processed_audio_seconds: float
    queue_depth: int
    estimated_latency_seconds: float | None
    end_emitted: bool


@dataclass(slots=True)
class _CapabilityState:
    active: bool = True
    lock: RLock = field(default_factory=RLock)


@dataclass(frozen=True, slots=True)
class LocalMicTestCapability:
    """Non-serializable exact-scope authorization for one local microphone run."""

    tenant_id: str
    call_id: str
    _resource: object = field(repr=False, compare=False)
    _nonce: object = field(repr=False, compare=False)
    _state: _CapabilityState = field(repr=False, compare=False)

    def __reduce__(self) -> Never:
        raise TypeError("local microphone capabilities cannot be serialized")

    @property
    def active(self) -> bool:
        with self._state.lock:
            return self._state.active

    def authorizes(
        self,
        *,
        tenant_id: str,
        call_id: str,
        resource: object,
    ) -> bool:
        with self._state.lock:
            return (
                self._state.active
                and tenant_id == self.tenant_id
                and call_id == self.call_id
                and resource is self._resource
            )

    def revoke(self) -> None:
        with self._state.lock:
            self._state.active = False


@runtime_checkable
class LocalMicrophoneSessionProtocol(Protocol):
    @property
    def capability(self) -> LocalMicTestCapability: ...

    @property
    def diagnostics(self) -> LocalMicrophoneDiagnostics: ...

    @property
    def component_key(self) -> str: ...

    def close(
        self,
        reason: LocalMicrophoneTerminalReason = (
            LocalMicrophoneTerminalReason.RESOURCE_CLOSED
        ),
    ) -> None: ...


def local_microphone_test_enabled(
    *,
    server_address: str | None,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Require both an explicit opt-in and an exact loopback bind address."""
    source = os.environ if environment is None else environment
    if source.get(LOCAL_MIC_GATE_ENVIRONMENT_KEY) != "1":
        return False
    if server_address is None:
        return False
    cleaned = server_address.strip().lower()
    if cleaned == "localhost":
        return True
    try:
        return ip_address(cleaned).is_loopback
    except ValueError:
        return False


def create_local_mic_test_capability(
    *,
    tenant_id: str,
    call_id: str,
    resource: object,
    server_address: str | None,
    environment: Mapping[str, str] | None = None,
) -> LocalMicTestCapability:
    if not local_microphone_test_enabled(
        server_address=server_address,
        environment=environment,
    ):
        raise PermissionError("local_microphone_test_disabled")
    cleaned_tenant = tenant_id.strip()
    cleaned_call = call_id.strip()
    if not cleaned_tenant or not cleaned_call:
        raise ValueError("invalid_local_microphone_scope")
    return LocalMicTestCapability(
        tenant_id=cleaned_tenant,
        call_id=cleaned_call,
        _resource=resource,
        _nonce=(uuid4(), object()),
        _state=_CapabilityState(),
    )


@dataclass(frozen=True, slots=True)
class _NormalizedAudio:
    pcm_s16le: bytes = field(repr=False)
    media_time_seconds: float | None


class _AudioNormalizerProtocol(Protocol):
    def normalize(self, frame: av.AudioFrame) -> tuple[_NormalizedAudio, ...]: ...

    def flush(self) -> tuple[_NormalizedAudio, ...]: ...


class PyAVLocalMicrophoneNormalizer:
    """Convert decoded WebRTC frames to bounded mono 16 kHz signed PCM16."""

    def __init__(self) -> None:
        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=LOCAL_MIC_SAMPLE_RATE_HZ,
        )

    def normalize(self, frame: av.AudioFrame) -> tuple[_NormalizedAudio, ...]:
        if not isinstance(frame, av.AudioFrame):
            raise TypeError("invalid_audio_frame")
        media_time = _frame_media_time(frame)
        return tuple(
            _NormalizedAudio(
                pcm_s16le=_validated_pcm_bytes(item),
                media_time_seconds=media_time,
            )
            for item in self._resampler.resample(frame)
        )

    def flush(self) -> tuple[_NormalizedAudio, ...]:
        return tuple(
            _NormalizedAudio(
                pcm_s16le=_validated_pcm_bytes(item),
                media_time_seconds=None,
            )
            for item in self._resampler.resample(None)
        )


class LocalMicrophoneIngressSession:
    """Thread-safe bridge from WebRTC frames to the existing ingress boundary."""

    def __init__(
        self,
        *,
        capability: LocalMicTestCapability,
        resource: object,
        provider_stream_id: str | None = None,
        normalizer: _AudioNormalizerProtocol | None = None,
        max_queue_depth: int = LOCAL_MIC_MAX_QUEUE_DEPTH,
    ) -> None:
        if not capability.authorizes(
            tenant_id=capability.tenant_id,
            call_id=capability.call_id,
            resource=resource,
        ):
            raise PermissionError("invalid_local_microphone_capability")
        self._capability = capability
        self._resource = resource
        self._provider_stream_id = provider_stream_id or uuid4().hex
        component_digest = sha256(self._provider_stream_id.encode("utf-8")).hexdigest()
        self._component_key = f"local-mic-{component_digest[:24]}"
        self._normalizer = normalizer or PyAVLocalMicrophoneNormalizer()
        self._boundary = LiveAudioIngressBoundary(
            tenant_id=capability.tenant_id,
            provider_name=LOCAL_MIC_PROVIDER_NAME,
            max_queue_depth=max_queue_depth,
        )
        self._condition = Condition(RLock())
        self._buffer = bytearray()
        self._buffer_capture_utc: datetime | None = None
        self._first_media_time: float | None = None
        self._capture_origin_utc: datetime | None = None
        self._sequence = 0
        self._status = LocalMicrophoneStatus.PERMISSION_PENDING
        self._asr_readiness = LocalMicrophoneASRReadiness.PREPARING_MODEL
        self._asr_timings = LocalMicrophoneASRTimings()
        self._received_chunk_count = 0
        self._processed_audio_seconds = 0.0
        self._estimated_latency_seconds: float | None = None
        self._end_reason: LiveAudioEndReason | None = None
        self._terminal_status: LocalMicrophoneStatus | None = None
        self._end_emitted = False
        self._closed = False

    @property
    def capability(self) -> LocalMicTestCapability:
        return self._capability

    @property
    def component_key(self) -> str:
        return self._component_key

    @property
    def diagnostics(self) -> LocalMicrophoneDiagnostics:
        with self._condition:
            return LocalMicrophoneDiagnostics(
                status=self._status,
                asr_readiness=self._asr_readiness,
                asr_timings=self._asr_timings,
                received_chunk_count=self._received_chunk_count,
                processed_audio_seconds=self._processed_audio_seconds,
                queue_depth=self._boundary.retained_audio_chunk_count,
                estimated_latency_seconds=self._estimated_latency_seconds,
                end_emitted=self._end_emitted,
            )

    def start(self, *, arrived_at_utc: datetime | None = None) -> bool:
        with self._condition:
            self._require_authorized()
            if self._status in {
                LocalMicrophoneStatus.READY,
                LocalMicrophoneStatus.STREAMING,
            }:
                return False
            if self._asr_readiness is not LocalMicrophoneASRReadiness.READY_TO_CAPTURE:
                raise RuntimeError("local_microphone_asr_not_ready")
            if self._status is not LocalMicrophoneStatus.PERMISSION_PENDING:
                raise RuntimeError("local_microphone_session_not_startable")
            now = _aware_utc(arrived_at_utc)
            result = self._boundary.accept(
                self._event(
                    LiveAudioEventType.START,
                    captured_at_utc=now,
                    arrived_at_utc=now,
                )
            )
            if result.status is not IngressAcceptanceStatus.ACCEPTED:
                raise RuntimeError("local_microphone_start_rejected")
            self._sequence += 1
            self._status = LocalMicrophoneStatus.READY
            self._asr_readiness = LocalMicrophoneASRReadiness.STREAMING
            return True

    def set_asr_readiness(
        self,
        readiness: LocalMicrophoneASRReadiness,
        *,
        resource: object,
    ) -> bool:
        if not isinstance(readiness, LocalMicrophoneASRReadiness):
            raise ValueError("invalid_local_microphone_asr_readiness")
        with self._condition:
            self._require_authorized()
            if resource is not self._resource:
                raise PermissionError("invalid_local_microphone_capability")
            if readiness is self._asr_readiness:
                return False
            allowed = {
                LocalMicrophoneASRReadiness.PREPARING_MODEL: {
                    LocalMicrophoneASRReadiness.WARMING_UP,
                    LocalMicrophoneASRReadiness.FAILED,
                },
                LocalMicrophoneASRReadiness.WARMING_UP: {
                    LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
                    LocalMicrophoneASRReadiness.FAILED,
                },
                LocalMicrophoneASRReadiness.READY_TO_CAPTURE: {
                    LocalMicrophoneASRReadiness.FAILED,
                },
                LocalMicrophoneASRReadiness.STREAMING: {
                    LocalMicrophoneASRReadiness.FAILED,
                },
            }.get(self._asr_readiness, set())
            if readiness not in allowed:
                raise RuntimeError("invalid_local_microphone_asr_transition")
            self._asr_readiness = readiness
            self._condition.notify_all()
            return True

    def record_asr_preparation(
        self,
        *,
        resource: object,
        engine_construction_seconds: float | None = None,
        model_loading_seconds: float | None = None,
        warmup_seconds: float | None = None,
    ) -> None:
        values = (
            engine_construction_seconds,
            model_loading_seconds,
            warmup_seconds,
        )
        if any(
            value is not None and (not isfinite(value) or value < 0) for value in values
        ):
            raise ValueError("invalid_local_microphone_asr_timing")
        with self._condition:
            self._require_authorized()
            if resource is not self._resource:
                raise PermissionError("invalid_local_microphone_capability")
            current = self._asr_timings
            self._asr_timings = LocalMicrophoneASRTimings(
                engine_construction_seconds=(
                    current.engine_construction_seconds
                    if engine_construction_seconds is None
                    else engine_construction_seconds
                ),
                model_loading_seconds=(
                    current.model_loading_seconds
                    if model_loading_seconds is None
                    else model_loading_seconds
                ),
                warmup_seconds=(
                    current.warmup_seconds if warmup_seconds is None else warmup_seconds
                ),
                first_audio_preparation_seconds=(
                    current.first_audio_preparation_seconds
                ),
                first_inference_seconds=current.first_inference_seconds,
                latest_audio_preparation_seconds=(
                    current.latest_audio_preparation_seconds
                ),
                latest_inference_seconds=current.latest_inference_seconds,
                inference_count=current.inference_count,
            )

    def record_asr_inference(
        self,
        *,
        resource: object,
        audio_preparation_seconds: float,
        inference_seconds: float,
    ) -> None:
        if (
            not isfinite(audio_preparation_seconds)
            or audio_preparation_seconds < 0
            or not isfinite(inference_seconds)
            or inference_seconds < 0
        ):
            raise ValueError("invalid_local_microphone_asr_timing")
        with self._condition:
            self._require_authorized()
            if resource is not self._resource:
                raise PermissionError("invalid_local_microphone_capability")
            current = self._asr_timings
            self._asr_timings = LocalMicrophoneASRTimings(
                engine_construction_seconds=current.engine_construction_seconds,
                model_loading_seconds=current.model_loading_seconds,
                warmup_seconds=current.warmup_seconds,
                first_audio_preparation_seconds=(
                    audio_preparation_seconds
                    if current.first_audio_preparation_seconds is None
                    else current.first_audio_preparation_seconds
                ),
                first_inference_seconds=(
                    inference_seconds
                    if current.first_inference_seconds is None
                    else current.first_inference_seconds
                ),
                latest_audio_preparation_seconds=audio_preparation_seconds,
                latest_inference_seconds=inference_seconds,
                inference_count=current.inference_count + 1,
            )

    def accept_frame(
        self,
        frame: av.AudioFrame,
        *,
        arrived_at_utc: datetime | None = None,
    ) -> av.AudioFrame:
        now = _aware_utc(arrived_at_utc)
        if self.diagnostics.asr_readiness not in {
            LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
            LocalMicrophoneASRReadiness.STREAMING,
        }:
            raise RuntimeError("local_microphone_asr_not_ready")
        if self.diagnostics.status is LocalMicrophoneStatus.PERMISSION_PENDING:
            self.start(arrived_at_utc=now)
        normalized = self._normalizer.normalize(frame)
        with self._condition:
            self._require_authorized()
            if self._status not in {
                LocalMicrophoneStatus.READY,
                LocalMicrophoneStatus.STREAMING,
            }:
                raise RuntimeError("local_microphone_session_not_streaming")
            for item in normalized:
                self._append_normalized(item, arrived_at_utc=now)
            if self._capability.active:
                self._status = LocalMicrophoneStatus.STREAMING
            self._condition.notify_all()
        return frame

    def request_stop(self, *, arrived_at_utc: datetime | None = None) -> bool:
        return self._request_draining_end(
            terminal_status=LocalMicrophoneStatus.COMPLETED,
            end_reason=LiveAudioEndReason.COMPLETED,
            arrived_at_utc=arrived_at_utc,
        )

    def _request_draining_end(
        self,
        *,
        terminal_status: LocalMicrophoneStatus,
        end_reason: LiveAudioEndReason,
        arrived_at_utc: datetime | None,
    ) -> bool:
        now = _aware_utc(arrived_at_utc)
        try:
            flushed = self._normalizer.flush()
        except Exception:
            self._cancel(LocalMicrophoneTerminalReason.FAILED)
            return False
        with self._condition:
            if self._end_reason is not None or self._closed:
                return False
            self._require_authorized()
            if self._boundary.active_stream_count == 0:
                self._cancel(
                    (
                        LocalMicrophoneTerminalReason.DISCONNECTED
                        if terminal_status is LocalMicrophoneStatus.DISCONNECTED
                        else LocalMicrophoneTerminalReason.COMPLETED
                    )
                )
                return True
            for item in flushed:
                self._append_normalized(item, arrived_at_utc=now)
            self._emit_buffer(arrived_at_utc=now, allow_short=True)
            if self._closed:
                return False
            self._status = LocalMicrophoneStatus.STOP_REQUESTED
            self._terminal_status = terminal_status
            self._end_reason = end_reason
            if self._boundary.retained_audio_chunk_count == 0:
                self._emit_end()
            self._condition.notify_all()
            return True

    def iter_audio_chunks(
        self,
        *,
        cancellation: Event,
    ) -> Iterable[AudioChunkEvent]:
        while True:
            with self._condition:
                while (
                    not cancellation.is_set()
                    and self._boundary.retained_audio_chunk_count == 0
                    and not self._end_emitted
                    and self._end_reason is None
                ):
                    self._condition.wait(timeout=0.25)
                if cancellation.is_set():
                    self._cancel(LocalMicrophoneTerminalReason.RESOURCE_CLOSED)
                    return
                drained = self._boundary.drain(
                    tenant_id=self._capability.tenant_id,
                    call_id=self._capability.call_id,
                    provider_name=LOCAL_MIC_PROVIDER_NAME,
                    provider_stream_id=self._provider_stream_id,
                    limit=1,
                )
                if drained:
                    chunk = drained[0].audio_chunk
                    self._condition.notify_all()
                elif self._end_reason is not None:
                    self._emit_end()
                    return
                elif self._end_emitted:
                    return
                else:
                    continue
            yield chunk

    def disconnect(self, *, arrived_at_utc: datetime | None = None) -> bool:
        return self._request_draining_end(
            terminal_status=LocalMicrophoneStatus.DISCONNECTED,
            end_reason=LiveAudioEndReason.CANCELLED,
            arrived_at_utc=arrived_at_utc,
        )

    def fail(self) -> None:
        with self._condition:
            self._asr_readiness = LocalMicrophoneASRReadiness.FAILED
        self._cancel(LocalMicrophoneTerminalReason.FAILED)

    def deny_permission(self) -> None:
        self._cancel(LocalMicrophoneTerminalReason.PERMISSION_DENIED)

    def close(
        self,
        reason: LocalMicrophoneTerminalReason = (
            LocalMicrophoneTerminalReason.RESOURCE_CLOSED
        ),
    ) -> None:
        self._cancel(reason)

    def _append_normalized(
        self,
        item: _NormalizedAudio,
        *,
        arrived_at_utc: datetime,
    ) -> None:
        if not item.pcm_s16le:
            return
        if len(item.pcm_s16le) % 2:
            raise ValueError("incomplete_pcm16_frame")
        capture_utc = self._estimated_capture_time(
            item.media_time_seconds,
            arrived_at_utc=arrived_at_utc,
        )
        offset = 0
        while offset < len(item.pcm_s16le):
            if not self._buffer:
                self._buffer_capture_utc = capture_utc
            remaining = LOCAL_MIC_CHUNK_BYTES - len(self._buffer)
            take = min(remaining, len(item.pcm_s16le) - offset)
            self._buffer.extend(item.pcm_s16le[offset : offset + take])
            offset += take
            if len(self._buffer) == LOCAL_MIC_CHUNK_BYTES:
                self._emit_buffer(arrived_at_utc=arrived_at_utc, allow_short=False)

    def _emit_buffer(
        self,
        *,
        arrived_at_utc: datetime,
        allow_short: bool,
    ) -> None:
        if not self._buffer:
            return
        if not allow_short and len(self._buffer) != LOCAL_MIC_CHUNK_BYTES:
            return
        duration = len(self._buffer) / (LOCAL_MIC_SAMPLE_RATE_HZ * 2)
        captured = self._buffer_capture_utc or arrived_at_utc
        if captured > arrived_at_utc:
            captured = arrived_at_utc
        result = self._boundary.accept(
            self._event(
                LiveAudioEventType.AUDIO_CHUNK,
                captured_at_utc=captured,
                arrived_at_utc=arrived_at_utc,
                duration_seconds=duration,
                audio_payload=bytes(self._buffer),
            )
        )
        if result.status is IngressAcceptanceStatus.OVERLOADED:
            self._buffer.clear()
            self._buffer_capture_utc = None
            self._cancel(LocalMicrophoneTerminalReason.OVERLOADED)
            return
        if result.status is not IngressAcceptanceStatus.ACCEPTED:
            raise RuntimeError("local_microphone_chunk_rejected")
        self._sequence += 1
        self._received_chunk_count += 1
        self._processed_audio_seconds += duration
        latency = (arrived_at_utc - captured).total_seconds()
        self._estimated_latency_seconds = max(latency, 0.0)
        self._buffer.clear()
        self._buffer_capture_utc = None
        self._condition.notify_all()

    def _estimated_capture_time(
        self,
        media_time_seconds: float | None,
        *,
        arrived_at_utc: datetime,
    ) -> datetime:
        if media_time_seconds is None or not isfinite(media_time_seconds):
            return arrived_at_utc
        if self._first_media_time is None:
            self._first_media_time = media_time_seconds
            self._capture_origin_utc = arrived_at_utc
        assert self._capture_origin_utc is not None
        delta = max(media_time_seconds - self._first_media_time, 0.0)
        estimated = self._capture_origin_utc + timedelta(seconds=delta)
        return min(estimated, arrived_at_utc)

    def _emit_end(self) -> None:
        if self._end_emitted:
            return
        now = datetime.now(UTC)
        reason = self._end_reason or LiveAudioEndReason.CANCELLED
        result = self._boundary.accept(
            self._event(
                LiveAudioEventType.END,
                captured_at_utc=now,
                arrived_at_utc=now,
                end_reason=reason,
            )
        )
        if result.status is not IngressAcceptanceStatus.COMPLETED:
            raise RuntimeError("local_microphone_end_rejected")
        self._sequence += 1
        self._end_emitted = True
        self._status = self._terminal_status or (
            LocalMicrophoneStatus.COMPLETED
            if reason is LiveAudioEndReason.COMPLETED
            else self._status
        )
        self._capability.revoke()

    def _cancel(self, reason: LocalMicrophoneTerminalReason) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._buffer.clear()
            self._buffer_capture_utc = None
            self._end_reason = LiveAudioEndReason.CANCELLED
            self._terminal_status = None
            self._status = {
                LocalMicrophoneTerminalReason.PERMISSION_DENIED: (
                    LocalMicrophoneStatus.PERMISSION_DENIED
                ),
                LocalMicrophoneTerminalReason.OVERLOADED: (
                    LocalMicrophoneStatus.OVERLOADED
                ),
                LocalMicrophoneTerminalReason.DISCONNECTED: (
                    LocalMicrophoneStatus.DISCONNECTED
                ),
                LocalMicrophoneTerminalReason.FAILED: LocalMicrophoneStatus.FAILED,
                LocalMicrophoneTerminalReason.REPLACED: (
                    LocalMicrophoneStatus.REPLACED
                ),
            }.get(reason, LocalMicrophoneStatus.COMPLETED)
            if self._boundary.active_stream_count:
                self._emit_end()
            else:
                self._end_emitted = True
                self._capability.revoke()
            self._condition.notify_all()

    def _event(
        self,
        event_type: LiveAudioEventType,
        *,
        captured_at_utc: datetime,
        arrived_at_utc: datetime,
        duration_seconds: float = 0.0,
        audio_payload: bytes = b"",
        end_reason: LiveAudioEndReason | None = None,
    ) -> LiveAudioIngressEvent:
        return LiveAudioIngressEvent(
            tenant_id=self._capability.tenant_id,
            call_id=self._capability.call_id,
            provider_name=LOCAL_MIC_PROVIDER_NAME,
            provider_stream_id=self._provider_stream_id,
            event_type=event_type,
            sequence_number=self._sequence,
            codec_name=LOCAL_MIC_CODEC_NAME,
            sample_rate_hz=LOCAL_MIC_SAMPLE_RATE_HZ,
            channel_count=LOCAL_MIC_CHANNEL_COUNT,
            captured_at_utc=captured_at_utc,
            arrived_at_utc=arrived_at_utc,
            duration_seconds=duration_seconds,
            audio_payload=audio_payload,
            end_reason=end_reason,
        )

    def _require_authorized(self) -> None:
        if self._closed or not self._capability.authorizes(
            tenant_id=self._capability.tenant_id,
            call_id=self._capability.call_id,
            resource=self._resource,
        ):
            raise PermissionError("local_microphone_capability_revoked")


def _frame_media_time(frame: av.AudioFrame) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    time_base = frame.time_base
    if not isinstance(time_base, Fraction):
        time_base = Fraction(time_base)
    value = float(frame.pts * time_base)
    return value if isfinite(value) else None


def _validated_pcm_bytes(frame: av.AudioFrame) -> bytes:
    if (
        frame.sample_rate != LOCAL_MIC_SAMPLE_RATE_HZ
        or frame.format.name != "s16"
        or frame.layout.name != "mono"
    ):
        raise ValueError("invalid_normalized_audio_format")
    payload = frame.to_ndarray().tobytes()
    if not payload or len(payload) % 2:
        raise ValueError("invalid_normalized_audio_payload")
    return payload


def _aware_utc(value: datetime | None) -> datetime:
    resolved = datetime.now(UTC) if value is None else value
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError("timezone_required")
    return resolved.astimezone(UTC)
