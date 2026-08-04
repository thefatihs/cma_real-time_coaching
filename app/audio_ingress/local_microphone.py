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
from math import isfinite, sqrt
from threading import Condition, Event, RLock
from typing import Never, Protocol, runtime_checkable
from uuid import uuid4

import av
import numpy as np

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
_MAX_LOCAL_MIC_DIAGNOSTIC_COUNT = 1_000_000
_MAX_LOCAL_MIC_AUDIO_VALUE_COUNT = 2_000_000_000


class LocalMicrophoneStatus(str, Enum):
    PERMISSION_PENDING = "PERMISSION_PENDING"
    READY = "READY"
    STREAMING = "STREAMING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    RECONNECTING = "RECONNECTING"
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


class LocalMicrophoneTranscriptRejectionReason(str, Enum):
    NONE = "NONE"
    STEP_SCOPE_MISMATCH = "STEP_SCOPE_MISMATCH"
    RESULT_SCOPE_MISMATCH = "RESULT_SCOPE_MISMATCH"
    EVENT_SCOPE_MISMATCH = "EVENT_SCOPE_MISMATCH"
    REVISION_REGRESSION = "REVISION_REGRESSION"


class LocalMicrophoneAudioRejectionReason(str, Enum):
    NONE = "NONE"
    STALE_CAPTURE_GENERATION = "STALE_CAPTURE_GENERATION"
    ASR_NOT_READY = "ASR_NOT_READY"
    CAPABILITY_REVOKED = "CAPABILITY_REVOKED"
    CAPTURE_NOT_STREAMING = "CAPTURE_NOT_STREAMING"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"
    INGRESS_REJECTED = "INGRESS_REJECTED"


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
    capture_generation: int
    pause_count: int
    reconnect_count: int
    in_flight_chunk_count: int
    asr_non_empty_result_count: int
    asr_empty_result_count: int
    asr_segment_count: int
    latest_window_duration_seconds: float
    partial_event_count: int
    stable_commit_count: int
    rejected_transcript_event_count: int
    latest_transcript_rejection_reason: LocalMicrophoneTranscriptRejectionReason
    callback_frame_count: int
    rejected_capture_frame_count: int
    latest_audio_rejection_reason: LocalMicrophoneAudioRejectionReason
    input_sample_rate_hz: int | None
    input_channel_count: int | None
    input_sample_count: int
    pre_resample_rms: float
    pre_resample_peak: float
    post_resample_rms: float
    post_resample_peak: float
    post_resample_nonzero_ratio: float
    post_resample_clipping_ratio: float
    produced_pcm_byte_count: int


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


@dataclass(frozen=True, slots=True)
class _AudioEnergy:
    sample_count: int
    square_sum: float
    peak: float
    nonzero_count: int
    clipping_count: int


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
        self._call_started = False
        self._capture_generation = 1
        self._pause_count = 0
        self._reconnect_count = 0
        self._pause_pending = False
        self._capture_stop_status: LocalMicrophoneStatus | None = None
        self._in_flight_chunk_count = 0
        self._asr_non_empty_result_count = 0
        self._asr_empty_result_count = 0
        self._asr_segment_count = 0
        self._latest_window_duration_seconds = 0.0
        self._partial_event_count = 0
        self._stable_commit_count = 0
        self._rejected_transcript_event_count = 0
        self._latest_transcript_rejection_reason = (
            LocalMicrophoneTranscriptRejectionReason.NONE
        )
        self._callback_frame_count = 0
        self._rejected_capture_frame_count = 0
        self._latest_audio_rejection_reason = LocalMicrophoneAudioRejectionReason.NONE
        self._input_sample_rate_hz: int | None = None
        self._input_channel_count: int | None = None
        self._input_sample_count = 0
        self._pre_resample_square_sum = 0.0
        self._pre_resample_peak = 0.0
        self._post_resample_sample_count = 0
        self._post_resample_square_sum = 0.0
        self._post_resample_peak = 0.0
        self._post_resample_nonzero_count = 0
        self._post_resample_clipping_count = 0
        self._produced_pcm_byte_count = 0
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
                capture_generation=self._capture_generation,
                pause_count=self._pause_count,
                reconnect_count=self._reconnect_count,
                in_flight_chunk_count=self._in_flight_chunk_count,
                asr_non_empty_result_count=self._asr_non_empty_result_count,
                asr_empty_result_count=self._asr_empty_result_count,
                asr_segment_count=self._asr_segment_count,
                latest_window_duration_seconds=(self._latest_window_duration_seconds),
                partial_event_count=self._partial_event_count,
                stable_commit_count=self._stable_commit_count,
                rejected_transcript_event_count=(self._rejected_transcript_event_count),
                latest_transcript_rejection_reason=(
                    self._latest_transcript_rejection_reason
                ),
                callback_frame_count=self._callback_frame_count,
                rejected_capture_frame_count=self._rejected_capture_frame_count,
                latest_audio_rejection_reason=self._latest_audio_rejection_reason,
                input_sample_rate_hz=self._input_sample_rate_hz,
                input_channel_count=self._input_channel_count,
                input_sample_count=self._input_sample_count,
                pre_resample_rms=_rms(
                    self._pre_resample_square_sum,
                    self._input_sample_count,
                ),
                pre_resample_peak=self._pre_resample_peak,
                post_resample_rms=_rms(
                    self._post_resample_square_sum,
                    self._post_resample_sample_count,
                ),
                post_resample_peak=self._post_resample_peak,
                post_resample_nonzero_ratio=_ratio(
                    self._post_resample_nonzero_count,
                    self._post_resample_sample_count,
                ),
                post_resample_clipping_ratio=_ratio(
                    self._post_resample_clipping_count,
                    self._post_resample_sample_count,
                ),
                produced_pcm_byte_count=self._produced_pcm_byte_count,
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
            if self._status not in {
                LocalMicrophoneStatus.PERMISSION_PENDING,
                LocalMicrophoneStatus.RECONNECTING,
            }:
                raise RuntimeError("local_microphone_session_not_startable")
            now = _aware_utc(arrived_at_utc)
            if not self._call_started:
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
                self._call_started = True
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
            self._require_resource(resource)
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
                    LocalMicrophoneASRReadiness.STREAMING,
                    LocalMicrophoneASRReadiness.FAILED,
                },
                LocalMicrophoneASRReadiness.STREAMING: {
                    LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
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
            self._require_resource(resource)
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
            self._require_resource(resource)
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

    def record_transcript_step(
        self,
        *,
        resource: object,
        asr_result_non_empty: bool,
        asr_segment_count: int = 0,
        window_duration_seconds: float = 0.0,
        partial_event_count: int,
        stable_commit_count: int,
        rejected_event_count: int = 0,
        rejection_reason: LocalMicrophoneTranscriptRejectionReason = (
            LocalMicrophoneTranscriptRejectionReason.NONE
        ),
    ) -> None:
        """Retain bounded content-free transcript pipeline counters."""
        counts = (
            asr_segment_count,
            partial_event_count,
            stable_commit_count,
            rejected_event_count,
        )
        if (
            type(asr_result_non_empty) is not bool
            or any(
                type(value) is not int or value < 0 or value > 64 for value in counts
            )
            or not isfinite(window_duration_seconds)
            or window_duration_seconds < 0
            or not isinstance(
                rejection_reason,
                LocalMicrophoneTranscriptRejectionReason,
            )
            or (
                rejected_event_count == 0
                and rejection_reason
                is not LocalMicrophoneTranscriptRejectionReason.NONE
            )
            or (
                rejected_event_count > 0
                and rejection_reason is LocalMicrophoneTranscriptRejectionReason.NONE
            )
        ):
            raise ValueError("invalid_local_microphone_transcript_diagnostic")
        with self._condition:
            self._require_resource(resource)
            if asr_result_non_empty:
                self._asr_non_empty_result_count = _bounded_counter_increment(
                    self._asr_non_empty_result_count
                )
            else:
                self._asr_empty_result_count = _bounded_counter_increment(
                    self._asr_empty_result_count
                )
            self._asr_segment_count = _bounded_counter_increment(
                self._asr_segment_count,
                asr_segment_count,
            )
            self._latest_window_duration_seconds = window_duration_seconds
            self._partial_event_count = _bounded_counter_increment(
                self._partial_event_count,
                partial_event_count,
            )
            self._stable_commit_count = _bounded_counter_increment(
                self._stable_commit_count,
                stable_commit_count,
            )
            self._rejected_transcript_event_count = _bounded_counter_increment(
                self._rejected_transcript_event_count,
                rejected_event_count,
            )
            if rejection_reason is not LocalMicrophoneTranscriptRejectionReason.NONE:
                self._latest_transcript_rejection_reason = rejection_reason

    def record_final_transcript_event(
        self,
        *,
        resource: object,
        accepted: bool,
        rejection_reason: LocalMicrophoneTranscriptRejectionReason = (
            LocalMicrophoneTranscriptRejectionReason.NONE
        ),
    ) -> None:
        if (
            type(accepted) is not bool
            or not isinstance(
                rejection_reason,
                LocalMicrophoneTranscriptRejectionReason,
            )
            or (
                accepted
                and rejection_reason
                is not LocalMicrophoneTranscriptRejectionReason.NONE
            )
        ):
            raise ValueError("invalid_local_microphone_transcript_diagnostic")
        with self._condition:
            self._require_resource(resource)
            if accepted:
                self._stable_commit_count = _bounded_counter_increment(
                    self._stable_commit_count
                )
            if rejection_reason is not LocalMicrophoneTranscriptRejectionReason.NONE:
                self._rejected_transcript_event_count = _bounded_counter_increment(
                    self._rejected_transcript_event_count
                )
                self._latest_transcript_rejection_reason = rejection_reason

    def accept_frame(
        self,
        frame: av.AudioFrame,
        *,
        arrived_at_utc: datetime | None = None,
        capture_generation: int | None = None,
    ) -> av.AudioFrame:
        now = _aware_utc(arrived_at_utc)
        with self._condition:
            if (
                capture_generation is not None
                and capture_generation != self._capture_generation
            ):
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.STALE_CAPTURE_GENERATION
                )
                raise PermissionError("stale_local_microphone_capture_generation")
            if self._asr_readiness not in {
                LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
                LocalMicrophoneASRReadiness.STREAMING,
            }:
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.ASR_NOT_READY
                )
                raise RuntimeError("local_microphone_asr_not_ready")
            if self._status in {
                LocalMicrophoneStatus.PERMISSION_PENDING,
                LocalMicrophoneStatus.RECONNECTING,
            }:
                self.start(arrived_at_utc=now)
            try:
                self._require_authorized()
            except PermissionError:
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.CAPABILITY_REVOKED
                )
                raise
            if self._status not in {
                LocalMicrophoneStatus.READY,
                LocalMicrophoneStatus.STREAMING,
            }:
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.CAPTURE_NOT_STREAMING
                )
                raise RuntimeError("local_microphone_session_not_streaming")
            try:
                input_energy = _frame_energy(frame)
            except Exception:
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.NORMALIZATION_FAILED
                )
                raise
            self._callback_frame_count = _bounded_counter_increment(
                self._callback_frame_count
            )
            self._input_sample_rate_hz = (
                frame.sample_rate if frame.sample_rate > 0 else None
            )
            channel_count = len(frame.layout.channels)
            self._input_channel_count = channel_count if channel_count > 0 else None
            self._accumulate_input_energy(input_energy)
            try:
                normalized = self._normalizer.normalize(frame)
            except Exception:
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.NORMALIZATION_FAILED
                )
                raise
            self._record_normalized_energy(normalized)
            try:
                for item in normalized:
                    self._append_normalized(item, arrived_at_utc=now)
            except Exception:
                self._record_audio_rejection(
                    LocalMicrophoneAudioRejectionReason.INGRESS_REJECTED
                )
                raise
            if self._capability.active:
                self._status = LocalMicrophoneStatus.STREAMING
            self._condition.notify_all()
        return frame

    def _record_audio_rejection(
        self,
        reason: LocalMicrophoneAudioRejectionReason,
    ) -> None:
        self._rejected_capture_frame_count = _bounded_counter_increment(
            self._rejected_capture_frame_count
        )
        self._latest_audio_rejection_reason = reason

    def _accumulate_input_energy(self, energy: _AudioEnergy) -> None:
        accepted = min(
            energy.sample_count,
            _MAX_LOCAL_MIC_AUDIO_VALUE_COUNT - self._input_sample_count,
        )
        if accepted <= 0:
            return
        retained_fraction = accepted / energy.sample_count
        self._input_sample_count += accepted
        self._pre_resample_square_sum += energy.square_sum * retained_fraction
        self._pre_resample_peak = max(self._pre_resample_peak, energy.peak)

    def _accumulate_output_energy(self, energy: _AudioEnergy) -> None:
        accepted = min(
            energy.sample_count,
            _MAX_LOCAL_MIC_AUDIO_VALUE_COUNT - self._post_resample_sample_count,
        )
        if accepted <= 0:
            return
        retained_fraction = accepted / energy.sample_count
        self._post_resample_sample_count += accepted
        self._post_resample_square_sum += energy.square_sum * retained_fraction
        self._post_resample_peak = max(self._post_resample_peak, energy.peak)
        self._post_resample_nonzero_count = min(
            self._post_resample_nonzero_count
            + round(energy.nonzero_count * retained_fraction),
            self._post_resample_sample_count,
        )
        self._post_resample_clipping_count = min(
            self._post_resample_clipping_count
            + round(energy.clipping_count * retained_fraction),
            self._post_resample_sample_count,
        )

    def _record_normalized_energy(
        self,
        normalized: tuple[_NormalizedAudio, ...],
    ) -> None:
        self._accumulate_output_energy(_pcm_energy(normalized))
        self._produced_pcm_byte_count = min(
            self._produced_pcm_byte_count
            + sum(len(item.pcm_s16le) for item in normalized),
            _MAX_LOCAL_MIC_AUDIO_VALUE_COUNT,
        )

    def pause_capture(
        self,
        *,
        resource: object,
        arrived_at_utc: datetime | None = None,
    ) -> bool:
        """Flush and revoke one capture without ending the retained call."""
        return self._stop_capture(
            resource=resource,
            final_status=LocalMicrophoneStatus.PAUSED,
            arrived_at_utc=arrived_at_utc,
            flush=True,
        )

    def resume_capture(
        self,
        capability: LocalMicTestCapability,
        *,
        resource: object,
        normalizer: _AudioNormalizerProtocol | None = None,
    ) -> bool:
        """Install a fresh exact-scope capture while retaining call state."""
        with self._condition:
            if self._closed or resource is not self._resource:
                raise PermissionError("invalid_local_microphone_capability")
            if self._status in {
                LocalMicrophoneStatus.PERMISSION_PENDING,
                LocalMicrophoneStatus.READY,
                LocalMicrophoneStatus.STREAMING,
                LocalMicrophoneStatus.RECONNECTING,
                LocalMicrophoneStatus.PAUSING,
            }:
                return False
            if self._status not in {
                LocalMicrophoneStatus.PAUSED,
                LocalMicrophoneStatus.PERMISSION_DENIED,
                LocalMicrophoneStatus.DISCONNECTED,
                LocalMicrophoneStatus.FAILED,
            }:
                raise RuntimeError("local_microphone_capture_not_resumable")
            if capability is self._capability or not capability.authorizes(
                tenant_id=self._capability.tenant_id,
                call_id=self._capability.call_id,
                resource=resource,
            ):
                raise PermissionError("invalid_local_microphone_capability")
            if self._capability.active:
                raise RuntimeError("previous_local_microphone_capability_active")
            self._capability = capability
            self._normalizer = normalizer or PyAVLocalMicrophoneNormalizer()
            self._capture_generation += 1
            component_digest = sha256(
                f"{self._provider_stream_id}:{self._capture_generation}".encode("utf-8")
            ).hexdigest()
            self._component_key = f"local-mic-{component_digest[:24]}"
            self._buffer.clear()
            self._buffer_capture_utc = None
            self._first_media_time = None
            self._capture_origin_utc = None
            self._pause_pending = False
            self._capture_stop_status = None
            self._status = LocalMicrophoneStatus.PERMISSION_PENDING
            self._asr_readiness = LocalMicrophoneASRReadiness.READY_TO_CAPTURE
            self._condition.notify_all()
            return True

    def mark_reconnecting(self, *, capture_generation: int | None = None) -> bool:
        """Treat a media-track end as transient until explicit user action."""
        with self._condition:
            if (
                capture_generation is not None
                and capture_generation != self._capture_generation
            ):
                return False
            if self._closed or not self._capability.active:
                return False
            if self._status is LocalMicrophoneStatus.RECONNECTING:
                return False
            if self._status not in {
                LocalMicrophoneStatus.READY,
                LocalMicrophoneStatus.STREAMING,
            }:
                return False
            self._status = LocalMicrophoneStatus.RECONNECTING
            self._reconnect_count += 1
            self._asr_readiness = LocalMicrophoneASRReadiness.READY_TO_CAPTURE
            self._condition.notify_all()
            return True

    def fail_capture(
        self,
        *,
        resource: object,
        status: LocalMicrophoneStatus = LocalMicrophoneStatus.FAILED,
    ) -> bool:
        if status not in {
            LocalMicrophoneStatus.FAILED,
            LocalMicrophoneStatus.PERMISSION_DENIED,
            LocalMicrophoneStatus.DISCONNECTED,
        }:
            raise ValueError("invalid_local_microphone_capture_failure")
        return self._stop_capture(
            resource=resource,
            final_status=status,
            arrived_at_utc=None,
            flush=False,
        )

    def _stop_capture(
        self,
        *,
        resource: object,
        final_status: LocalMicrophoneStatus,
        arrived_at_utc: datetime | None,
        flush: bool,
    ) -> bool:
        now = _aware_utc(arrived_at_utc)
        with self._condition:
            if self._closed or resource is not self._resource:
                raise PermissionError("invalid_local_microphone_capability")
            if self._status in {
                LocalMicrophoneStatus.PAUSING,
                LocalMicrophoneStatus.PAUSED,
            }:
                return False
            if not self._capability.active:
                return False
            self._status = LocalMicrophoneStatus.PAUSING
            flushed: tuple[_NormalizedAudio, ...] = ()
            if flush:
                try:
                    flushed = self._normalizer.flush()
                    self._record_normalized_energy(flushed)
                except Exception:
                    self._record_audio_rejection(
                        LocalMicrophoneAudioRejectionReason.NORMALIZATION_FAILED
                    )
                    final_status = LocalMicrophoneStatus.FAILED
            try:
                for item in flushed:
                    self._append_normalized(item, arrived_at_utc=now)
                self._emit_buffer(arrived_at_utc=now, allow_short=True)
            except Exception:
                self._buffer.clear()
                self._buffer_capture_utc = None
                final_status = LocalMicrophoneStatus.FAILED
            self._pause_count += 1
            self._pause_pending = True
            self._capture_stop_status = final_status
            self._capability.revoke()
            self._asr_readiness = LocalMicrophoneASRReadiness.READY_TO_CAPTURE
            self._first_media_time = None
            self._capture_origin_utc = None
            self._finish_capture_stop_if_drained()
            self._condition.notify_all()
            return True

    def _finish_capture_stop_if_drained(self) -> None:
        if (
            self._pause_pending
            and self._boundary.retained_audio_chunk_count == 0
            and self._in_flight_chunk_count == 0
        ):
            self._pause_pending = False
            self._status = self._capture_stop_status or LocalMicrophoneStatus.PAUSED
            self._capture_stop_status = None

    def acknowledge_processed_chunk(self, *, resource: object) -> None:
        with self._condition:
            self._require_resource(resource)
            if self._in_flight_chunk_count <= 0:
                raise RuntimeError("local_microphone_chunk_not_in_flight")
            self._in_flight_chunk_count -= 1
            self._finish_capture_stop_if_drained()
            self._condition.notify_all()

    def request_stop(self, *, arrived_at_utc: datetime | None = None) -> bool:
        return self.finish_call(
            resource=self._resource,
            arrived_at_utc=arrived_at_utc,
        )

    def finish_call(
        self,
        *,
        resource: object,
        arrived_at_utc: datetime | None = None,
    ) -> bool:
        return self._request_draining_end(
            resource=resource,
            terminal_status=LocalMicrophoneStatus.COMPLETED,
            end_reason=LiveAudioEndReason.COMPLETED,
            arrived_at_utc=arrived_at_utc,
        )

    def _request_draining_end(
        self,
        *,
        resource: object,
        terminal_status: LocalMicrophoneStatus,
        end_reason: LiveAudioEndReason,
        arrived_at_utc: datetime | None,
    ) -> bool:
        now = _aware_utc(arrived_at_utc)
        with self._condition:
            if self._end_reason is not None or self._closed:
                return False
            if resource is not self._resource:
                raise PermissionError("invalid_local_microphone_capability")
            flushed: tuple[_NormalizedAudio, ...] = ()
            if self._capability.active:
                try:
                    flushed = self._normalizer.flush()
                    self._record_normalized_energy(flushed)
                except Exception:
                    self._record_audio_rejection(
                        LocalMicrophoneAudioRejectionReason.NORMALIZATION_FAILED
                    )
                    self._cancel(LocalMicrophoneTerminalReason.FAILED)
                    return False
            if self._boundary.active_stream_count == 0:
                self._cancel(LocalMicrophoneTerminalReason.COMPLETED)
                return True
            for item in flushed:
                self._append_normalized(item, arrived_at_utc=now)
            self._emit_buffer(arrived_at_utc=now, allow_short=True)
            if self._closed:
                return False
            self._status = LocalMicrophoneStatus.STOP_REQUESTED
            self._pause_pending = False
            self._capture_stop_status = None
            self._terminal_status = terminal_status
            self._end_reason = end_reason
            self._capability.revoke()
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
                    self._in_flight_chunk_count += 1
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
        del arrived_at_utc
        return self.mark_reconnecting()

    def fail(self) -> None:
        with self._condition:
            self._asr_readiness = LocalMicrophoneASRReadiness.FAILED
        self._cancel(LocalMicrophoneTerminalReason.FAILED)

    def deny_permission(self) -> None:
        self.fail_capture(
            resource=self._resource,
            status=LocalMicrophoneStatus.PERMISSION_DENIED,
        )

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

    def _require_resource(self, resource: object) -> None:
        if self._closed or resource is not self._resource:
            raise PermissionError("invalid_local_microphone_capability")


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


def _frame_energy(frame: av.AudioFrame) -> _AudioEnergy:
    if frame.samples == 0:
        return _AudioEnergy(0, 0.0, 0.0, 0, 0)
    samples = frame.to_ndarray()
    if np.issubdtype(samples.dtype, np.floating):
        normalized = samples.astype(np.float64, copy=False)
    elif np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype.name)
        scale = float(max(abs(info.min), info.max))
        normalized = samples.astype(np.float64) / scale
    elif np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype.name)
        midpoint = (float(info.max) + 1.0) / 2.0
        normalized = (samples.astype(np.float64) - midpoint) / midpoint
    else:
        raise TypeError("unsupported_local_microphone_sample_format")
    return _array_energy(normalized)


def _pcm_energy(normalized: tuple[_NormalizedAudio, ...]) -> _AudioEnergy:
    result = _AudioEnergy(0, 0.0, 0.0, 0, 0)
    for item in normalized:
        samples = np.frombuffer(item.pcm_s16le, dtype="<i2").astype(np.float64)
        result = _combine_energy(result, _array_energy(samples / 32768.0))
    return result


def _array_energy(samples: np.ndarray) -> _AudioEnergy:
    flattened = samples.reshape(-1)
    if flattened.size == 0:
        return _AudioEnergy(0, 0.0, 0.0, 0, 0)
    if not bool(np.isfinite(flattened).all()):
        raise ValueError("non_finite_local_microphone_samples")
    absolute = np.abs(flattened)
    return _AudioEnergy(
        sample_count=int(flattened.size),
        square_sum=float(np.dot(flattened, flattened)),
        peak=float(absolute.max()),
        nonzero_count=int(np.count_nonzero(flattened)),
        clipping_count=int(np.count_nonzero(absolute >= (32767.0 / 32768.0))),
    )


def _combine_energy(first: _AudioEnergy, second: _AudioEnergy) -> _AudioEnergy:
    return _AudioEnergy(
        sample_count=first.sample_count + second.sample_count,
        square_sum=first.square_sum + second.square_sum,
        peak=max(first.peak, second.peak),
        nonzero_count=first.nonzero_count + second.nonzero_count,
        clipping_count=first.clipping_count + second.clipping_count,
    )


def _rms(square_sum: float, sample_count: int) -> float:
    return sqrt(square_sum / sample_count) if sample_count else 0.0


def _ratio(count: int, sample_count: int) -> float:
    return count / sample_count if sample_count else 0.0


def _bounded_counter_increment(current: int, increment: int = 1) -> int:
    return min(current + increment, _MAX_LOCAL_MIC_DIAGNOSTIC_COUNT)
