from __future__ import annotations

import pickle
import fractions
import sys
from datetime import UTC, datetime, timedelta
from threading import Event
from types import SimpleNamespace

import av
import numpy as np
import pytest

from app.audio_ingress.local_microphone import (
    LOCAL_MIC_CHUNK_BYTES,
    LOCAL_MIC_GATE_ENVIRONMENT_KEY,
    LocalMicrophoneASRReadiness,
    LocalMicrophoneAudioRejectionReason,
    LocalMicrophoneIngressSession,
    LocalMicrophoneStatus,
    LocalMicrophoneTerminalReason,
    LocalMicrophoneTranscriptRejectionReason,
    PyAVLocalMicrophoneNormalizer,
    _NormalizedAudio,
    create_local_mic_test_capability,
    local_microphone_test_enabled,
)
from app.streaming.audio_window import ASRAudioWindow
from app.streaming.window_transcriber import prepare_whisper_waveform
from live_dashboard.local_microphone import (
    LOCAL_MIC_STATUS_TEXT,
    LOCAL_MIC_WARNING_LINES,
    LocalMicrophoneFrameCallback,
    microphone_webrtc_streamer,
)


NOW = datetime(2026, 7, 31, 8, 0, tzinfo=UTC)
ENABLED = {LOCAL_MIC_GATE_ENVIRONMENT_KEY: "1"}


class SyntheticNormalizer:
    def __init__(self, payloads: list[bytes]) -> None:
        self._payloads = payloads

    def normalize(self, frame: av.AudioFrame) -> tuple[_NormalizedAudio, ...]:
        if not self._payloads:
            return ()
        return (_NormalizedAudio(self._payloads.pop(0), float(frame.pts or 0)),)

    def flush(self) -> tuple[_NormalizedAudio, ...]:
        return ()


class FlushTrackingNormalizer(SyntheticNormalizer):
    def __init__(self, payloads: list[bytes]) -> None:
        super().__init__(payloads)
        self.flush_calls = 0

    def flush(self) -> tuple[_NormalizedAudio, ...]:
        self.flush_calls += 1
        return ()


def audio_frame(
    *,
    samples: int = 960,
    sample_rate: int = 48_000,
    layout: str = "stereo",
    pts: int = 0,
) -> av.AudioFrame:
    channels = 2 if layout == "stereo" else 1
    data = np.zeros((channels, samples), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(data, format="s16p", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, sample_rate)
    return frame


def tone_audio_frame(
    *,
    sample_format: str,
    layout: str = "stereo",
    samples: int = 4_800,
    sample_rate: int = 48_000,
    amplitude: float = 0.5,
    pts: int = 0,
) -> av.AudioFrame:
    channels = 2 if layout == "stereo" else 1
    signal = amplitude * np.sin(
        2.0 * np.pi * 440.0 * np.arange(samples, dtype=np.float64) / sample_rate
    )
    if sample_format in {"flt", "fltp"}:
        channel_data = np.tile(signal.astype(np.float32), (channels, 1))
    else:
        channel_data = np.tile(
            np.rint(signal * 32767.0).astype(np.int16),
            (channels, 1),
        )
    data = (
        channel_data if sample_format.endswith("p") else channel_data.T.reshape(1, -1)
    )
    frame = av.AudioFrame.from_ndarray(
        data,
        format=sample_format,
        layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, sample_rate)
    return frame


def capability(resource: object, **changes: object):
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "resource": resource,
        "server_address": "127.0.0.1",
        "environment": ENABLED,
    }
    values.update(changes)
    return create_local_mic_test_capability(**values)  # type: ignore[arg-type]


def session(
    *,
    payloads: list[bytes],
    max_queue_depth: int = 8,
    provider_stream_id: str = "synthetic-stream",
    asr_ready: bool = True,
) -> tuple[object, LocalMicrophoneIngressSession]:
    resource = object()
    subject = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        provider_stream_id=provider_stream_id,
        normalizer=SyntheticNormalizer(payloads),
        max_queue_depth=max_queue_depth,
    )
    if asr_ready:
        subject.set_asr_readiness(
            LocalMicrophoneASRReadiness.WARMING_UP,
            resource=resource,
        )
        subject.set_asr_readiness(
            LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
            resource=resource,
        )
    return resource, subject


def test_feature_is_disabled_by_default_and_public_hosts_fail_closed() -> None:
    assert not local_microphone_test_enabled(
        server_address="127.0.0.1",
        environment={},
    )
    for address in (None, "", "0.0.0.0", "192.168.1.20", "public.example"):
        assert not local_microphone_test_enabled(
            server_address=address,
            environment=ENABLED,
        )
    assert local_microphone_test_enabled(
        server_address="localhost",
        environment=ENABLED,
    )
    assert local_microphone_test_enabled(
        server_address="::1",
        environment=ENABLED,
    )


def test_capability_is_exact_scope_non_serializable_and_revocable() -> None:
    resource = object()
    subject = capability(resource)

    assert subject.authorizes(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=resource,
    )
    assert not subject.authorizes(
        tenant_id="tenant_beta",
        call_id="call_001",
        resource=resource,
    )
    assert not subject.authorizes(
        tenant_id="tenant_alpha",
        call_id="call_001",
        resource=object(),
    )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(subject)

    subject.revoke()
    assert not subject.active


def test_frames_are_not_normalized_or_admitted_before_asr_readiness() -> None:
    _resource, subject = session(
        payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES],
        asr_ready=False,
    )

    with pytest.raises(RuntimeError, match="asr_not_ready"):
        subject.accept_frame(audio_frame(), arrived_at_utc=NOW)

    diagnostics = subject.diagnostics
    assert diagnostics.asr_readiness is LocalMicrophoneASRReadiness.PREPARING_MODEL
    assert diagnostics.received_chunk_count == 0
    assert diagnostics.queue_depth == 0
    assert diagnostics.status is LocalMicrophoneStatus.PERMISSION_PENDING


def test_capture_starts_only_after_ready_to_capture() -> None:
    resource, subject = session(
        payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES],
        asr_ready=False,
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    with pytest.raises(RuntimeError, match="asr_not_ready"):
        subject.start(arrived_at_utc=NOW)

    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )
    assert subject.start(arrived_at_utc=NOW)
    assert subject.diagnostics.asr_readiness is LocalMicrophoneASRReadiness.STREAMING


def test_asr_preparation_and_inference_timings_are_bounded_aggregates() -> None:
    resource, subject = session(payloads=[])

    subject.record_asr_preparation(
        resource=resource,
        engine_construction_seconds=0.01,
        model_loading_seconds=3.0,
        warmup_seconds=0.5,
    )
    subject.record_asr_inference(
        resource=resource,
        audio_preparation_seconds=0.02,
        inference_seconds=0.8,
    )
    subject.record_asr_inference(
        resource=resource,
        audio_preparation_seconds=0.03,
        inference_seconds=0.6,
    )

    timings = subject.diagnostics.asr_timings
    assert timings.engine_construction_seconds == pytest.approx(0.01)
    assert timings.model_loading_seconds == pytest.approx(3.0)
    assert timings.warmup_seconds == pytest.approx(0.5)
    assert timings.first_audio_preparation_seconds == pytest.approx(0.02)
    assert timings.first_inference_seconds == pytest.approx(0.8)
    assert timings.latest_audio_preparation_seconds == pytest.approx(0.03)
    assert timings.latest_inference_seconds == pytest.approx(0.6)
    assert timings.inference_count == 2


def test_transcript_diagnostics_are_content_free_bounded_and_scope_safe() -> None:
    resource, subject = session(payloads=[])

    subject.record_transcript_step(
        resource=resource,
        asr_result_non_empty=False,
        partial_event_count=0,
        stable_commit_count=0,
    )
    subject.record_transcript_step(
        resource=resource,
        asr_result_non_empty=True,
        asr_segment_count=2,
        window_duration_seconds=4.0,
        partial_event_count=1,
        stable_commit_count=1,
        rejected_event_count=1,
        rejection_reason=(LocalMicrophoneTranscriptRejectionReason.REVISION_REGRESSION),
    )
    subject.record_final_transcript_event(resource=resource, accepted=True)
    subject.record_final_transcript_event(
        resource=resource,
        accepted=False,
        rejection_reason=(
            LocalMicrophoneTranscriptRejectionReason.EVENT_SCOPE_MISMATCH
        ),
    )

    diagnostics = subject.diagnostics
    assert diagnostics.asr_empty_result_count == 1
    assert diagnostics.asr_non_empty_result_count == 1
    assert diagnostics.asr_segment_count == 2
    assert diagnostics.latest_window_duration_seconds == 4.0
    assert diagnostics.partial_event_count == 1
    assert diagnostics.stable_commit_count == 2
    assert diagnostics.rejected_transcript_event_count == 2
    assert (
        diagnostics.latest_transcript_rejection_reason
        is LocalMicrophoneTranscriptRejectionReason.EVENT_SCOPE_MISMATCH
    )
    assert "synthetic transcript" not in repr(diagnostics)

    with pytest.raises(PermissionError, match="invalid_local_microphone"):
        subject.record_transcript_step(
            resource=object(),
            asr_result_non_empty=True,
            partial_event_count=0,
            stable_commit_count=0,
        )
    with pytest.raises(ValueError, match="transcript_diagnostic"):
        subject.record_transcript_step(
            resource=resource,
            asr_result_non_empty=True,
            partial_event_count=0,
            stable_commit_count=0,
            rejected_event_count=1,
        )


def test_pyav_normalizes_stereo_48khz_to_mono_pcm16_16khz() -> None:
    normalizer = PyAVLocalMicrophoneNormalizer()
    outputs = (
        *normalizer.normalize(audio_frame(samples=4_800)),
        *normalizer.flush(),
    )
    payload = b"".join(item.pcm_s16le for item in outputs)

    assert payload
    assert len(payload) % 2 == 0
    assert len(payload) / 2 == pytest.approx(1_600, abs=2)


@pytest.mark.parametrize("sample_format", ["fltp", "flt", "s16p", "s16"])
def test_pyav_preserves_nonzero_amplitude_across_frame_layouts(
    sample_format: str,
) -> None:
    normalizer = PyAVLocalMicrophoneNormalizer()
    outputs = (
        *normalizer.normalize(tone_audio_frame(sample_format=sample_format)),
        *normalizer.flush(),
    )
    samples = np.frombuffer(
        b"".join(item.pcm_s16le for item in outputs),
        dtype="<i2",
    )

    assert samples.size == pytest.approx(1_600, abs=2)
    assert np.max(np.abs(samples)) == pytest.approx(16_384, abs=80)
    assert np.sqrt(np.mean(np.square(samples.astype(np.float64)))) == pytest.approx(
        11_585,
        abs=100,
    )


def test_float_to_pcm16_scaling_clips_without_wrapping() -> None:
    frame = av.AudioFrame.from_ndarray(
        np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32),
        format="flt",
        layout="mono",
    )
    frame.sample_rate = 16_000
    output = PyAVLocalMicrophoneNormalizer().normalize(frame)
    samples = np.frombuffer(output[0].pcm_s16le, dtype="<i2")

    assert samples.tolist() == [32767, -32768, 16384, -16384]


def test_synthetic_non_silent_pcm_reaches_whisper_adapter_at_original_scale() -> None:
    pcm = np.rint(
        0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(32_000) / 16_000) * 32767.0
    ).astype("<i2")
    window = ASRAudioWindow(
        tenant_id="tenant_alpha",
        call_id="call_001",
        first_sequence=1,
        last_sequence=1,
        start_seconds=0.0,
        end_seconds=2.0,
        duration_seconds=2.0,
        sample_rate_hz=16_000,
        channel_count=1,
        codec_name="pcm_s16le",
        pcm_bytes=pcm.tobytes(),
    )

    waveform = prepare_whisper_waveform(window)

    assert waveform.dtype == np.float32
    assert np.max(np.abs(waveform)) == pytest.approx(0.5, abs=0.001)
    assert np.sqrt(np.mean(np.square(waveform.astype(np.float64)))) == pytest.approx(
        0.3535,
        abs=0.001,
    )


def test_silence_energy_diagnostics_remain_zero() -> None:
    resource = object()
    subject = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        normalizer=PyAVLocalMicrophoneNormalizer(),
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )

    subject.accept_frame(audio_frame(samples=4_800), arrived_at_utc=NOW)
    diagnostics = subject.diagnostics

    assert diagnostics.callback_frame_count == 1
    assert diagnostics.input_sample_rate_hz == 48_000
    assert diagnostics.input_channel_count == 2
    assert diagnostics.input_sample_count == 9_600
    assert diagnostics.pre_resample_rms == 0.0
    assert diagnostics.pre_resample_peak == 0.0
    assert diagnostics.post_resample_rms == 0.0
    assert diagnostics.post_resample_peak == 0.0
    assert diagnostics.post_resample_nonzero_ratio == 0.0
    assert diagnostics.produced_pcm_byte_count > 0


def test_non_silent_energy_diagnostics_are_content_free_aggregates() -> None:
    resource = object()
    subject = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        normalizer=PyAVLocalMicrophoneNormalizer(),
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )

    subject.accept_frame(
        tone_audio_frame(sample_format="fltp"),
        arrived_at_utc=NOW,
    )
    diagnostics = subject.diagnostics

    assert diagnostics.pre_resample_rms == pytest.approx(0.3535, abs=0.001)
    assert diagnostics.pre_resample_peak == pytest.approx(0.5, abs=0.001)
    assert diagnostics.post_resample_rms == pytest.approx(0.3535, abs=0.003)
    assert diagnostics.post_resample_peak == pytest.approx(0.5, abs=0.003)
    assert diagnostics.post_resample_nonzero_ratio > 0.99
    assert diagnostics.post_resample_clipping_ratio == 0.0
    assert "pcm_s16le" not in repr(diagnostics)


def test_live_chunk_is_available_before_end_and_scope_is_preserved() -> None:
    _resource, subject = session(payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES])
    frame = audio_frame()
    subject.start(arrived_at_utc=NOW)
    subject.accept_frame(frame, arrived_at_utc=NOW + timedelta(milliseconds=20))

    assert subject.diagnostics.status is LocalMicrophoneStatus.STREAMING
    assert subject.diagnostics.received_chunk_count == 1
    assert not subject.diagnostics.end_emitted

    subject.request_stop(arrived_at_utc=NOW + timedelta(seconds=2))
    chunks = tuple(subject.iter_audio_chunks(cancellation=Event()))
    assert len(chunks) == 1
    assert chunks[0].tenant_id == "tenant_alpha"
    assert chunks[0].call_id == "call_001"
    assert chunks[0].codec_name == "pcm_s16le"
    assert chunks[0].sample_rate_hz == 16_000
    assert chunks[0].channel_count == 1
    assert subject.diagnostics.end_emitted
    assert not subject.capability.active


def test_duplicate_start_is_idempotent_and_does_not_advance_sequence() -> None:
    _resource, subject = session(payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES])

    assert subject.start(arrived_at_utc=NOW)
    assert not subject.start(arrived_at_utc=NOW)
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW)
    subject.request_stop(arrived_at_utc=NOW + timedelta(seconds=2))

    chunks = tuple(subject.iter_audio_chunks(cancellation=Event()))
    assert [chunk.sequence_number for chunk in chunks] == [1]


def test_short_terminal_chunk_flushes_and_end_is_exactly_once() -> None:
    _resource, subject = session(payloads=[b"\1\0" * 8_000])
    subject.start(arrived_at_utc=NOW)
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW)

    assert subject.diagnostics.received_chunk_count == 0
    assert subject.request_stop(arrived_at_utc=NOW + timedelta(seconds=1))
    assert not subject.request_stop(arrived_at_utc=NOW + timedelta(seconds=1))
    chunks = tuple(subject.iter_audio_chunks(cancellation=Event()))

    assert len(chunks) == 1
    assert chunks[0].chunk_duration_seconds == pytest.approx(0.5)
    assert subject.diagnostics.end_emitted


def test_transient_audio_end_enters_reconnecting_without_ending_call() -> None:
    _resource, subject = session(payloads=[b"\1\0" * 8_000, b""])
    callback = LocalMicrophoneFrameCallback(subject)
    subject.start(arrived_at_utc=NOW)
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW + timedelta(seconds=1))

    callback.on_audio_ended()
    callback.on_audio_ended()
    assert subject.diagnostics.status is LocalMicrophoneStatus.RECONNECTING
    assert subject.diagnostics.reconnect_count == 1
    assert subject.diagnostics.queue_depth == 0
    assert subject.capability.active
    assert not subject.diagnostics.end_emitted
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW + timedelta(seconds=2))
    assert subject.diagnostics.status is LocalMicrophoneStatus.STREAMING
    assert subject.diagnostics.reconnect_count == 1


def test_silence_does_not_pause_or_end_capture() -> None:
    _resource, subject = session(payloads=[])

    subject.accept_frame(audio_frame(), arrived_at_utc=NOW)
    for index in range(5):
        subject.accept_frame(
            audio_frame(pts=index + 1),
            arrived_at_utc=NOW + timedelta(seconds=index + 1),
        )

    assert subject.diagnostics.status is LocalMicrophoneStatus.STREAMING
    assert subject.diagnostics.received_chunk_count == 0
    assert subject.diagnostics.queue_depth == 0
    assert not subject.diagnostics.end_emitted
    assert subject.capability.active


def test_manual_pause_flushes_once_and_resume_keeps_call_without_replay() -> None:
    resource = object()
    normalizer = FlushTrackingNormalizer([b"\1\0" * 8_000, b"\2\0" * 32_000])
    subject = LocalMicrophoneIngressSession(
        capability=capability(resource),
        resource=resource,
        provider_stream_id="persistent-call-stream",
        normalizer=normalizer,
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.WARMING_UP,
        resource=resource,
    )
    subject.set_asr_readiness(
        LocalMicrophoneASRReadiness.READY_TO_CAPTURE,
        resource=resource,
    )
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW)
    old_capability = subject.capability
    old_component_key = subject.component_key

    assert subject.pause_capture(
        resource=resource,
        arrived_at_utc=NOW + timedelta(seconds=1),
    )
    assert not subject.pause_capture(resource=resource)
    assert normalizer.flush_calls == 1
    assert subject.diagnostics.status is LocalMicrophoneStatus.PAUSING
    assert not old_capability.active
    assert not subject.diagnostics.end_emitted

    chunks = iter(subject.iter_audio_chunks(cancellation=Event()))
    first = next(chunks)
    assert first.call_id == "call_001"
    assert first.sequence_number == 1
    assert first.chunk_duration_seconds == pytest.approx(0.5)
    subject.acknowledge_processed_chunk(resource=resource)
    assert subject.diagnostics.status is LocalMicrophoneStatus.PAUSED

    new_capability = capability(resource)
    assert subject.resume_capture(
        new_capability,
        resource=resource,
        normalizer=SyntheticNormalizer([b"\2\0" * 32_000]),
    )
    assert not subject.resume_capture(
        new_capability,
        resource=resource,
    )
    assert subject.capability is new_capability
    assert new_capability.active
    assert subject.component_key != old_component_key
    assert subject.diagnostics.capture_generation == 2
    assert subject.diagnostics.received_chunk_count == 1

    subject.accept_frame(audio_frame(), arrived_at_utc=NOW + timedelta(seconds=2))
    second = next(chunks)
    assert second.call_id == first.call_id
    assert second.sequence_number == 2
    assert second.chunk_start_seconds == pytest.approx(first.chunk_duration_seconds)
    subject.acknowledge_processed_chunk(resource=resource)
    assert subject.diagnostics.received_chunk_count == 2

    assert subject.pause_capture(resource=resource)
    assert subject.diagnostics.status is LocalMicrophoneStatus.PAUSED
    assert not subject.pause_capture(resource=resource)
    third_capability = capability(resource)
    assert subject.resume_capture(
        third_capability,
        resource=resource,
        normalizer=SyntheticNormalizer([]),
    )
    assert not new_capability.active
    assert third_capability.active
    assert subject.diagnostics.capture_generation == 3
    assert subject.diagnostics.pause_count == 2
    assert subject.diagnostics.received_chunk_count == 2

    assert subject.finish_call(
        resource=resource,
        arrived_at_utc=NOW + timedelta(seconds=4),
    )
    with pytest.raises(StopIteration):
        next(chunks)
    assert subject.diagnostics.end_emitted
    assert not third_capability.active


def test_permission_failure_preserves_call_and_allows_fresh_capability() -> None:
    resource, subject = session(payloads=[])
    old_capability = subject.capability

    subject.deny_permission()

    assert subject.diagnostics.status is LocalMicrophoneStatus.PERMISSION_DENIED
    assert not subject.diagnostics.end_emitted
    assert not old_capability.active
    replacement = capability(resource)
    assert subject.resume_capture(replacement, resource=resource)
    assert subject.capability is replacement
    assert subject.diagnostics.status is LocalMicrophoneStatus.PERMISSION_PENDING

    subject.accept_frame(audio_frame(), arrived_at_utc=NOW + timedelta(seconds=2))
    assert subject.diagnostics.status is LocalMicrophoneStatus.STREAMING
    assert subject.capability.active
    assert not subject.diagnostics.end_emitted


def test_reset_deterministically_rejects_queued_audio() -> None:
    _resource, subject = session(payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES])
    subject.start(arrived_at_utc=NOW)
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW)

    subject.close(LocalMicrophoneTerminalReason.RESET)

    assert tuple(subject.iter_audio_chunks(cancellation=Event())) == ()
    assert subject.diagnostics.queue_depth == 0
    assert subject.diagnostics.end_emitted
    assert not subject.capability.active


def test_bounded_eight_chunk_queue_overload_revokes_capability() -> None:
    _resource, subject = session(
        payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES for _ in range(9)]
    )
    subject.start(arrived_at_utc=NOW)
    for index in range(9):
        if not subject.capability.active:
            break
        subject.accept_frame(
            audio_frame(pts=index),
            arrived_at_utc=NOW + timedelta(seconds=index),
        )

    diagnostics = subject.diagnostics
    assert diagnostics.status is LocalMicrophoneStatus.OVERLOADED
    assert diagnostics.queue_depth == 0
    assert diagnostics.end_emitted
    assert not subject.capability.active


@pytest.mark.parametrize(
    ("method", "status"),
    [
        ("fail", LocalMicrophoneStatus.FAILED),
    ],
)
def test_terminal_paths_revoke_without_audio_retention(
    method: str,
    status: LocalMicrophoneStatus,
) -> None:
    _resource, subject = session(payloads=[])
    getattr(subject, method)()

    assert subject.diagnostics.status is status
    assert subject.diagnostics.queue_depth == 0
    assert not subject.capability.active


def test_replacement_cleanup_is_idempotent() -> None:
    _resource, subject = session(payloads=[])
    subject.close(LocalMicrophoneTerminalReason.REPLACED)
    subject.close(LocalMicrophoneTerminalReason.REPLACED)

    assert subject.diagnostics.status is LocalMicrophoneStatus.REPLACED
    assert not subject.capability.active


def test_callback_only_normalizes_and_enqueues() -> None:
    _resource, subject = session(payloads=[b"\0" * LOCAL_MIC_CHUNK_BYTES])
    callback = LocalMicrophoneFrameCallback(subject)
    frame = audio_frame()

    assert callback(frame) is frame
    assert subject.diagnostics.received_chunk_count == 1
    assert not subject.diagnostics.end_emitted


def test_webrtc_component_is_audio_only_local_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resource, subject = session(payloads=[])
    captured: dict[str, object] = {}
    expected_context = object()

    def fake_streamer(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected_context

    monkeypatch.setitem(
        sys.modules,
        "streamlit_webrtc",
        SimpleNamespace(
            WebRtcMode=SimpleNamespace(SENDONLY="SENDONLY"),
            webrtc_streamer=fake_streamer,
        ),
    )

    context = microphone_webrtc_streamer(
        session=subject,
        key="synthetic-local-microphone",
    )

    assert context is expected_context
    assert captured["mode"] == "SENDONLY"
    assert captured["rtc_configuration"] == {"iceServers": []}
    assert captured["media_stream_constraints"] == {
        "video": False,
        "audio": True,
    }
    assert captured["audio_receiver_size"] == 1
    assert captured["sendback_audio"] is False


def test_webrtc_component_identity_and_session_survive_ordinary_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resource, subject = session(payloads=[])
    contexts: dict[str, object] = {}
    callbacks: list[object] = []

    def fake_streamer(**kwargs: object) -> object:
        key = str(kwargs["key"])
        callbacks.append(kwargs["audio_frame_callback"])
        return contexts.setdefault(key, object())

    monkeypatch.setitem(
        sys.modules,
        "streamlit_webrtc",
        SimpleNamespace(
            WebRtcMode=SimpleNamespace(SENDONLY="SENDONLY"),
            webrtc_streamer=fake_streamer,
        ),
    )

    first = microphone_webrtc_streamer(
        session=subject,
        key=subject.component_key,
    )
    second = microphone_webrtc_streamer(
        session=subject,
        key=subject.component_key,
    )

    assert first is second
    assert len(contexts) == 1
    assert callbacks[0] is not callbacks[1]
    assert subject.diagnostics.status is LocalMicrophoneStatus.PERMISSION_PENDING
    assert subject.capability.active


def test_stale_callback_cannot_submit_to_resumed_capture_generation() -> None:
    resource, subject = session(payloads=[])
    old_callback = LocalMicrophoneFrameCallback(subject)
    subject.start(arrived_at_utc=NOW)
    assert subject.pause_capture(resource=resource, arrived_at_utc=NOW)
    replacement = capability(resource)
    assert subject.resume_capture(
        replacement,
        resource=resource,
        normalizer=PyAVLocalMicrophoneNormalizer(),
    )
    new_callback = LocalMicrophoneFrameCallback(subject)
    frame = tone_audio_frame(sample_format="fltp")

    with pytest.raises(PermissionError, match="stale_local_microphone"):
        old_callback(frame)
    after_rejection = subject.diagnostics
    assert after_rejection.callback_frame_count == 0
    assert after_rejection.rejected_capture_frame_count == 1
    assert (
        after_rejection.latest_audio_rejection_reason
        is LocalMicrophoneAudioRejectionReason.STALE_CAPTURE_GENERATION
    )

    assert new_callback(frame) is frame
    diagnostics = subject.diagnostics
    assert diagnostics.callback_frame_count == 1
    assert diagnostics.post_resample_rms > 0.3
    assert diagnostics.post_resample_nonzero_ratio > 0.99


def test_stale_audio_end_does_not_reconnect_current_capture() -> None:
    resource, subject = session(payloads=[])
    old_callback = LocalMicrophoneFrameCallback(subject)
    subject.start(arrived_at_utc=NOW)
    assert subject.pause_capture(resource=resource)
    assert subject.resume_capture(capability(resource), resource=resource)
    new_callback = LocalMicrophoneFrameCallback(subject)

    old_callback.on_audio_ended()
    assert subject.diagnostics.status is LocalMicrophoneStatus.PERMISSION_PENDING
    new_callback(audio_frame())
    assert subject.diagnostics.status is LocalMicrophoneStatus.STREAMING


def test_replacement_session_gets_a_new_component_identity() -> None:
    _first_resource, first = session(
        payloads=[],
        provider_stream_id="synthetic-stream-one",
    )
    _second_resource, second = session(
        payloads=[],
        provider_stream_id="synthetic-stream-two",
    )

    assert first.component_key != second.component_key


def test_turkish_states_and_warnings_are_fixed_and_bounded() -> None:
    assert set(LOCAL_MIC_STATUS_TEXT) == set(LocalMicrophoneStatus)
    assert LOCAL_MIC_STATUS_TEXT[LocalMicrophoneStatus.PERMISSION_PENDING] == (
        "Mikrofon izni bekleniyor"
    )
    assert LOCAL_MIC_STATUS_TEXT[LocalMicrophoneStatus.PERMISSION_DENIED] == (
        "Mikrofon erişimi reddedildi"
    )
    assert LOCAL_MIC_WARNING_LINES == (
        "Yalnızca yerel geliştirme testi",
        "Konuşmacı rolü kalıcı olarak kaydedilmez",
        "Gerçek çağrı veya diarization testi değildir",
    )
