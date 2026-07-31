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
    LocalMicrophoneIngressSession,
    LocalMicrophoneStatus,
    LocalMicrophoneTerminalReason,
    PyAVLocalMicrophoneNormalizer,
    _NormalizedAudio,
    create_local_mic_test_capability,
    local_microphone_test_enabled,
)
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


def test_disconnect_drains_queued_audio_before_exactly_once_end() -> None:
    _resource, subject = session(payloads=[b"\1\0" * 8_000])
    callback = LocalMicrophoneFrameCallback(subject)
    subject.start(arrived_at_utc=NOW)
    subject.accept_frame(audio_frame(), arrived_at_utc=NOW + timedelta(seconds=1))

    callback.on_audio_ended()
    callback.on_audio_ended()
    assert subject.diagnostics.status is LocalMicrophoneStatus.STOP_REQUESTED
    assert subject.diagnostics.queue_depth == 1
    assert subject.capability.active

    chunks = tuple(subject.iter_audio_chunks(cancellation=Event()))

    assert len(chunks) == 1
    assert chunks[0].chunk_duration_seconds == pytest.approx(0.5)
    assert subject.diagnostics.status is LocalMicrophoneStatus.DISCONNECTED
    assert subject.diagnostics.end_emitted
    assert not subject.capability.active


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
        ("deny_permission", LocalMicrophoneStatus.PERMISSION_DENIED),
        ("disconnect", LocalMicrophoneStatus.DISCONNECTED),
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
