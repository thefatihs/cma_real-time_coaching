"""Presentation-only WebRTC adapter for the localhost microphone test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import av

if TYPE_CHECKING:
    from streamlit_webrtc import WebRtcStreamerContext

from app.audio_ingress.local_microphone import (
    LocalMicrophoneASRReadiness,
    LocalMicrophoneIngressSession,
    LocalMicrophoneStatus,
)


LOCAL_MIC_WARNING_LINES = (
    "Yalnızca yerel geliştirme testi",
    "Konuşmacı rolü kalıcı olarak kaydedilmez",
    "Gerçek çağrı veya diarization testi değildir",
)

LOCAL_MIC_STATUS_TEXT = {
    LocalMicrophoneStatus.PERMISSION_PENDING: "Mikrofon izni bekleniyor",
    LocalMicrophoneStatus.READY: "Mikrofon hazır",
    LocalMicrophoneStatus.STREAMING: "Canlı ses alınıyor",
    LocalMicrophoneStatus.STOP_REQUESTED: "Mikrofon durduruluyor",
    LocalMicrophoneStatus.COMPLETED: "Mikrofon durduruldu",
    LocalMicrophoneStatus.PERMISSION_DENIED: "Mikrofon erişimi reddedildi",
    LocalMicrophoneStatus.OVERLOADED: "Ses kuyruğu kapasitesi aşıldı",
    LocalMicrophoneStatus.DISCONNECTED: "Mikrofon bağlantısı kesildi",
    LocalMicrophoneStatus.FAILED: "Mikrofon testi başarısız",
    LocalMicrophoneStatus.REPLACED: "Mikrofon bağlantısı kesildi",
}


@dataclass(frozen=True, slots=True)
class LocalMicrophoneConnectionView:
    status_text: str
    received_chunk_count: int
    processed_audio_seconds: float
    estimated_latency_seconds: float | None


class LocalMicrophoneFrameCallback:
    """Audio callback restricted to bounded normalization and enqueueing."""

    def __init__(self, session: LocalMicrophoneIngressSession) -> None:
        self._session = session

    def __call__(self, frame: av.AudioFrame) -> av.AudioFrame:
        return self._session.accept_frame(frame)

    def on_audio_ended(self) -> None:
        status = self._session.diagnostics.status
        if status not in {
            LocalMicrophoneStatus.STOP_REQUESTED,
            LocalMicrophoneStatus.COMPLETED,
        }:
            self._session.disconnect()


def local_microphone_connection_view(
    session: LocalMicrophoneIngressSession,
) -> LocalMicrophoneConnectionView:
    diagnostics = session.diagnostics
    status_text = LOCAL_MIC_STATUS_TEXT[diagnostics.status]
    if (
        diagnostics.status is LocalMicrophoneStatus.PERMISSION_PENDING
        and diagnostics.asr_readiness is LocalMicrophoneASRReadiness.READY_TO_CAPTURE
    ):
        status_text = "Mikrofon hazır; konuşabilirsiniz"
    return LocalMicrophoneConnectionView(
        status_text=status_text,
        received_chunk_count=diagnostics.received_chunk_count,
        processed_audio_seconds=diagnostics.processed_audio_seconds,
        estimated_latency_seconds=diagnostics.estimated_latency_seconds,
    )


def microphone_webrtc_streamer(
    *,
    session: LocalMicrophoneIngressSession,
    key: str,
    desired_playing_state: bool | None = None,
) -> WebRtcStreamerContext:
    """Render an audio-only, no-STUN/TURN WebRTC source."""
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    callback = LocalMicrophoneFrameCallback(session)
    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={"video": False, "audio": True},
        desired_playing_state=desired_playing_state,
        audio_frame_callback=callback,
        on_audio_ended=callback.on_audio_ended,
        async_processing=True,
        audio_receiver_size=1,
        sendback_audio=False,
        media_toggle_controls=False,
        translations={
            "start": "Mikrofonu başlat",
            "stop": "Mikrofonu durdur",
            "select_device": "Mikrofon seç",
            "media_api_not_available": "Mikrofon API kullanılamıyor",
            "device_ask_permission": "Mikrofon izni bekleniyor",
            "device_not_available": "Mikrofon bulunamadı",
            "device_access_denied": "Mikrofon erişimi reddedildi",
            "mute_microphone": "Mikrofonu sessize al",
            "unmute_microphone": "Mikrofonu aç",
        },
    )
