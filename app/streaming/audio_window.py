"""Build contiguous, tenant-aware PCM windows for ASR input."""

from pydantic import BaseModel, ConfigDict, Field

from app.events.models import canonical_audio_codec_name
from app.streaming.rolling_buffer import RollingAudioBuffer


class ASRAudioWindow(BaseModel):
    """Immutable PCM audio and safe metadata for one ASR invocation."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    call_id: str
    first_sequence: int
    last_sequence: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    sample_rate_hz: int
    channel_count: int
    codec_name: str
    pcm_bytes: bytes = Field(repr=False)

    def metadata_summary(self) -> dict[str, object]:
        """Return metadata suitable for diagnostics without raw PCM data."""
        return {
            "tenant_id": self.tenant_id,
            "call_id": self.call_id,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_count": self.channel_count,
            "codec_name": self.codec_name,
        }


class AudioWindowBuilder:
    """Convert rolling PCM chunks into one exact, frame-aligned window."""

    _SUPPORTED_CODEC = "pcm_s16le"
    _BYTES_PER_SAMPLE = 2

    def build(self, buffer: RollingAudioBuffer) -> ASRAudioWindow:
        events = buffer.events()
        if not events:
            raise ValueError("Cannot build an ASR audio window from an empty buffer")

        first = events[0]
        codec_name = canonical_audio_codec_name(first.codec_name)
        if codec_name != self._SUPPORTED_CODEC:
            raise ValueError(
                f"Unsupported audio codec {first.codec_name!r}; "
                f"only {self._SUPPORTED_CODEC!r} is supported"
            )

        frame_size = self._BYTES_PER_SAMPLE * first.channel_count
        for event in events:
            if len(event.audio_bytes) % frame_size:
                raise ValueError(
                    "PCM audio bytes must contain complete interleaved channel frames"
                )

        pcm_bytes = b"".join(event.audio_bytes for event in events)
        available_frames = len(pcm_bytes) // frame_size
        window_frames = round(buffer.window_seconds * first.sample_rate_hz)
        retained_frames = min(available_frames, window_frames)
        trim_bytes = (available_frames - retained_frames) * frame_size
        pcm_bytes = pcm_bytes[trim_bytes:]

        skipped_frames = available_frames - retained_frames
        first_sequence = first.sequence_number
        for event in events:
            event_frames = len(event.audio_bytes) // frame_size
            if skipped_frames < event_frames:
                break
            skipped_frames -= event_frames
            first_sequence = event.sequence_number + 1

        last = events[-1]
        end_seconds = last.chunk_start_seconds + last.chunk_duration_seconds
        duration_seconds = retained_frames / first.sample_rate_hz
        return ASRAudioWindow(
            tenant_id=first.tenant_id,
            call_id=first.call_id,
            first_sequence=first_sequence,
            last_sequence=last.sequence_number,
            start_seconds=end_seconds - duration_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration_seconds,
            sample_rate_hz=first.sample_rate_hz,
            channel_count=first.channel_count,
            codec_name=codec_name,
            pcm_bytes=pcm_bytes,
        )
