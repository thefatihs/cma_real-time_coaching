"""Tenant-safe in-memory rolling storage for ordered audio chunk events."""

from collections import deque
from math import isfinite

from app.events.models import AudioChunkEvent


class RollingAudioBuffer:
    """Retain a time-bounded sequence of audio chunks for one call."""

    def __init__(self, window_seconds: float = 20.0) -> None:
        if not isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self.window_seconds = window_seconds
        self._events: deque[AudioChunkEvent] = deque()
        self._tenant_id: str | None = None
        self._call_id: str | None = None
        self._sample_rate_hz: int | None = None
        self._channel_count: int | None = None
        self._codec_name: str | None = None
        self._last_sequence: int | None = None
        self._last_end_seconds: float | None = None

    def append(self, event: AudioChunkEvent) -> None:
        """Validate and append an event, then evict chunks outside the window."""
        if self._last_sequence is None:
            self._bind(event)
        else:
            self._validate_binding(event)
            expected_sequence = self._last_sequence + 1
            if event.sequence_number != expected_sequence:
                raise ValueError(
                    "Audio sequence numbers must be contiguous and strictly increasing"
                )
            assert self._last_end_seconds is not None
            if event.chunk_start_seconds < self._last_end_seconds:
                raise ValueError(
                    "Audio chunk timestamps cannot overlap or move backward"
                )

        self._events.append(event)
        self._last_sequence = event.sequence_number
        self._last_end_seconds = _event_end(event)
        self._evict_old_events()

    def clear(self) -> None:
        """Remove all events and reset call, format, and sequence binding."""
        self._events.clear()
        self._tenant_id = None
        self._call_id = None
        self._sample_rate_hz = None
        self._channel_count = None
        self._codec_name = None
        self._last_sequence = None
        self._last_end_seconds = None

    def events(self) -> tuple[AudioChunkEvent, ...]:
        return tuple(self._events)

    @property
    def is_empty(self) -> bool:
        return not self._events

    @property
    def chunk_count(self) -> int:
        return len(self._events)

    @property
    def start_seconds(self) -> float | None:
        return self._events[0].chunk_start_seconds if self._events else None

    @property
    def end_seconds(self) -> float | None:
        return _event_end(self._events[-1]) if self._events else None

    @property
    def duration_seconds(self) -> float | None:
        if self.start_seconds is None or self.end_seconds is None:
            return None
        return self.end_seconds - self.start_seconds

    @property
    def first_sequence(self) -> int | None:
        return self._events[0].sequence_number if self._events else None

    @property
    def last_sequence(self) -> int | None:
        return self._events[-1].sequence_number if self._events else None

    def _bind(self, event: AudioChunkEvent) -> None:
        self._tenant_id = event.tenant_id
        self._call_id = event.call_id
        self._sample_rate_hz = event.sample_rate_hz
        self._channel_count = event.channel_count
        self._codec_name = event.codec_name

    def _validate_binding(self, event: AudioChunkEvent) -> None:
        if event.tenant_id != self._tenant_id:
            raise ValueError("Audio chunk tenant_id does not match buffer binding")
        if event.call_id != self._call_id:
            raise ValueError("Audio chunk call_id does not match buffer binding")
        if event.sample_rate_hz != self._sample_rate_hz:
            raise ValueError("Audio chunk sample_rate_hz does not match buffer format")
        if event.channel_count != self._channel_count:
            raise ValueError("Audio chunk channel_count does not match buffer format")
        if event.codec_name != self._codec_name:
            raise ValueError("Audio chunk codec_name does not match buffer format")

    def _evict_old_events(self) -> None:
        assert self._last_end_seconds is not None
        cutoff = self._last_end_seconds - self.window_seconds
        while self._events and _event_end(self._events[0]) <= cutoff:
            self._events.popleft()


def _event_end(event: AudioChunkEvent) -> float:
    return event.chunk_start_seconds + event.chunk_duration_seconds
