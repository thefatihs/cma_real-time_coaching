from pathlib import Path
from types import TracebackType
from unittest.mock import patch

import av
import numpy as np
import pytest

from app.streaming.chunk_generator import generate_audio_chunks


class SyntheticStreams:
    def __init__(self) -> None:
        self.audio: list[object] = [object()]


class SyntheticContainer:
    def __init__(self, frames: list[av.AudioFrame]) -> None:
        self._frames = frames
        self.streams = SyntheticStreams()

    def __enter__(self) -> "SyntheticContainer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def decode(self, stream: object) -> list[av.AudioFrame]:
        return self._frames


def make_frame(samples: int, sample_rate: int = 8_000) -> av.AudioFrame:
    data = np.zeros((1, samples), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(data, format="s16", layout="mono")
    frame.sample_rate = sample_rate
    return frame


def test_generates_ordered_chunks_and_short_final_chunk(tmp_path: Path) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.touch()
    container = SyntheticContainer([make_frame(9_000), make_frame(9_000)])

    with patch("app.streaming.chunk_generator.av.open", return_value=container):
        events = list(generate_audio_chunks(audio_path, "tenant-a", "call-a"))

    assert [event.sequence_number for event in events] == [0, 1]
    assert [event.chunk_start_seconds for event in events] == [0.0, 2.0]
    assert [event.chunk_duration_seconds for event in events] == [2.0, 0.25]
    assert all(event.sample_rate_hz == 8_000 for event in events)
    assert all(event.channel_count == 1 for event in events)
    assert all(event.tenant_id == "tenant-a" for event in events)
    assert all(event.call_id == "call-a" for event in events)
    assert len(events[0].audio_bytes) == 16_000 * 2
    assert len(events[1].audio_bytes) == 2_000 * 2


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_duration(tmp_path: Path, duration: float) -> None:
    audio_path = tmp_path / "synthetic.wav"
    audio_path.touch()

    with pytest.raises(ValueError, match="finite and positive"):
        generate_audio_chunks(audio_path, "tenant-a", "call-a", duration)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_audio_chunks(tmp_path / "missing.wav", "tenant-a", "call-a")


def test_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        generate_audio_chunks(tmp_path, "tenant-a", "call-a")


def test_rejects_media_without_audio_stream(tmp_path: Path) -> None:
    audio_path = tmp_path / "synthetic.bin"
    audio_path.touch()
    container = SyntheticContainer([])
    container.streams.audio = []

    with (
        patch("app.streaming.chunk_generator.av.open", return_value=container),
        pytest.raises(ValueError, match="no audio stream"),
    ):
        list(generate_audio_chunks(audio_path, "tenant-a", "call-a"))
