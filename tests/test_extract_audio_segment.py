from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.extract_audio_segment import extract_audio_segment, main


class FakeContainer:
    def __init__(self, audio_streams: list[object] | None = None) -> None:
        self.streams = SimpleNamespace(audio=audio_streams or [])
        self.seek_calls: list[tuple[int, bool]] = []
        self.frames: list[object] = []
        self.output_stream = FakeOutputStream()
        self.muxed_packets: list[object] = []

    def __enter__(self) -> "FakeContainer":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def seek(self, offset: int, *, backward: bool) -> None:
        self.seek_calls.append((offset, backward))

    def decode(self, stream: object) -> list[object]:
        return self.frames

    def add_stream(self, codec_name: str, *, rate: int) -> "FakeOutputStream":
        self.output_stream.codec_name = codec_name
        self.output_stream.rate = rate
        return self.output_stream

    def mux(self, packet: object) -> None:
        self.muxed_packets.append(packet)


class FakeOutputStream:
    def __init__(self) -> None:
        self.codec_name = ""
        self.rate = 0
        self.codec_context = SimpleNamespace(layout=None)

    def encode(self, frame: object | None) -> list[str]:
        return ["flush"] if frame is None else ["packet"]


class FakeResampler:
    def __init__(self, **settings: object) -> None:
        self.settings = settings

    def resample(self, frame: object | None) -> list[object]:
        return [] if frame is None else [frame]


def test_extracts_mono_pcm_s16_at_source_sample_rate() -> None:
    stream = SimpleNamespace(
        codec_context=SimpleNamespace(
            sample_rate=8_000, layout=SimpleNamespace(name="mono")
        )
    )
    input_container = FakeContainer([stream])
    input_container.frames = [
        SimpleNamespace(time=29.0, samples=160, sample_rate=8_000),
        SimpleNamespace(time=30.0, samples=160, sample_rate=8_000),
        SimpleNamespace(time=45.0, samples=160, sample_rate=8_000),
        SimpleNamespace(time=75.0, samples=160, sample_rate=8_000),
    ]
    output_container = FakeContainer()
    opened: list[tuple[tuple[object, ...], dict[str, object]]] = []
    resamplers: list[FakeResampler] = []

    def media_opener(*args: object, **kwargs: object) -> FakeContainer:
        opened.append((args, kwargs))
        return output_container if kwargs.get("mode") == "w" else input_container

    def resampler_factory(**settings: object) -> FakeResampler:
        resampler = FakeResampler(**settings)
        resamplers.append(resampler)
        return resampler

    extract_audio_segment(
        Path("private/call.wav"),
        Path("private/segment.wav"),
        30.0,
        75.0,
        media_opener=media_opener,
        resampler_factory=resampler_factory,
        path_validator=lambda input_path, output_path: None,
    )

    assert input_container.seek_calls
    assert opened[1][1] == {"mode": "w", "format": "wav"}
    assert output_container.output_stream.codec_name == "pcm_s16le"
    assert output_container.output_stream.rate == 8_000
    assert output_container.output_stream.codec_context.layout == "mono"
    assert resamplers[0].settings == {
        "format": "s16",
        "layout": "mono",
        "rate": 8_000,
    }
    assert output_container.muxed_packets == ["packet", "packet", "flush"]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(-1.0, 5.0, "non-negative"), (5.0, 5.0, "greater than start")],
)
def test_invalid_time_range_fails_before_opening_media(
    start: float, end: float, message: str
) -> None:
    def media_opener(*args: object, **kwargs: object) -> Any:
        raise AssertionError("Media must not be opened")

    with pytest.raises(ValueError, match=message):
        extract_audio_segment(
            Path("private/call.wav"),
            Path("private/segment.wav"),
            start,
            end,
            media_opener=media_opener,
            path_validator=lambda input_path, output_path: None,
        )


def test_cli_returns_nonzero_when_extraction_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_extractor(
        input_path: Path, output_path: Path, start: float, end: float
    ) -> None:
        raise ValueError("End time must be greater than start time")

    exit_code = main(
        ["call.wav", "--start", "10", "--end", "5", "--output", "part.wav"],
        extractor=failing_extractor,
    )

    assert exit_code == 1
    assert "greater than start" in capsys.readouterr().err
