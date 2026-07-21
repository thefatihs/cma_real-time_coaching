from contextlib import AbstractContextManager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.audio.metadata import AudioMetadata, inspect_audio_metadata
from scripts.inspect_audio import main


class FakeContainer(AbstractContextManager["FakeContainer"]):
    def __init__(
        self,
        audio_streams: list[object],
        *,
        container_format: object | None = None,
        duration: int | None = None,
        bit_rate: int | None = None,
    ) -> None:
        self.streams = SimpleNamespace(audio=audio_streams)
        self.format = container_format
        self.duration = duration
        self.bit_rate = bit_rate
        self.was_closed = False

    def __exit__(self, *args: object) -> None:
        self.was_closed = True


def make_stream(
    *,
    channels: int = 1,
    layout: str = "mono",
    sample_rate: int = 16_000,
    sample_format: str = "s16",
    bit_rate: int | None = 256_000,
) -> SimpleNamespace:
    codec_context = SimpleNamespace(
        name="pcm_s16le",
        sample_rate=sample_rate,
        channels=channels,
        layout=SimpleNamespace(name=layout),
        format=SimpleNamespace(name=sample_format),
        bit_rate=bit_rate,
    )
    return SimpleNamespace(
        codec_context=codec_context,
        duration=80_000,
        time_base=Fraction(1, 16_000),
        bit_rate=bit_rate,
    )


def inspect_fake_container(
    container: FakeContainer, audio_path: Path = Path("private/call.wav")
) -> AudioMetadata:
    return inspect_audio_metadata(
        audio_path,
        media_opener=lambda path: container,
        path_validator=lambda path: None,
    )


def test_successful_metadata_extraction() -> None:
    container = FakeContainer(
        [make_stream()], container_format=SimpleNamespace(name="wav"), bit_rate=256_000
    )

    metadata = inspect_fake_container(container)

    assert metadata == AudioMetadata(
        file_extension=".wav",
        duration_seconds=5.0,
        container_format="wav",
        codec_name="pcm_s16le",
        sample_rate_hz=16_000,
        channel_count=1,
        channel_layout="mono",
        sample_format="s16",
        bit_rate=256_000,
    )
    assert container.was_closed


@pytest.mark.parametrize(
    ("channels", "layout"),
    [(1, "mono"), (2, "stereo")],
)
def test_channel_metadata(channels: int, layout: str) -> None:
    container = FakeContainer(
        [make_stream(channels=channels, layout=layout)],
        container_format=SimpleNamespace(name="wav"),
    )

    metadata = inspect_fake_container(container)

    assert metadata.channel_count == channels
    assert metadata.channel_layout == layout


def test_missing_optional_metadata() -> None:
    stream = make_stream(bit_rate=None)
    stream.codec_context.name = None
    stream.codec_context.sample_rate = None
    stream.codec_context.channels = None
    stream.codec_context.layout = None
    stream.codec_context.format = None
    stream.duration = None
    stream.time_base = None
    container = FakeContainer([stream])

    metadata = inspect_fake_container(container, Path("private/call"))

    assert metadata.file_extension == ""
    assert metadata.duration_seconds is None
    assert metadata.container_format is None
    assert metadata.codec_name is None
    assert metadata.sample_rate_hz is None
    assert metadata.channel_count is None
    assert metadata.channel_layout is None
    assert metadata.sample_format is None
    assert metadata.bit_rate is None


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("Audio file does not exist"),
        IsADirectoryError("Audio path is a directory"),
    ],
)
def test_path_validation_errors_do_not_open_media(error: OSError) -> None:
    opened = False

    def failing_validator(path: Path) -> None:
        raise error

    def media_opener(path: str) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("Media must not be opened")

    with pytest.raises(type(error)):
        inspect_audio_metadata(
            Path("private/call.wav"),
            media_opener=media_opener,
            path_validator=failing_validator,
        )

    assert not opened


def test_file_without_audio_stream() -> None:
    container = FakeContainer([], container_format=SimpleNamespace(name="mp4"))

    with pytest.raises(ValueError, match="does not contain an audio stream"):
        inspect_fake_container(container, Path("private/video.mp4"))


def test_corrupted_media_returns_nonzero_without_printing_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = "C:/Private/secret-call.wav"

    def failing_inspector(path: Path) -> AudioMetadata:
        raise RuntimeError(f"corrupt media: {path}")

    exit_code = main([private_path], inspector=failing_inspector)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unsupported, corrupted" in captured.err
    assert private_path not in captured.err
    assert captured.out == ""
