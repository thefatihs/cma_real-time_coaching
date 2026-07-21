from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    file_extension: str
    duration_seconds: float | None
    container_format: str | None
    codec_name: str | None
    sample_rate_hz: int | None
    channel_count: int | None
    channel_layout: str | None
    sample_format: str | None
    bit_rate: int | None


MediaOpener = Callable[[str], Any]
PathValidator = Callable[[Path], None]


def validate_audio_path(audio_path: Path) -> None:
    if not audio_path.exists():
        raise FileNotFoundError("Audio file does not exist")
    if audio_path.is_dir():
        raise IsADirectoryError("Audio path is a directory, not a file")


def inspect_audio_metadata(
    audio_path: Path,
    media_opener: MediaOpener = av.open,
    path_validator: PathValidator = validate_audio_path,
) -> AudioMetadata:
    """Read container and first-stream metadata without decoding audio frames."""
    path_validator(audio_path)

    with media_opener(str(audio_path)) as container:
        audio_stream = next(iter(container.streams.audio), None)
        if audio_stream is None:
            raise ValueError("Media file does not contain an audio stream")

        codec_context = audio_stream.codec_context
        duration_seconds = _duration_seconds(audio_stream, container)

        return AudioMetadata(
            file_extension=audio_path.suffix.lower(),
            duration_seconds=duration_seconds,
            container_format=_attribute_name(container.format),
            codec_name=_optional_string(codec_context.name),
            sample_rate_hz=_optional_int(codec_context.sample_rate),
            channel_count=_optional_int(codec_context.channels),
            channel_layout=_attribute_name(codec_context.layout),
            sample_format=_attribute_name(codec_context.format),
            bit_rate=_first_available_int(
                getattr(audio_stream, "bit_rate", None),
                getattr(codec_context, "bit_rate", None),
                getattr(container, "bit_rate", None),
            ),
        )


def _duration_seconds(audio_stream: Any, container: Any) -> float | None:
    stream_duration = getattr(audio_stream, "duration", None)
    time_base = getattr(audio_stream, "time_base", None)
    if stream_duration is not None and time_base is not None:
        return float(stream_duration * time_base)

    container_duration = getattr(container, "duration", None)
    if container_duration is not None:
        return float(container_duration / av.time_base)
    return None


def _attribute_name(value: Any) -> str | None:
    if value is None:
        return None
    return _optional_string(getattr(value, "name", None))


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _first_available_int(*values: Any) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None
