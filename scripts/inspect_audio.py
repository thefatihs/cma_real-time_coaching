import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio.metadata import AudioMetadata, inspect_audio_metadata  # noqa: E402


MetadataInspector = Callable[[Path], AudioMetadata]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect technical audio metadata without decoding or transcribing."
    )
    parser.add_argument("audio_path", type=Path)
    return parser


def _display(value: object | None, suffix: str = "") -> str:
    return f"{value}{suffix}" if value is not None else "Unavailable"


def format_metadata(metadata: AudioMetadata) -> str:
    return "\n".join(
        [
            f"File extension: {_display(metadata.file_extension or None)}",
            f"Container: {_display(metadata.container_format)}",
            f"Codec: {_display(metadata.codec_name)}",
            f"Duration: {_display(metadata.duration_seconds, ' seconds')}",
            f"Sample rate: {_display(metadata.sample_rate_hz, ' Hz')}",
            f"Channels: {_display(metadata.channel_count)}",
            f"Channel layout: {_display(metadata.channel_layout)}",
            f"Sample format: {_display(metadata.sample_format)}",
            f"Bit rate: {_display(metadata.bit_rate, ' bit/s')}",
        ]
    )


def main(
    argv: Sequence[str] | None = None,
    inspector: MetadataInspector = inspect_audio_metadata,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        metadata = inspector(args.audio_path)
    except FileNotFoundError:
        print("Error: Audio file does not exist.", file=sys.stderr)
        return 1
    except IsADirectoryError:
        print("Error: Audio path is a directory, not a file.", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "Error: Media is unsupported, corrupted, or could not be inspected.",
            file=sys.stderr,
        )
        return 1

    print(format_metadata(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
