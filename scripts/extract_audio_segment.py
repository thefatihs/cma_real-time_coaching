import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import av


MediaOpener = Callable[..., Any]
ResamplerFactory = Callable[..., Any]


def validate_segment(start_seconds: float, end_seconds: float) -> None:
    if start_seconds < 0:
        raise ValueError("Start time must be non-negative")
    if end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time")


def validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError("Input audio file does not exist")
    if input_path.is_dir():
        raise IsADirectoryError("Input audio path is a directory")
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output path must use the .wav extension")


def extract_audio_segment(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    *,
    media_opener: MediaOpener = av.open,
    resampler_factory: ResamplerFactory = av.AudioResampler,
    path_validator: Callable[[Path, Path], None] = validate_paths,
) -> None:
    """Extract a short PCM s16 WAV segment while preserving rate and layout."""
    validate_segment(start_seconds, end_seconds)
    path_validator(input_path, output_path)

    with media_opener(str(input_path)) as input_container:
        input_stream = next(iter(input_container.streams.audio), None)
        if input_stream is None:
            raise ValueError("Input media does not contain an audio stream")

        codec_context = input_stream.codec_context
        sample_rate = codec_context.sample_rate
        layout = codec_context.layout.name
        if sample_rate is None or not layout:
            raise ValueError("Input audio sample rate or channel layout is unavailable")

        resampler = resampler_factory(format="s16", layout=layout, rate=sample_rate)
        input_container.seek(int(start_seconds * av.time_base), backward=True)

        with media_opener(str(output_path), mode="w", format="wav") as output_container:
            output_stream = output_container.add_stream("pcm_s16le", rate=sample_rate)
            output_stream.codec_context.layout = layout

            for frame in input_container.decode(input_stream):
                frame_start = _frame_start_seconds(frame)
                frame_end = frame_start + (frame.samples / frame.sample_rate)
                if frame_end <= start_seconds:
                    continue
                if frame_start >= end_seconds:
                    break

                for resampled_frame in resampler.resample(frame):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)

            for resampled_frame in resampler.resample(None):
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)
            for packet in output_stream.encode(None):
                output_container.mux(packet)


def _frame_start_seconds(frame: Any) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    raise ValueError("Audio frame timestamp is unavailable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a short PCM s16 WAV segment.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    extractor: Callable[[Path, Path, float, float], None] = extract_audio_segment,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        extractor(args.input_path, args.output, args.start, args.end)
    except (FileNotFoundError, IsADirectoryError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error: Audio segment extraction failed.", file=sys.stderr)
        return 1

    print("Audio segment created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
