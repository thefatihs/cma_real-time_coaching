"""Command-line entry point for safe local audio stream simulation."""

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.streaming.simulator import (  # noqa: E402
    DEFAULT_WINDOW_SECONDS,
    StreamStep,
    simulate_audio_stream,
)


Simulator = Callable[..., Iterator[StreamStep]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate a safe local audio stream.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--chunk-duration", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--realtime", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    simulator: Simulator = simulate_audio_stream,
) -> int:
    args = build_parser().parse_args(argv)
    total_chunks = 0
    audio_duration = 0.0
    try:
        for step in simulator(
            args.audio_path,
            args.tenant_id,
            args.call_id,
            args.chunk_duration,
            args.window_seconds,
            realtime=args.realtime,
        ):
            print(json.dumps(asdict(step), separators=(",", ":")))
            total_chunks += 1
            audio_duration = step.chunk_end_seconds
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    summary = {
        "type": "summary",
        "total_chunk_count": total_chunks,
        "audio_duration_seconds": audio_duration,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
