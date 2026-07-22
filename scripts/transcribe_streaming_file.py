import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.streaming.pipeline import (  # noqa: E402
    StreamingASRPipeline,
    StreamingASRResult,
)
from app.streaming.window_transcriber import WindowTranscriber  # noqa: E402
from app.tenancy.models import TenantASRConfig, TenantContext  # noqa: E402


class PipelineProtocol(Protocol):
    def run(self, audio_path: Path, call_id: str) -> StreamingASRResult: ...


EngineFactory = Callable[..., object]
PipelineFactory = Callable[
    [TenantContext, TenantASRConfig, WindowTranscriber], PipelineProtocol
]
TranscriptWriter = Callable[[Path, str], None]
Clock = Callable[[], float]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run streaming Faster-Whisper ASR on one local audio file."
    )
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--beam-size", type=_positive_int, default=5)
    parser.add_argument("--cpu-threads", type=_positive_int, default=4)
    parser.add_argument("--chunk-duration", type=_positive_float, default=2.0)
    parser.add_argument("--window-seconds", type=_positive_float, default=20.0)
    parser.add_argument("--stable-region-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
    )
    parser.add_argument("--initial-prompt")
    parser.add_argument("--show-steps", action="store_true")
    parser.add_argument("--output-text", type=Path)
    return parser


def format_configuration(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "Streaming ASR configuration:",
            f"Tenant ID: {args.tenant_id}",
            f"Call ID: {args.call_id}",
            f"Model: {args.model}",
            f"Language: {args.language}",
            "Device: cpu",
            "Compute type: int8",
            f"Beam size: {args.beam_size}",
            f"CPU threads: {args.cpu_threads}",
            f"Chunk duration: {args.chunk_duration:.2f} seconds",
            f"Window duration: {args.window_seconds:.2f} seconds",
            f"Stable region: {args.stable_region_seconds:.2f} seconds",
            f"VAD filter: {args.vad_filter}",
            f"Condition on previous text: {args.condition_on_previous_text}",
            f"Initial prompt configured: {args.initial_prompt is not None}",
        ]
    )


def format_step(step: object) -> str:
    event_kinds = (
        ",".join(event.kind.value for event in getattr(step, "transcript_events"))
        or "none"
    )
    return (
        f"Chunk {getattr(step, 'sequence_number')}: "
        f"window={getattr(step, 'window_start_seconds'):.2f}-"
        f"{getattr(step, 'window_end_seconds'):.2f}s "
        f"processing={getattr(step, 'transcription_time_seconds'):.2f}s "
        f"events={event_kinds} "
        f"stable={getattr(step, 'stable_transcript')!r} "
        f"partial={getattr(step, 'partial_transcript')!r}"
    )


def format_final_output(result: StreamingASRResult, wall_seconds: float) -> str:
    rtf = (
        f"{wall_seconds / result.audio_duration_seconds:.3f}"
        if result.audio_duration_seconds > 0
        else "N/A"
    )
    return "\n".join(
        [
            "Final transcript:",
            result.stable_transcript,
            "Summary:",
            f"Total chunks: {result.total_chunks}",
            f"Audio duration: {result.audio_duration_seconds:.2f} seconds",
            f"Wall-clock processing time: {wall_seconds:.2f} seconds",
            f"Approximate overall RTF: {rtf}",
            f"Final stable transcript length: {len(result.stable_transcript)}",
        ]
    )


def write_transcript(output_path: Path, transcript: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript, encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    engine_factory: EngineFactory = FasterWhisperEngine,
    pipeline_factory: PipelineFactory = StreamingASRPipeline,
    transcript_writer: TranscriptWriter = write_transcript,
    clock: Clock = perf_counter,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        if not args.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
        if not args.audio_path.is_file():
            raise ValueError(
                f"Audio path is a directory, not a file: {args.audio_path}"
            )

        tenant_context = TenantContext(
            tenant_id=args.tenant_id,
            tenant_name=args.tenant_id,
        )
        asr_config = TenantASRConfig(
            model_name=args.model,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            condition_on_previous_text=args.condition_on_previous_text,
            initial_prompt=args.initial_prompt,
            rolling_window_seconds=args.window_seconds,
            chunk_duration_seconds=args.chunk_duration,
            stable_region_seconds=args.stable_region_seconds,
        )
        print(format_configuration(args))

        engine = engine_factory(
            model_size=args.model,
            device="cpu",
            compute_type="int8",
            language=args.language,
            beam_size=args.beam_size,
            cpu_threads=args.cpu_threads,
            vad_filter=args.vad_filter,
            condition_on_previous_text=args.condition_on_previous_text,
            initial_prompt=args.initial_prompt,
        )
        pipeline = pipeline_factory(
            tenant_context,
            asr_config,
            WindowTranscriber(engine),  # type: ignore[arg-type]
        )
        started_at = clock()
        result = pipeline.run(args.audio_path, args.call_id)
        wall_seconds = max(0.0, clock() - started_at)
        if args.output_text is not None:
            transcript_writer(args.output_text, result.stable_transcript)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Streaming transcription failed: {error}", file=sys.stderr)
        return 1

    if args.show_steps:
        for step in result.steps:
            print(format_step(step))
    print(format_final_output(result, wall_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
