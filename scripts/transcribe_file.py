import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asr.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.asr.models import TranscriptionResult  # noqa: E402


class TranscriptionEngine(Protocol):
    def transcribe_file(self, audio_path: Path) -> TranscriptionResult: ...


EngineFactory = Callable[..., TranscriptionEngine]
TranscriptWriter = Callable[[Path, str], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe one local audio file with Faster-Whisper."
    )
    parser.add_argument("audio_path", type=Path, help="Path to the local audio file")
    parser.add_argument("--model", default="tiny", help="Whisper model name")
    parser.add_argument("--language", default="tr", help="Audio language code")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam search size")
    parser.add_argument(
        "--cpu-threads", type=int, default=4, help="Number of CPU threads"
    )
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
    )
    parser.add_argument("--initial-prompt")
    parser.add_argument("--output-text", type=Path)
    return parser


def format_result(
    result: TranscriptionResult,
    model_name: str,
    vad_filter: bool,
    condition_on_previous_text: bool,
    initial_prompt: str | None,
) -> str:
    if result.duration_seconds > 0:
        real_time_factor = result.processing_time_seconds / result.duration_seconds
        real_time_factor_text = f"{real_time_factor:.3f}"
    else:
        real_time_factor_text = "N/A"

    lines = [
        f"Model: {model_name}",
        f"VAD filter: {vad_filter}",
        f"Condition on previous text: {condition_on_previous_text}",
        f"Initial prompt: {initial_prompt or 'None'}",
        "Transcript:",
        result.text,
        f"Language: {result.language or 'unknown'}",
        f"Language probability: {result.language_probability:.2%}",
        f"Audio duration: {result.duration_seconds:.2f} seconds",
        f"Processing time: {result.processing_time_seconds:.2f} seconds",
        f"Real-time factor: {real_time_factor_text}",
        "Segments:",
    ]
    lines.extend(
        f"[{segment.start_seconds:.2f}s -> {segment.end_seconds:.2f}s] {segment.text}"
        for segment in result.segments
    )
    return "\n".join(lines)


def write_transcript(output_path: Path, transcript: str) -> None:
    output_path.write_text(transcript, encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    engine_factory: EngineFactory = FasterWhisperEngine,
    transcript_writer: TranscriptWriter = write_transcript,
) -> int:
    args = build_parser().parse_args(argv)

    try:
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
        result = engine.transcribe_file(args.audio_path)
        if args.output_text is not None:
            transcript_writer(args.output_text, result.text)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Model loading or transcription failed: {error}", file=sys.stderr)
        return 1

    print(
        format_result(
            result,
            args.model,
            args.vad_filter,
            args.condition_on_previous_text,
            args.initial_prompt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
