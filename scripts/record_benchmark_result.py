import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark.models import BenchmarkRun  # noqa: E402
from app.benchmark.repository import BenchmarkRepository  # noqa: E402
from app.evaluation.metrics import evaluate_transcript  # noqa: E402
from scripts.evaluate_transcript import read_utf8_text  # noqa: E402


TextReader = Callable[[Path, str], str]
Clock = Callable[[], datetime]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record privacy-safe ASR benchmark metrics."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--beam-size", default=5, type=int)
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
    )
    parser.add_argument("--initial-prompt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", default=4, type=int)
    parser.add_argument("--processing-time", type=float)
    parser.add_argument("--codec-name")
    parser.add_argument("--sample-rate-hz", type=int)
    parser.add_argument("--channel-count", type=int)
    parser.add_argument("--channel-layout")
    parser.add_argument("--sample-format")
    parser.add_argument("--bit-rate", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def stable_run_id(values: dict[str, object]) -> str:
    canonical = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def main(
    argv: Sequence[str] | None = None,
    text_reader: TextReader = read_utf8_text,
    clock: Clock = lambda: datetime.now(UTC),
) -> int:
    args = build_parser().parse_args(argv)

    try:
        reference = text_reader(args.reference, "Reference")
        hypothesis = text_reader(args.hypothesis, "Hypothesis")
        evaluation = evaluate_transcript(reference, hypothesis)
        duration = args.end - args.start
        if duration <= 0:
            raise ValueError("End time must be greater than start time")
        if args.processing_time is not None and args.processing_time < 0:
            raise ValueError("Processing time must be non-negative")

        identity = {
            "experiment_id": args.experiment_id,
            "recording_id": args.recording_id,
            "segment_id": args.segment_id,
            "start_seconds": args.start,
            "end_seconds": args.end,
            "model_name": args.model,
            "language": args.language,
            "beam_size": args.beam_size,
            "vad_filter": args.vad_filter,
            "condition_on_previous_text": args.condition_on_previous_text,
            "initial_prompt": args.initial_prompt,
            "device": args.device,
            "compute_type": args.compute_type,
            "reference_filename": args.reference.name,
            "hypothesis_filename": args.hypothesis.name,
        }
        run = BenchmarkRun(
            schema_version=1,
            run_id=args.run_id or stable_run_id(identity),
            experiment_id=args.experiment_id,
            recording_id=args.recording_id,
            segment_id=args.segment_id,
            created_at_utc=clock().astimezone(UTC).isoformat(),
            start_seconds=args.start,
            end_seconds=args.end,
            duration_seconds=duration,
            model_name=args.model,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            condition_on_previous_text=args.condition_on_previous_text,
            initial_prompt=args.initial_prompt,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            processing_time_seconds=args.processing_time,
            real_time_factor=(
                args.processing_time / duration
                if args.processing_time is not None
                else None
            ),
            wer=evaluation.wer,
            cer=evaluation.cer,
            substitutions=evaluation.substitutions,
            deletions=evaluation.deletions,
            insertions=evaluation.insertions,
            correct_words=evaluation.correct_words,
            reference_word_count=evaluation.reference_word_count,
            word_error_count=(
                evaluation.substitutions + evaluation.deletions + evaluation.insertions
            ),
            character_error_count=evaluation.character_error_count,
            reference_character_count=evaluation.reference_character_count,
            codec_name=args.codec_name,
            sample_rate_hz=args.sample_rate_hz,
            channel_count=args.channel_count,
            channel_layout=args.channel_layout,
            sample_format=args.sample_format,
            bit_rate=args.bit_rate,
            reference_filename=args.reference.name,
            hypothesis_filename=args.hypothesis.name,
        )
        repository = BenchmarkRepository(args.results_dir)
        repository.save(run, overwrite=args.overwrite)
        repository.rebuild_csv()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved benchmark run {run.run_id}: "
        f"WER={run.wer:.2%}, CER={run.cer:.2%}, words={run.reference_word_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
