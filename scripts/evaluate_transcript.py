import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.metrics import evaluate_transcript  # noqa: E402
from app.evaluation.models import TranscriptEvaluationResult  # noqa: E402


TextReader = Callable[[Path, str], str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a human reference transcript with an ASR hypothesis."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path)
    return parser


def read_utf8_text(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if path.is_dir():
        raise ValueError(f"{label} path is a directory, not a file: {path}")
    return path.read_text(encoding="utf-8")


def format_result(result: TranscriptEvaluationResult) -> str:
    return "\n".join(
        [
            f"Normalized reference: {result.normalized_reference}",
            f"Normalized hypothesis: {result.normalized_hypothesis}",
            f"WER: {result.wer:.2%}",
            f"CER: {result.cer:.2%}",
            f"Substitutions: {result.substitutions}",
            f"Deletions: {result.deletions}",
            f"Insertions: {result.insertions}",
            f"Correct words: {result.correct_words}",
        ]
    )


def main(
    argv: Sequence[str] | None = None,
    text_reader: TextReader = read_utf8_text,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        reference_text = text_reader(args.reference, "Reference")
        hypothesis_text = text_reader(args.hypothesis, "Hypothesis")
        result = evaluate_transcript(reference_text, hypothesis_text)
    except (FileNotFoundError, ValueError, UnicodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Could not read transcript file: {error}", file=sys.stderr)
        return 1

    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
