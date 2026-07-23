"""Safe JSONL loading and validation for synthetic classification examples."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
import unicodedata

from pydantic import ValidationError

from app.classification.models import ClassificationExample, ClassificationTaxonomy


@dataclass(frozen=True, slots=True)
class ClassificationDataset:
    examples: tuple[ClassificationExample, ...]
    label_counts: Mapping[str, int]
    split_counts: Mapping[str, int]

    @property
    def total_examples(self) -> int:
        return len(self.examples)


def load_classification_taxonomy(path: str | Path) -> ClassificationTaxonomy:
    taxonomy_path = Path(path)
    try:
        payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed taxonomy JSON at line {error.lineno}") from error
    except OSError as error:
        raise ValueError("Unable to read classification taxonomy") from error
    try:
        return ClassificationTaxonomy.model_validate(payload)
    except ValidationError as error:
        raise ValueError("Invalid classification taxonomy") from error


def normalize_example_text(text: str) -> str:
    """Normalize casing, whitespace, and punctuation only at the text boundary."""
    normalized = unicodedata.normalize("NFC", text).casefold().strip()
    normalized = " ".join(normalized.split())
    start = 0
    end = len(normalized)
    while start < end and _is_surrounding_punctuation(normalized[start]):
        start += 1
    while end > start and _is_surrounding_punctuation(normalized[end - 1]):
        end -= 1
    return normalized[start:end].strip()


def load_classification_dataset(
    path: str | Path, taxonomy: ClassificationTaxonomy
) -> ClassificationDataset:
    dataset_path = Path(path)
    examples: list[ClassificationExample] = []
    seen_ids: set[str] = set()
    normalized_locations: dict[str, tuple[str, int]] = {}
    label_ids = set(taxonomy.label_ids)

    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("Unable to read classification dataset") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Empty JSONL record at line {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSON at line {line_number}") from error
        try:
            example = ClassificationExample.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"Invalid example at line {line_number}") from error

        unknown_labels = set(example.labels) - label_ids
        if unknown_labels:
            raise ValueError(
                f"Unknown label in example {example.example_id} at line {line_number}"
            )
        if example.example_id in seen_ids:
            raise ValueError(
                f"Duplicate example_id {example.example_id} at line {line_number}"
            )

        normalized_text = normalize_example_text(example.text)
        if not normalized_text:
            raise ValueError(f"Empty normalized text at line {line_number}")
        previous = normalized_locations.get(normalized_text)
        if previous is not None:
            previous_split, previous_line = previous
            if previous_split != example.split.value:
                raise ValueError(
                    "Normalized text appears in multiple splits "
                    f"at lines {previous_line} and {line_number}"
                )
            raise ValueError(
                f"Duplicate normalized text at lines {previous_line} and {line_number}"
            )

        seen_ids.add(example.example_id)
        normalized_locations[normalized_text] = (example.split.value, line_number)
        examples.append(example)

    label_counts = Counter(label for example in examples for label in example.labels)
    split_counts = Counter(example.split.value for example in examples)
    return ClassificationDataset(
        examples=tuple(examples),
        label_counts=MappingProxyType(dict(sorted(label_counts.items()))),
        split_counts=MappingProxyType(dict(sorted(split_counts.items()))),
    )


def _is_surrounding_punctuation(character: str) -> bool:
    return unicodedata.category(character)[0] in {"P", "S"}
