"""Strict provider codecs for a future PostgreSQL vector adapter."""

import json
import math
import struct
from collections.abc import Sequence
from numbers import Real

from app.vector_store.models import Metadata
from app.vector_store.postgres.contracts import PostgreSQLCosineSearchRow


def canonicalize_float32_embedding(
    embedding: object,
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    """Return an exact IEEE-754 float32 representation of an embedding."""
    if type(expected_dimension) is not int or expected_dimension <= 0:
        raise ValueError("expected_dimension must be a positive integer")
    if isinstance(embedding, (str, bytes, bytearray)) or not isinstance(
        embedding, Sequence
    ):
        raise ValueError("embedding must be an ordered non-string sequence")
    if not embedding:
        raise ValueError("embedding cannot be empty")
    if len(embedding) != expected_dimension:
        raise ValueError("embedding dimension does not match expected_dimension")

    canonical: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("embedding must contain only real numeric values")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("embedding must contain only finite values")
        try:
            packed = struct.pack("!f", numeric_value)
        except (OverflowError, struct.error) as error:
            raise ValueError("embedding value exceeds float32 range") from error
        canonical_value = struct.unpack("!f", packed)[0]
        if not math.isfinite(canonical_value):
            raise ValueError("embedding value exceeds float32 range")
        canonical.append(canonical_value)
    return tuple(canonical)


def encode_ordered_metadata(metadata: Metadata) -> str:
    """Encode canonical ordered metadata as a compact JSON array of pairs."""
    pairs = _validate_metadata_pairs(metadata)
    return json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))


def decode_ordered_metadata(metadata_json: object) -> Metadata:
    """Decode strict canonical ordered metadata without repairing stored data."""
    if not isinstance(metadata_json, str):
        raise ValueError("metadata_json must be a string")
    try:
        decoded = json.loads(metadata_json)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("metadata_json must contain valid JSON") from error
    if not isinstance(decoded, list):
        raise ValueError("metadata_json must contain an array of pairs")
    return _validate_metadata_pairs(decoded)


def cosine_minimum_score_to_maximum_distance(minimum_score: float) -> float:
    """Convert bounded cosine relevance into the matching distance threshold."""
    score = _validated_unit_interval(minimum_score, "minimum_score")
    return 2.0 * (1.0 - score)


def cosine_distance_to_relevance(cosine_distance: object) -> float:
    """Convert a valid cosine distance into bounded relevance."""
    distance = _validated_unit_interval(
        cosine_distance,
        "cosine_distance",
        upper_bound=2.0,
    )
    return 1.0 - distance / 2.0


def order_cosine_search_rows(
    rows: tuple[PostgreSQLCosineSearchRow, ...],
) -> tuple[PostgreSQLCosineSearchRow, ...]:
    """Validate every distance, then order by relevance and stable identity."""
    relevance_by_position = tuple(
        cosine_distance_to_relevance(row.cosine_distance) for row in rows
    )
    indexed_rows = zip(rows, relevance_by_position, strict=True)
    return tuple(
        row
        for row, _ in sorted(
            indexed_rows,
            key=lambda item: (
                -item[1],
                item[0].document_id,
                item[0].chunk_id,
            ),
        )
    )


def _validate_metadata_pairs(metadata: object) -> Metadata:
    if isinstance(metadata, (str, bytes, bytearray)) or not isinstance(
        metadata, Sequence
    ):
        raise ValueError("metadata must be an ordered sequence of pairs")
    result: list[tuple[str, str]] = []
    keys: set[str] = set()
    for pair in metadata:
        if (
            isinstance(pair, (str, bytes, bytearray))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError("metadata items must be two-element pairs")
        key, value = pair
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("metadata keys and values must be strings")
        if not key or key != key.strip() or not value or value != value.strip():
            raise ValueError("metadata keys and values must be canonical and nonblank")
        if key in keys:
            raise ValueError("metadata keys must be unique")
        keys.add(key)
        result.append((key, value))
    return tuple(result)


def _validated_unit_interval(
    value: object,
    field_name: str,
    *,
    upper_bound: float = 1.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real numeric value")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= upper_bound:
        raise ValueError(
            f"{field_name} must be finite and between 0 and {upper_bound:g}"
        )
    return numeric_value
