from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.events.models import (
    AudioChunkEvent,
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionEvent,
    RetrievalRequestEvent,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def make_audio_event(**changes: object) -> AudioChunkEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "sequence_number": 0,
        "received_at_utc": NOW,
        "chunk_start_seconds": 0.0,
        "chunk_duration_seconds": 2.0,
        "sample_rate_hz": 8_000,
        "channel_count": 1,
        "codec_name": "pcm_s16le",
        "audio_bytes": b"synthetic-audio",
    }
    values.update(changes)
    return AudioChunkEvent.model_validate(values)


def make_transcript_event(**changes: object) -> TranscriptEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "event_id": "transcript_001",
        "kind": TranscriptKind.PARTIAL,
        "text": "Merhaba",
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "revision": 1,
        "created_at_utc": NOW,
    }
    values.update(changes)
    return TranscriptEvent.model_validate(values)


def test_valid_audio_event_and_safe_summary() -> None:
    event = make_audio_event()
    summary = event.metadata_summary()

    assert event.sample_rate_hz == 8_000
    assert "audio_bytes" not in summary
    assert b"synthetic-audio" not in summary.values()


def test_audio_bytes_are_hidden_from_repr() -> None:
    assert "synthetic-audio" not in repr(make_audio_event())
    assert "audio_bytes" not in repr(make_audio_event())


@pytest.mark.parametrize(
    "changes",
    [
        {"audio_bytes": b""},
        {"sequence_number": -1},
        {"chunk_duration_seconds": 0},
    ],
)
def test_invalid_audio_fields_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_audio_event(**changes)


def test_timezone_aware_datetime_is_required() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_audio_event(received_at_utc=datetime(2026, 7, 22, 12, 0))


def test_transcript_time_validation_and_text_stripping() -> None:
    event = make_transcript_event(text="  Merhaba dünya  ")
    assert event.text == "Merhaba dünya"

    with pytest.raises(ValidationError, match="end_seconds"):
        make_transcript_event(start_seconds=3, end_seconds=2)


def test_classification_score_validation() -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        ClassificationLabel(name="şikayet", score=1.1)


def test_classification_labels_are_unique_and_sorted() -> None:
    event = ClassificationResultEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        transcript_event_id="transcript_001",
        labels=[
            ClassificationLabel(name="satış", score=0.6),
            ClassificationLabel(name="şikayet", score=0.9),
        ],
        action=CoachingAction.TEMPLATE_ACTION,
        model_id="setfit-alpha",
        created_at_utc=NOW,
    )
    assert [label.name for label in event.labels] == ["şikayet", "satış"]

    with pytest.raises(ValidationError, match="unique"):
        ClassificationResultEvent(
            tenant_id="tenant_alpha",
            call_id="call_001",
            transcript_event_id="transcript_001",
            labels=[
                ClassificationLabel(name="satış", score=0.5),
                ClassificationLabel(name="satış", score=0.6),
            ],
            action=CoachingAction.NO_ACTION,
            model_id="setfit-alpha",
            created_at_utc=NOW,
        )


def test_invalid_retrieval_request_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalRequestEvent(
            tenant_id="tenant_alpha",
            call_id="call_001",
            transcript_event_id="transcript_001",
            query=" ",
            knowledge_base_id="kb-alpha",
            top_k=0,
            minimum_score=2,
            created_at_utc=NOW,
        )


def test_coaching_evidence_ids_are_normalized() -> None:
    event = CoachingSuggestionEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        suggestion_id="suggestion_001",
        source_transcript_event_id="transcript_001",
        action=CoachingAction.RAG_ACTION,
        priority=SuggestionPriority.HIGH,
        title="İade bilgisi",
        suggestion="İade koşullarını açıklayın.",
        evidence_ids=["doc_1", "doc_2", "doc_1"],
        created_at_utc=NOW,
    )

    assert event.evidence_ids == ["doc_1", "doc_2"]
