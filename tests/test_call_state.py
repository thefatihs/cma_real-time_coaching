from datetime import UTC, datetime

import pytest

from app.calls.models import CallState
from app.events.models import AudioChunkEvent, TranscriptEvent, TranscriptKind


NOW = datetime(2026, 7, 22, tzinfo=UTC)


def audio_event(sequence: int, **changes: object) -> AudioChunkEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "sequence_number": sequence,
        "received_at_utc": NOW,
        "chunk_start_seconds": sequence * 2.0,
        "chunk_duration_seconds": 2.0,
        "sample_rate_hz": 8_000,
        "channel_count": 1,
        "codec_name": "pcm_s16le",
        "audio_bytes": b"synthetic",
    }
    values.update(changes)
    return AudioChunkEvent.model_validate(values)


def transcript_event(
    kind: TranscriptKind, text: str, revision: int, **changes: object
) -> TranscriptEvent:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "event_id": f"transcript_{revision}",
        "kind": kind,
        "text": text,
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "revision": revision,
        "created_at_utc": NOW,
    }
    values.update(changes)
    return TranscriptEvent.model_validate(values)


def test_audio_sequence_progression_and_rejection() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    state.apply_audio_chunk(audio_event(0))
    state.apply_audio_chunk(audio_event(2))
    assert state.last_audio_sequence == 2

    with pytest.raises(ValueError, match="greater"):
        state.apply_audio_chunk(audio_event(2))
    with pytest.raises(ValueError, match="greater"):
        state.apply_audio_chunk(audio_event(1))


def test_partial_stable_and_final_transcript_behavior() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    state.apply_transcript(transcript_event(TranscriptKind.PARTIAL, "Merhaba", 1))
    state.apply_transcript(transcript_event(TranscriptKind.PARTIAL, "Merhaba size", 2))
    assert state.partial_transcript == "Merhaba size"

    state.apply_transcript(transcript_event(TranscriptKind.STABLE, "Merhaba size", 3))
    state.apply_transcript(transcript_event(TranscriptKind.STABLE, "Merhaba size", 4))
    assert state.stable_transcript == "Merhaba size"

    state.apply_transcript(transcript_event(TranscriptKind.FINAL, "nasılsınız", 5))
    assert state.stable_transcript == "Merhaba size nasılsınız"
    assert state.partial_transcript == ""


def test_old_revision_is_rejected() -> None:
    state = CallState(
        tenant_id="tenant_alpha", call_id="call_001", transcript_revision=3
    )
    with pytest.raises(ValueError, match="older"):
        state.apply_transcript(transcript_event(TranscriptKind.PARTIAL, "metin", 2))


@pytest.mark.parametrize(
    "event",
    [
        audio_event(0, tenant_id="tenant_beta"),
        audio_event(0, call_id="call_002"),
    ],
)
def test_tenant_and_call_mismatches_are_rejected(event: AudioChunkEvent) -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    with pytest.raises(ValueError, match="Mismatched"):
        state.apply_audio_chunk(event)


def test_active_labels_and_suggestions_are_normalized() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    state.update_active_labels(["satış", "", " şikayet ", "satış"])
    state.mark_suggestion_shown("suggestion_001")
    state.mark_suggestion_shown("suggestion_001")

    assert state.active_labels == ["satış", "şikayet"]
    assert state.shown_suggestion_ids == ["suggestion_001"]
    with pytest.raises(ValueError, match="cannot be empty"):
        state.mark_suggestion_shown(" ")


def test_coaching_cooldown_and_trigger_marking() -> None:
    state = CallState(tenant_id="tenant_alpha", call_id="call_001")
    assert state.can_trigger_coaching(10, 20)

    state.mark_coaching_triggered(10)
    assert not state.can_trigger_coaching(29, 20)
    assert state.can_trigger_coaching(30, 20)
    assert state.last_coaching_trigger_seconds == 10

    with pytest.raises(ValueError, match="negative"):
        state.mark_coaching_triggered(-1)
    with pytest.raises(ValueError, match="negative"):
        state.can_trigger_coaching(-1, 20)
