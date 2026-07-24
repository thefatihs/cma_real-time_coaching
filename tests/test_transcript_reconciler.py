from datetime import UTC, datetime
from itertools import count

import pytest

from app.events.models import TranscriptKind
from app.streaming.transcript_reconciler import TranscriptReconciler
from app.streaming.window_transcriber import (
    WindowTranscriptionResult,
    WindowTranscriptionSegment,
)


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def segment(text: str, start: float, end: float) -> WindowTranscriptionSegment:
    return WindowTranscriptionSegment(text, start, end, start, end)


def result(
    *segments: WindowTranscriptionSegment, **changes: object
) -> WindowTranscriptionResult:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "first_sequence": 0,
        "last_sequence": 1,
        "window_start_seconds": 0.0,
        "window_end_seconds": 10.0,
        "window_duration_seconds": 10.0,
        "text": "",
        "detected_language": "tr",
        "language_probability": 0.9,
        "processing_time_seconds": 0.1,
        "segments": tuple(segments),
    }
    values.update(changes)
    return WindowTranscriptionResult(**values)  # type: ignore[arg-type]


def reconciler() -> TranscriptReconciler:
    identifiers = count(1)
    return TranscriptReconciler(
        event_id_factory=lambda: f"event_{next(identifiers)}",
        utc_datetime_factory=lambda: NOW,
    )


def test_first_partial_result_and_partial_replacement() -> None:
    subject = reconciler()
    first = subject.ingest(result(segment("hello", 8.0, 9.0)))[0]
    second = subject.ingest(result(segment("hello world", 8.0, 9.5), last_sequence=2))[
        0
    ]
    assert (first.kind, first.text, first.start_seconds, first.end_seconds) == (
        TranscriptKind.PARTIAL,
        "hello",
        8.0,
        9.0,
    )
    assert second.text == "hello world"
    assert subject.partial_transcript == "hello world"


def test_stable_text_emission_and_repeated_overlap_deduplication() -> None:
    subject = reconciler()
    event = subject.ingest(result(segment("Hello, world!", 1.0, 4.0)))[0]
    repeated = subject.ingest(result(segment("hello world", 1.0, 4.0), last_sequence=2))
    assert (event.kind, event.text) == (TranscriptKind.STABLE, "Hello, world!")
    assert repeated == ()
    assert subject.stable_transcript == "Hello, world!"


def test_repeated_old_window_that_is_not_the_current_suffix_is_not_appended() -> None:
    subject = reconciler()
    subject.ingest(
        result(
            segment("Birinci cümle.", 1.0, 2.0),
            segment("İkinci cümle.", 2.0, 4.0),
        )
    )
    repeated_old = subject.ingest(
        result(
            segment("BİRİNCİ CÜMLE", 1.0, 2.0),
            last_sequence=2,
        )
    )
    assert repeated_old == ()
    assert subject.stable_transcript == "Birinci cümle. İkinci cümle."


def test_suffix_prefix_overlap_preserves_new_original_wording() -> None:
    subject = reconciler()
    subject.ingest(result(segment("One TWO.", 1.0, 4.0)))
    events = subject.ingest(result(segment("two, THREE!", 3.0, 5.0), last_sequence=2))
    assert events[0].text == "THREE!"
    assert subject.stable_transcript == "One TWO. THREE!"


def test_small_asr_substitution_and_partial_sentence_overlap_are_deduplicated() -> None:
    subject = reconciler()
    subject.ingest(
        result(segment("Paket fiyatını ve özelliklerini öğrenmek istiyorum.", 1.0, 4.0))
    )
    events = subject.ingest(
        result(
            segment(
                "paketin ücretini ve özelliklerini öğrenmek istiyorum, ayrıca destek.",
                1.0,
                5.0,
            ),
            last_sequence=2,
        )
    )
    assert events[0].text == "ayrıca destek."
    assert subject.stable_transcript.count("özelliklerini") == 1


def test_legitimate_repeated_speech_at_a_later_time_is_preserved() -> None:
    subject = reconciler()
    subject.ingest(result(segment("Bağlantı çalışmıyor.", 1.0, 4.0)))
    events = subject.ingest(
        result(
            segment("Bağlantı çalışmıyor.", 6.0, 7.0),
            last_sequence=2,
            window_end_seconds=12.0,
            window_duration_seconds=12.0,
        )
    )
    assert events[0].text == "Bağlantı çalışmıyor."
    assert subject.stable_transcript == "Bağlantı çalışmıyor. Bağlantı çalışmıyor."


def test_stable_and_partial_are_emitted_from_same_window_in_order() -> None:
    events = reconciler().ingest(
        result(segment("stable", 1.0, 4.0), segment("partial", 7.0, 9.0))
    )
    assert [event.kind for event in events] == [
        TranscriptKind.STABLE,
        TranscriptKind.PARTIAL,
    ]
    assert [event.revision for event in events] == [1, 2]


def test_partial_clearing_emits_no_empty_event() -> None:
    subject = reconciler()
    subject.ingest(result(segment("pending", 8.0, 9.0)))
    events = subject.ingest(result(last_sequence=2))
    assert events == ()
    assert subject.partial_transcript == ""


def test_finalize_pending_partial_and_never_twice() -> None:
    subject = reconciler()
    subject.ingest(result(segment("pending words", 8.0, 9.0)))
    event = subject.finalize()
    assert event is not None
    assert (event.kind, event.text, event.revision) == (
        TranscriptKind.FINAL,
        "pending words",
        2,
    )
    assert (event.tenant_id, event.call_id, event.start_seconds, event.end_seconds) == (
        "tenant_alpha",
        "call_001",
        8.0,
        9.0,
    )
    assert subject.stable_transcript == "pending words"
    assert subject.partial_transcript == ""
    assert subject.finalize() is None


def test_finalize_with_no_partial_returns_none() -> None:
    assert reconciler().finalize() is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tenant_id": "tenant_beta"}, "tenant_id"),
        ({"call_id": "call_002"}, "call_id"),
        ({"last_sequence": 0}, "last_sequence"),
        ({"window_end_seconds": 9.0}, "window_end_seconds"),
    ],
)
def test_scope_and_forward_order_are_enforced(
    changes: dict[str, object], message: str
) -> None:
    subject = reconciler()
    subject.ingest(result(last_sequence=1, window_end_seconds=10.0))
    with pytest.raises(ValueError, match=message):
        subject.ingest(result(**changes))


def test_empty_speech_result_changes_no_transcript() -> None:
    subject = reconciler()
    assert subject.ingest(result()) == ()
    assert (
        subject.stable_transcript,
        subject.partial_transcript,
        subject.revision,
    ) == ("", "", 0)


def test_clear_completely_resets_and_allows_reuse() -> None:
    subject = reconciler()
    subject.ingest(result(segment("first", 8.0, 9.0)))
    subject.clear()
    events = subject.ingest(
        result(
            segment("second", 8.0, 9.0),
            tenant_id="tenant_beta",
            call_id="call_002",
            last_sequence=0,
        )
    )
    assert (
        subject.stable_transcript,
        subject.partial_transcript,
        subject.revision,
    ) == ("", "second", 1)
    assert (events[0].tenant_id, events[0].call_id) == ("tenant_beta", "call_002")


def test_source_results_remain_unchanged() -> None:
    source = result(segment("unchanged", 8.0, 9.0))
    before = repr(source)
    reconciler().ingest(source)
    assert repr(source) == before
