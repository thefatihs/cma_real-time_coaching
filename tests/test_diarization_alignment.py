import pytest
from pydantic import ValidationError

from app.asr.models import ASRWordTimestamp
from app.diarization import (
    UNKNOWN_LOCAL_SPEAKER_ID,
    DiarizationTurn,
    SpeakerRole,
    WordAlignmentError,
    WordAlignmentErrorCategory,
    WordAlignmentRequest,
    align_words_to_speakers,
)


def word(
    text: str,
    start: float,
    end: float,
    probability: float | None = None,
) -> ASRWordTimestamp:
    return ASRWordTimestamp(text, start, end, probability)


def turn(
    speaker: str | tuple[str, ...],
    start: float,
    end: float,
    *,
    tenant_id: str = "tenant_alpha",
    call_id: str = "call_001",
    role: SpeakerRole = SpeakerRole.UNKNOWN,
) -> DiarizationTurn:
    speaker_ids = (speaker,) if isinstance(speaker, str) else speaker
    return DiarizationTurn(
        tenant_id=tenant_id,
        call_id=call_id,
        start_seconds=start,
        end_seconds=end,
        local_speaker_ids=speaker_ids,
        role=role,
    )


def request(
    *,
    words: tuple[ASRWordTimestamp, ...] = (),
    turns: tuple[DiarizationTurn, ...] = (),
    tenant_id: str = "tenant_alpha",
    call_id: str = "call_001",
    transcript_revision: int = 3,
    parent_start_seconds: float = 10.0,
    parent_end_seconds: float = 15.0,
) -> WordAlignmentRequest:
    return WordAlignmentRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=transcript_revision,
        parent_start_seconds=parent_start_seconds,
        parent_end_seconds=parent_end_seconds,
        words=words,
        turns=turns,
    )


def test_greatest_positive_overlap_wins_with_absolute_timestamps() -> None:
    source_word = word("merhaba", 11.0, 12.0, 0.95)
    result = align_words_to_speakers(
        request(
            words=(source_word,),
            turns=(
                turn("speaker_a", 10.5, 11.3),
                turn("speaker_b", 11.2, 12.5),
            ),
        )
    )

    assert len(result) == 1
    assert result[0].start_seconds == 11.0
    assert result[0].end_seconds == 12.0
    assert result[0].local_speaker_ids == ("speaker_b",)
    assert result[0].tenant_id == "tenant_alpha"
    assert result[0].call_id == "call_001"
    assert result[0].transcript_revision == 3


def test_ties_use_earliest_turn_then_speaker_id() -> None:
    source_word = word("tie", 11.0, 12.0)
    earliest = align_words_to_speakers(
        request(
            words=(source_word,),
            turns=(
                turn("speaker_a", 11.5, 12.5),
                turn("speaker_b", 10.5, 11.5),
            ),
        )
    )
    speaker_id = align_words_to_speakers(
        request(
            words=(source_word,),
            turns=(
                turn("speaker_b", 10.5, 12.5),
                turn("speaker_a", 10.5, 12.5),
            ),
        )
    )

    assert earliest[0].local_speaker_ids == ("speaker_b",)
    assert speaker_id[0].local_speaker_ids == ("speaker_a",)


def test_no_overlap_uses_unknown_without_global_identity() -> None:
    result = align_words_to_speakers(
        request(
            words=(word("silence", 13.0, 13.5),),
            turns=(turn("speaker_a", 10.0, 11.0),),
        )
    )

    assert result[0].local_speaker_ids == (UNKNOWN_LOCAL_SPEAKER_ID,)
    assert result[0].role is SpeakerRole.UNKNOWN
    assert result[0].global_speaker_id is None
    assert result[0].speaker_confidence is None
    assert result[0].role_confidence is None


def test_overlapping_turn_preserves_all_speakers_and_overlap_role() -> None:
    result = align_words_to_speakers(
        request(
            words=(word("together", 11.0, 12.0),),
            turns=(
                turn(
                    ("speaker_a", "speaker_b"),
                    10.5,
                    12.5,
                    role=SpeakerRole.OVERLAP,
                ),
            ),
        )
    )

    assert result[0].local_speaker_ids == ("speaker_a", "speaker_b")
    assert result[0].role is SpeakerRole.OVERLAP


def test_input_collection_order_does_not_change_output() -> None:
    words = (
        word("second", 12.0, 12.5),
        word("first", 10.5, 11.0),
    )
    turns = (
        turn("speaker_b", 12.0, 13.0),
        turn("speaker_a", 10.0, 11.5),
    )

    first = align_words_to_speakers(request(words=words, turns=turns))
    second = align_words_to_speakers(
        request(words=tuple(reversed(words)), turns=tuple(reversed(turns)))
    )

    assert first == second
    assert [item.text for item in first] == ["first", "second"]


@pytest.mark.parametrize(
    ("changes", "category"),
    [
        (
            {"tenant_id": " "},
            WordAlignmentErrorCategory.INVALID_SCOPE,
        ),
        (
            {"transcript_revision": -1},
            WordAlignmentErrorCategory.INVALID_REVISION,
        ),
        (
            {"parent_start_seconds": float("nan")},
            WordAlignmentErrorCategory.INVALID_PARENT_RANGE,
        ),
        (
            {"parent_end_seconds": 10.0},
            WordAlignmentErrorCategory.INVALID_PARENT_RANGE,
        ),
    ],
)
def test_request_scope_revision_and_parent_range_fail_closed(
    changes: dict[str, object],
    category: WordAlignmentErrorCategory,
) -> None:
    with pytest.raises(WordAlignmentError) as error:
        request(**changes)  # type: ignore[arg-type]

    assert error.value.category is category


def test_rejects_wrong_turn_scope_and_out_of_parent_inputs() -> None:
    with pytest.raises(WordAlignmentError) as scope_error:
        align_words_to_speakers(
            request(turns=(turn("speaker", 10.0, 11.0, call_id="call_002"),))
        )
    with pytest.raises(WordAlignmentError) as word_error:
        align_words_to_speakers(request(words=(word("outside", 9.0, 10.5),)))
    with pytest.raises(WordAlignmentError) as turn_error:
        align_words_to_speakers(request(turns=(turn("speaker", 14.0, 16.0),)))

    assert scope_error.value.category is WordAlignmentErrorCategory.SCOPE_MISMATCH
    assert word_error.value.category is WordAlignmentErrorCategory.WORD_OUTSIDE_PARENT
    assert turn_error.value.category is WordAlignmentErrorCategory.TURN_OUTSIDE_PARENT


def test_conflicting_duplicate_words_and_turns_are_rejected() -> None:
    with pytest.raises(WordAlignmentError) as word_error:
        align_words_to_speakers(
            request(
                words=(
                    word("first", 11.0, 11.5),
                    word("conflict", 11.0, 11.5),
                )
            )
        )
    base_turn = turn("speaker", 10.0, 11.0)
    conflicting_turn = base_turn.model_copy(update={"speaker_confidence": 0.5})
    with pytest.raises(WordAlignmentError) as turn_error:
        align_words_to_speakers(request(turns=(base_turn, conflicting_turn)))

    assert word_error.value.category is WordAlignmentErrorCategory.CONFLICTING_DUPLICATE
    assert turn_error.value.category is WordAlignmentErrorCategory.CONFLICTING_DUPLICATE


def test_alignment_is_immutable_and_does_not_mutate_inputs() -> None:
    words = (word("safe", 11.0, 11.5),)
    turns = (turn("speaker", 10.0, 12.0),)
    source = request(words=words, turns=turns)
    words_before = repr(words)
    turns_before = repr(turns)
    unrelated_runtime = {"transcript_revision": 99, "suggestions": ["keep"]}

    result = align_words_to_speakers(source)

    assert repr(words) == words_before
    assert repr(turns) == turns_before
    assert unrelated_runtime == {
        "transcript_revision": 99,
        "suggestions": ["keep"],
    }
    with pytest.raises(ValidationError):
        result[0].text = "changed"  # type: ignore[misc]


def test_alignment_errors_are_privacy_safe() -> None:
    private_turn = turn(
        "speaker",
        10.0,
        11.0,
        tenant_id="PRIVATE_PATH_TOKEN",
    )

    with pytest.raises(WordAlignmentError) as error:
        align_words_to_speakers(request(turns=(private_turn,)))

    assert str(error.value) == "alignment_scope_mismatch"
    assert "PRIVATE" not in repr(error.value)
