"""Pure deterministic alignment of immutable ASR words to speaker turns."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.asr.models import ASRWordTimestamp
from app.diarization.models import DiarizationTurn, DiarizedWord, SpeakerRole


UNKNOWN_LOCAL_SPEAKER_ID = "UNKNOWN"


class WordAlignmentErrorCategory(str, Enum):
    INVALID_SCOPE = "invalid_alignment_scope"
    INVALID_REVISION = "invalid_alignment_revision"
    INVALID_PARENT_RANGE = "invalid_alignment_parent_range"
    WORD_OUTSIDE_PARENT = "alignment_word_outside_parent"
    TURN_OUTSIDE_PARENT = "alignment_turn_outside_parent"
    SCOPE_MISMATCH = "alignment_scope_mismatch"
    CONFLICTING_DUPLICATE = "alignment_conflicting_duplicate"


class WordAlignmentError(ValueError):
    def __init__(self, category: WordAlignmentErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(category={self.category.value!r})"


@dataclass(frozen=True, slots=True)
class WordAlignmentRequest:
    tenant_id: str
    call_id: str
    transcript_revision: int
    parent_start_seconds: float
    parent_end_seconds: float
    words: tuple[ASRWordTimestamp, ...]
    turns: tuple[DiarizationTurn, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.call_id.strip():
            raise WordAlignmentError(WordAlignmentErrorCategory.INVALID_SCOPE)
        if self.transcript_revision < 0:
            raise WordAlignmentError(WordAlignmentErrorCategory.INVALID_REVISION)
        if (
            not isfinite(self.parent_start_seconds)
            or not isfinite(self.parent_end_seconds)
            or self.parent_start_seconds < 0
            or self.parent_end_seconds <= self.parent_start_seconds
        ):
            raise WordAlignmentError(WordAlignmentErrorCategory.INVALID_PARENT_RANGE)


def align_words_to_speakers(
    request: WordAlignmentRequest,
) -> tuple[DiarizedWord, ...]:
    """Return aligned words without mutating trusted request collections."""
    words = _unique_words(request)
    turns = _unique_turns(request)
    aligned = tuple(_align_word(request, word, turns) for word in words)
    return tuple(
        sorted(
            aligned,
            key=lambda word: (
                word.start_seconds,
                word.end_seconds,
                word.text,
                word.local_speaker_ids,
            ),
        )
    )


def _unique_words(
    request: WordAlignmentRequest,
) -> tuple[ASRWordTimestamp, ...]:
    unique: dict[tuple[float, float], ASRWordTimestamp] = {}
    for word in request.words:
        if (
            word.start_seconds < request.parent_start_seconds
            or word.end_seconds > request.parent_end_seconds
        ):
            raise WordAlignmentError(WordAlignmentErrorCategory.WORD_OUTSIDE_PARENT)
        key = (word.start_seconds, word.end_seconds)
        existing = unique.get(key)
        if existing is not None and existing != word:
            raise WordAlignmentError(WordAlignmentErrorCategory.CONFLICTING_DUPLICATE)
        unique[key] = word
    return tuple(
        sorted(
            unique.values(),
            key=lambda word: (word.start_seconds, word.end_seconds, word.text),
        )
    )


def _unique_turns(
    request: WordAlignmentRequest,
) -> tuple[DiarizationTurn, ...]:
    unique: dict[tuple[float, float, tuple[str, ...]], DiarizationTurn] = {}
    for turn in request.turns:
        if turn.tenant_id != request.tenant_id or turn.call_id != request.call_id:
            raise WordAlignmentError(WordAlignmentErrorCategory.SCOPE_MISMATCH)
        if (
            turn.start_seconds < request.parent_start_seconds
            or turn.end_seconds > request.parent_end_seconds
        ):
            raise WordAlignmentError(WordAlignmentErrorCategory.TURN_OUTSIDE_PARENT)
        key = (turn.start_seconds, turn.end_seconds, turn.local_speaker_ids)
        existing = unique.get(key)
        if existing is not None and existing != turn:
            raise WordAlignmentError(WordAlignmentErrorCategory.CONFLICTING_DUPLICATE)
        unique[key] = turn
    return tuple(
        sorted(
            unique.values(),
            key=lambda turn: (
                turn.start_seconds,
                turn.end_seconds,
                tuple(sorted(turn.local_speaker_ids)),
            ),
        )
    )


def _align_word(
    request: WordAlignmentRequest,
    word: ASRWordTimestamp,
    turns: tuple[DiarizationTurn, ...],
) -> DiarizedWord:
    candidates = [
        (
            min(word.end_seconds, turn.end_seconds)
            - max(word.start_seconds, turn.start_seconds),
            turn,
        )
        for turn in turns
    ]
    positive = [(overlap, turn) for overlap, turn in candidates if overlap > 0]
    if positive:
        _, winner = min(
            positive,
            key=lambda item: (
                -item[0],
                item[1].start_seconds,
                tuple(sorted(item[1].local_speaker_ids)),
            ),
        )
        local_speaker_ids = winner.local_speaker_ids
        global_speaker_id = winner.global_speaker_id
        global_speaker_ids = winner.global_speaker_ids
        role = winner.role
        speaker_confidence = winner.speaker_confidence
        role_confidence = winner.role_confidence
    else:
        local_speaker_ids = (UNKNOWN_LOCAL_SPEAKER_ID,)
        global_speaker_id = None
        global_speaker_ids = ()
        role = SpeakerRole.UNKNOWN
        speaker_confidence = None
        role_confidence = None
    return DiarizedWord(
        tenant_id=request.tenant_id,
        call_id=request.call_id,
        transcript_revision=request.transcript_revision,
        start_seconds=word.start_seconds,
        end_seconds=word.end_seconds,
        text=word.text,
        local_speaker_ids=local_speaker_ids,
        global_speaker_id=global_speaker_id,
        global_speaker_ids=global_speaker_ids,
        role=role,
        speaker_confidence=speaker_confidence,
        role_confidence=role_confidence,
    )
