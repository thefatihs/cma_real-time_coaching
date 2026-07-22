"""Deterministically reconcile overlapping ASR window transcripts."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
import unicodedata
from uuid import uuid4

from app.events.models import TranscriptEvent, TranscriptKind
from app.streaming.window_transcriber import (
    WindowTranscriptionResult,
    WindowTranscriptionSegment,
)


class TranscriptReconciler:
    def __init__(
        self,
        stable_region_seconds: float = 5.0,
        timestamp_tolerance_seconds: float = 0.25,
        event_id_factory: Callable[[], str] | None = None,
        utc_datetime_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if stable_region_seconds < 0:
            raise ValueError("stable_region_seconds cannot be negative")
        if timestamp_tolerance_seconds < 0:
            raise ValueError("timestamp_tolerance_seconds cannot be negative")
        self.stable_region_seconds = stable_region_seconds
        self.timestamp_tolerance_seconds = timestamp_tolerance_seconds
        self._event_id_factory = event_id_factory or (lambda: str(uuid4()))
        self._utc_datetime_factory = utc_datetime_factory or (lambda: datetime.now(UTC))
        self.clear()

    @property
    def stable_transcript(self) -> str:
        return self._stable_transcript

    @property
    def partial_transcript(self) -> str:
        return self._partial_transcript

    @property
    def revision(self) -> int:
        return self._revision

    def ingest(self, result: WindowTranscriptionResult) -> tuple[TranscriptEvent, ...]:
        self._validate_scope_and_order(result)
        cutoff = result.window_end_seconds - self.stable_region_seconds
        ordered = sorted(
            result.segments,
            key=lambda segment: (
                segment.absolute_start_seconds,
                segment.absolute_end_seconds,
            ),
        )
        stable_segments = [
            segment
            for segment in ordered
            if segment.absolute_end_seconds <= cutoff + self.timestamp_tolerance_seconds
        ]
        partial_segments = [
            segment for segment in ordered if segment not in stable_segments
        ]

        events: list[TranscriptEvent] = []
        stable_candidate = _join_segment_text(stable_segments)
        new_stable_text = _new_suffix(self._stable_transcript, stable_candidate)
        if new_stable_text:
            self._stable_transcript = _join_text(
                self._stable_transcript, new_stable_text
            )
            events.append(
                self._make_event(
                    TranscriptKind.STABLE,
                    new_stable_text,
                    stable_segments,
                    result.last_sequence,
                )
            )

        partial_candidate = _join_segment_text(partial_segments)
        if partial_candidate != self._partial_transcript:
            self._partial_transcript = partial_candidate
            if partial_candidate:
                self._partial_start_seconds, self._partial_end_seconds = _segment_times(
                    partial_segments
                )
                self._partial_source_sequence = result.last_sequence
                events.append(
                    self._make_event(
                        TranscriptKind.PARTIAL,
                        partial_candidate,
                        partial_segments,
                        result.last_sequence,
                    )
                )
            else:
                self._clear_partial()

        self._last_sequence = result.last_sequence
        self._last_window_end_seconds = result.window_end_seconds
        return tuple(events)

    def finalize(self) -> TranscriptEvent | None:
        if not self._partial_transcript:
            return None
        text = _new_suffix(self._stable_transcript, self._partial_transcript)
        self._clear_partial()
        if not text:
            return None
        self._stable_transcript = _join_text(self._stable_transcript, text)
        return self._make_event_from_times(
            TranscriptKind.FINAL,
            text,
            self._finalize_start_seconds,
            self._finalize_end_seconds,
            self._finalize_source_sequence,
        )

    def clear(self) -> None:
        self._tenant_id: str | None = None
        self._call_id: str | None = None
        self._stable_transcript = ""
        self._partial_transcript = ""
        self._partial_start_seconds: float | None = None
        self._partial_end_seconds: float | None = None
        self._partial_source_sequence: int | None = None
        self._finalize_start_seconds = 0.0
        self._finalize_end_seconds = 0.0
        self._finalize_source_sequence: int | None = None
        self._last_sequence: int | None = None
        self._last_window_end_seconds: float | None = None
        self._revision = 0

    def _validate_scope_and_order(self, result: WindowTranscriptionResult) -> None:
        if self._tenant_id is None:
            self._tenant_id = result.tenant_id
            self._call_id = result.call_id
        elif result.tenant_id != self._tenant_id:
            raise ValueError("Mismatched tenant_id")
        elif result.call_id != self._call_id:
            raise ValueError("Mismatched call_id")
        if (
            self._last_sequence is not None
            and result.last_sequence < self._last_sequence
        ):
            raise ValueError("last_sequence cannot decrease")
        if (
            self._last_window_end_seconds is not None
            and result.window_end_seconds < self._last_window_end_seconds
        ):
            raise ValueError("window_end_seconds cannot decrease")

    def _make_event(
        self,
        kind: TranscriptKind,
        text: str,
        segments: Sequence[WindowTranscriptionSegment],
        source_sequence: int,
    ) -> TranscriptEvent:
        start, end = _segment_times(segments)
        return self._make_event_from_times(kind, text, start, end, source_sequence)

    def _make_event_from_times(
        self,
        kind: TranscriptKind,
        text: str,
        start: float,
        end: float,
        source_sequence: int | None,
    ) -> TranscriptEvent:
        if self._tenant_id is None or self._call_id is None:
            raise RuntimeError("Reconciler is not bound to a call")
        self._revision += 1
        return TranscriptEvent(
            tenant_id=self._tenant_id,
            call_id=self._call_id,
            event_id=self._event_id_factory(),
            kind=kind,
            text=text,
            start_seconds=max(0.0, start),
            end_seconds=max(max(0.0, start), end),
            revision=self._revision,
            created_at_utc=self._utc_datetime_factory(),
            source_chunk_sequence=source_sequence,
        )

    def _clear_partial(self) -> None:
        self._finalize_start_seconds = self._partial_start_seconds or 0.0
        self._finalize_end_seconds = self._partial_end_seconds or 0.0
        self._finalize_source_sequence = self._partial_source_sequence
        self._partial_transcript = ""
        self._partial_start_seconds = None
        self._partial_end_seconds = None
        self._partial_source_sequence = None


def _join_segment_text(segments: Sequence[WindowTranscriptionSegment]) -> str:
    return " ".join(
        cleaned for segment in segments if (cleaned := " ".join(segment.text.split()))
    )


def _join_text(first: str, second: str) -> str:
    return " ".join(part for part in (first, second) if part)


def _new_suffix(existing: str, candidate: str) -> str:
    existing_words = existing.split()
    candidate_words = candidate.split()
    normalized_existing = [_normalize_word(word) for word in existing_words]
    normalized_candidate = [_normalize_word(word) for word in candidate_words]
    overlap = 0
    for size in range(min(len(existing_words), len(candidate_words)), 0, -1):
        if normalized_existing[-size:] == normalized_candidate[:size]:
            overlap = size
            break
    return " ".join(candidate_words[overlap:])


def _normalize_word(word: str) -> str:
    return "".join(
        character
        for character in word.casefold()
        if not unicodedata.category(character).startswith(("P", "S"))
    )


def _segment_times(
    segments: Sequence[WindowTranscriptionSegment],
) -> tuple[float, float]:
    if not segments:
        raise ValueError("Transcript events require at least one segment")
    return (
        min(segment.absolute_start_seconds for segment in segments),
        max(segment.absolute_end_seconds for segment in segments),
    )
