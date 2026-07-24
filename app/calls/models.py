from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.events.models import (
    AudioChunkEvent,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.events.validation import ensure_same_call, ensure_same_tenant


class CallDetectedLabelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    first_detected_revision: int
    latest_detected_revision: int
    source: CoachingSuggestionSource
    model_id: str | None = None
    threshold_profile_id: str | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label cannot be empty")
        return cleaned

    @field_validator("first_detected_revision", "latest_detected_revision")
    @classmethod
    def validate_detected_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("detected revision cannot be negative")
        return value


class CallState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    tenant_id: str
    call_id: str
    stable_transcript: str = ""
    partial_transcript: str = ""
    last_audio_sequence: int = -1
    active_labels: list[str] = Field(default_factory=list)
    detected_labels: list[CallDetectedLabelMetadata] = Field(default_factory=list)
    shown_suggestion_ids: list[str] = Field(default_factory=list)
    last_coaching_trigger_seconds: float | None = None
    transcript_revision: int = 0
    classification_model_id: str | None = None
    classification_threshold_profile_id: str | None = None
    classification_transcript_revision: int | None = None
    classification_source_sequence: int | None = None
    classification_inference_time_ms: float | None = None
    classification_context_sentence_count: int | None = None
    classification_preceding_sentence_count: int | None = None
    classification_delta_word_count: int | None = None
    coaching_suggestions: list["CallCoachingMetadata"] = Field(default_factory=list)
    coaching_transcript_revision: int | None = None

    @field_validator("tenant_id", "call_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                f"{getattr(info, 'field_name', 'identifier')} cannot be empty"
            )
        return cleaned

    @field_validator("last_audio_sequence")
    @classmethod
    def validate_audio_sequence(cls, value: int) -> int:
        if value < -1:
            raise ValueError("last_audio_sequence must be at least -1")
        return value

    @field_validator("transcript_revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 0:
            raise ValueError("transcript_revision cannot be negative")
        return value

    @field_validator(
        "classification_transcript_revision", "classification_source_sequence"
    )
    @classmethod
    def validate_optional_sequence(cls, value: int | None, info: object) -> int | None:
        if value is not None and value < 0:
            raise ValueError(
                f"{getattr(info, 'field_name', 'sequence')} cannot be negative"
            )
        return value

    @field_validator("classification_inference_time_ms")
    @classmethod
    def validate_classification_time(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("classification_inference_time_ms cannot be negative")
        return value

    @field_validator("classification_model_id", "classification_threshold_profile_id")
    @classmethod
    def validate_optional_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("active_labels", "shown_suggestion_ids")
    @classmethod
    def validate_unique_values(cls, values: list[str], info: object) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError(f"{getattr(info, 'field_name', 'values')} must be unique")
        return values

    def apply_audio_chunk(self, event: AudioChunkEvent) -> None:
        self._ensure_same_scope(event)
        if event.sequence_number <= self.last_audio_sequence:
            raise ValueError("Audio sequence must be greater than last_audio_sequence")
        self.last_audio_sequence = event.sequence_number

    def apply_transcript(self, event: TranscriptEvent) -> None:
        self._ensure_same_scope(event)
        if event.revision < self.transcript_revision:
            raise ValueError("Transcript revision is older than current state")

        if event.kind is TranscriptKind.PARTIAL:
            self.partial_transcript = event.text
        elif event.kind is TranscriptKind.STABLE:
            self._append_stable(event.text)
        elif event.kind is TranscriptKind.FINAL:
            self._append_stable(event.text)
            self.partial_transcript = ""
        self.transcript_revision = event.revision

    def update_active_labels(self, labels: list[str]) -> None:
        self.active_labels = _clean_unique(labels)

    def mark_classification_attempt(
        self, transcript_revision: int, source_sequence: int | None
    ) -> None:
        if transcript_revision < 0:
            raise ValueError("transcript_revision cannot be negative")
        if source_sequence is not None and source_sequence < 0:
            raise ValueError("source_sequence cannot be negative")
        self.classification_transcript_revision = transcript_revision
        self.classification_source_sequence = source_sequence

    def apply_classification(
        self,
        event: ClassificationResultEvent,
        *,
        transcript_revision: int,
        source_sequence: int | None,
        context_sentence_count: int | None = None,
        preceding_sentence_count: int | None = None,
        delta_word_count: int | None = None,
    ) -> None:
        self._ensure_same_scope(event)
        self.mark_classification_attempt(transcript_revision, source_sequence)
        self.update_active_labels([label.name for label in event.labels])
        self.record_detected_labels(
            [label.name for label in event.labels],
            transcript_revision=transcript_revision,
            source=CoachingSuggestionSource.CLASSIFICATION,
            model_id=event.model_id,
            threshold_profile_id=event.threshold_profile_id,
        )
        self.classification_model_id = event.model_id
        self.classification_threshold_profile_id = event.threshold_profile_id
        self.classification_inference_time_ms = event.processing_time_ms
        self.classification_context_sentence_count = context_sentence_count
        self.classification_preceding_sentence_count = preceding_sentence_count
        self.classification_delta_word_count = delta_word_count

    def classification_metadata(self) -> "CallClassificationMetadata":
        return CallClassificationMetadata(
            active_labels=tuple(self.active_labels),
            model_id=self.classification_model_id,
            threshold_profile_id=self.classification_threshold_profile_id,
            transcript_revision=self.classification_transcript_revision,
            source_sequence=self.classification_source_sequence,
            inference_time_ms=self.classification_inference_time_ms,
            detected_labels=tuple(self.detected_labels),
            context_sentence_count=self.classification_context_sentence_count,
            preceding_sentence_count=self.classification_preceding_sentence_count,
            delta_word_count=self.classification_delta_word_count,
        )

    def record_detected_labels(
        self,
        labels: list[str],
        *,
        transcript_revision: int,
        source: CoachingSuggestionSource,
        model_id: str | None = None,
        threshold_profile_id: str | None = None,
    ) -> None:
        cleaned = _clean_unique(labels)
        business_labels = [label for label in cleaned if label != "no_action"]
        if business_labels:
            incoming = business_labels
            existing = [
                item for item in self.detected_labels if item.label != "no_action"
            ]
        elif self.detected_labels:
            return
        else:
            incoming = [label for label in cleaned if label == "no_action"]
            existing = list(self.detected_labels)

        by_label = {item.label: item for item in existing}
        for label in incoming:
            previous = by_label.get(label)
            if previous is None:
                by_label[label] = CallDetectedLabelMetadata(
                    label=label,
                    first_detected_revision=transcript_revision,
                    latest_detected_revision=transcript_revision,
                    source=source,
                    model_id=(
                        model_id
                        if source is not CoachingSuggestionSource.RULE
                        else None
                    ),
                    threshold_profile_id=(
                        threshold_profile_id
                        if source is not CoachingSuggestionSource.RULE
                        else None
                    ),
                )
                continue
            combined_source = _combined_source(previous.source, source)
            by_label[label] = previous.model_copy(
                update={
                    "latest_detected_revision": transcript_revision,
                    "source": combined_source,
                    "model_id": (
                        model_id
                        if source is not CoachingSuggestionSource.RULE and model_id
                        else previous.model_id
                    ),
                    "threshold_profile_id": (
                        threshold_profile_id
                        if source is not CoachingSuggestionSource.RULE
                        and threshold_profile_id
                        else previous.threshold_profile_id
                    ),
                }
            )
        self.detected_labels = list(by_label.values())

    def apply_coaching_suggestion(
        self,
        event: CoachingSuggestionEvent,
        *,
        transcript_revision: int,
        model_id: str | None,
        threshold_profile_id: str | None,
    ) -> None:
        self._ensure_same_scope(event)
        metadata = CallCoachingMetadata(
            suggestion_id=event.suggestion_id,
            action=event.action,
            priority=event.priority,
            source=event.source,
            transcript_revision=transcript_revision,
            created_at_utc=event.created_at_utc,
            model_id=model_id if event.source.value != "rule" else None,
            threshold_profile_id=(
                threshold_profile_id if event.source.value != "rule" else None
            ),
        )
        if metadata.suggestion_id not in {
            item.suggestion_id for item in self.coaching_suggestions
        }:
            self.coaching_suggestions = [*self.coaching_suggestions, metadata]
        if event.label_id is not None:
            self.record_detected_labels(
                [event.label_id],
                transcript_revision=transcript_revision,
                source=event.source,
                model_id=model_id,
                threshold_profile_id=threshold_profile_id,
            )
        self.coaching_transcript_revision = transcript_revision

    def mark_suggestion_shown(self, suggestion_id: str) -> None:
        cleaned = suggestion_id.strip()
        if not cleaned:
            raise ValueError("suggestion_id cannot be empty")
        if cleaned not in self.shown_suggestion_ids:
            self.shown_suggestion_ids = [*self.shown_suggestion_ids, cleaned]

    def can_trigger_coaching(
        self, current_seconds: float, cooldown_seconds: float
    ) -> bool:
        if current_seconds < 0 or cooldown_seconds < 0:
            raise ValueError("Coaching times cannot be negative")
        if self.last_coaching_trigger_seconds is None:
            return True
        return current_seconds - self.last_coaching_trigger_seconds >= cooldown_seconds

    def mark_coaching_triggered(self, current_seconds: float) -> None:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        self.last_coaching_trigger_seconds = current_seconds

    def _ensure_same_scope(self, event: object) -> None:
        ensure_same_tenant(self, event)
        ensure_same_call(self, event)

    def _append_stable(self, text: str) -> None:
        cleaned = " ".join(text.split())
        if self.stable_transcript == cleaned or self.stable_transcript.endswith(
            f" {cleaned}"
        ):
            return
        self.stable_transcript = " ".join(
            part for part in (self.stable_transcript.strip(), cleaned) if part
        )


def _clean_unique(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    return cleaned


def _combined_source(
    first: CoachingSuggestionSource,
    second: CoachingSuggestionSource,
) -> CoachingSuggestionSource:
    if first is second:
        return first
    if CoachingSuggestionSource.BOTH in {first, second}:
        return CoachingSuggestionSource.BOTH
    return CoachingSuggestionSource.BOTH


class CallClassificationMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_labels: tuple[str, ...] = ()
    model_id: str | None = None
    threshold_profile_id: str | None = None
    transcript_revision: int | None = None
    source_sequence: int | None = None
    inference_time_ms: float | None = None
    detected_labels: tuple[CallDetectedLabelMetadata, ...] = ()
    context_sentence_count: int | None = None
    preceding_sentence_count: int | None = None
    delta_word_count: int | None = None

    @property
    def current_revision_labels(self) -> tuple[str, ...]:
        return self.active_labels

    @property
    def labels_detected_during_call(self) -> tuple[CallDetectedLabelMetadata, ...]:
        return self.detected_labels


class CallCoachingMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestion_id: str
    action: CoachingAction
    priority: SuggestionPriority
    source: CoachingSuggestionSource
    transcript_revision: int
    created_at_utc: datetime
    model_id: str | None = None
    threshold_profile_id: str | None = None
