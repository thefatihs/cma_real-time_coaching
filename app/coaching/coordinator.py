"""Coordinate deterministic coaching evaluation with per-call display state."""

from dataclasses import dataclass
from enum import Enum
import logging

from app.calls.models import (
    CallCoachingMetadata,
    CallDetectedLabelMetadata,
    CallRevisionLabelDiagnostic,
    CallState,
)
from app.coaching.rule_engine import RuleBasedCoachingEngine
from app.events.labels import canonical_label
from app.events.models import (
    ClassificationResultEvent,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.events.validation import ensure_same_call, ensure_same_tenant
from app.tenancy.models import TenantConfig


@dataclass(frozen=True, slots=True)
class CoachingCoordinatorResult:
    classification_event: ClassificationResultEvent | None
    displayed_suggestions: tuple[CoachingSuggestionEvent, ...]
    suppressed_suggestions: tuple[CoachingSuggestionEvent, ...]
    matched_rule_ids: tuple[str, ...]
    suppression_reasons: tuple[str, ...]
    transcript_revision: int | None = None
    current_revision_labels: tuple[str, ...] = ()
    replaced_suggestion_ids: tuple[str, ...] = ()
    suggestion_decisions: tuple["SafeSuggestionDecision", ...] = ()


@dataclass(frozen=True, slots=True)
class SafeSuggestionDecision:
    transcript_revision: int
    label_id: str | None
    priority: SuggestionPriority
    reason: str
    moved_to_history: bool


class CoachingProcessingStatus(str, Enum):
    PROCESSED = "processed"
    PARTIAL_SKIPPED = "partial_skipped"
    DUPLICATE_REVISION_SKIPPED = "duplicate_revision_skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StableCoachingOutcome:
    status: CoachingProcessingStatus
    transcript_revision: int
    result: CoachingCoordinatorResult | None = None
    error_type: str | None = None
    error_code: str | None = None


SuggestionFingerprint = tuple[str, str, str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class CoachingStateSnapshot:
    active_labels: tuple[str, ...]
    detected_labels: tuple[CallDetectedLabelMetadata, ...]
    label_revision_timeline: tuple[CallRevisionLabelDiagnostic, ...]
    shown_suggestion_ids: tuple[str, ...]
    last_coaching_trigger_seconds: float | None
    coaching_suggestions: tuple[CallCoachingMetadata, ...]
    active_coaching_suggestions: tuple[CallCoachingMetadata, ...]
    coaching_suggestion_history: tuple[CallCoachingMetadata, ...]
    coaching_transcript_revision: int | None
    displayed_fingerprints: frozenset[SuggestionFingerprint]
    displayed_candidate_times: tuple[tuple[str, float], ...]
    processed_revisions: frozenset[int]


class CoachingCoordinator:
    def __init__(
        self,
        tenant_config: TenantConfig,
        call_state: CallState,
        rule_engine: RuleBasedCoachingEngine,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        tenant_id = ensure_same_tenant(
            tenant_config.context,
            call_state,
            rule_engine,
        )
        if tenant_id != tenant_config.context.tenant_id:
            raise ValueError("Coaching coordinator tenant validation failed")
        self._tenant_config = tenant_config
        self._call_state = call_state
        self._rule_engine = rule_engine
        self._displayed_fingerprints: set[SuggestionFingerprint] = set()
        self._displayed_candidate_times: dict[str, float] = {}
        self._processed_revisions: set[int] = set()
        self._logger = logger or logging.getLogger(__name__)

    @property
    def call_state(self) -> CallState:
        return self._call_state

    def snapshot_coaching_state(self) -> CoachingStateSnapshot:
        return CoachingStateSnapshot(
            active_labels=tuple(self._call_state.active_labels),
            detected_labels=tuple(self._call_state.detected_labels),
            label_revision_timeline=tuple(self._call_state.label_revision_timeline),
            shown_suggestion_ids=tuple(self._call_state.shown_suggestion_ids),
            last_coaching_trigger_seconds=(
                self._call_state.last_coaching_trigger_seconds
            ),
            coaching_suggestions=tuple(self._call_state.coaching_suggestions),
            active_coaching_suggestions=tuple(
                self._call_state.active_coaching_suggestions
            ),
            coaching_suggestion_history=tuple(
                self._call_state.coaching_suggestion_history
            ),
            coaching_transcript_revision=self._call_state.coaching_transcript_revision,
            displayed_fingerprints=frozenset(self._displayed_fingerprints),
            displayed_candidate_times=tuple(
                sorted(self._displayed_candidate_times.items())
            ),
            processed_revisions=frozenset(self._processed_revisions),
        )

    def restore_coaching_state(self, snapshot: CoachingStateSnapshot) -> None:
        self._call_state.active_labels = list(snapshot.active_labels)
        self._call_state.detected_labels = list(snapshot.detected_labels)
        self._call_state.label_revision_timeline = list(
            snapshot.label_revision_timeline
        )
        self._call_state.shown_suggestion_ids = list(snapshot.shown_suggestion_ids)
        self._call_state.last_coaching_trigger_seconds = (
            snapshot.last_coaching_trigger_seconds
        )
        self._call_state.coaching_suggestions = list(snapshot.coaching_suggestions)
        self._call_state.active_coaching_suggestions = list(
            snapshot.active_coaching_suggestions
        )
        self._call_state.coaching_suggestion_history = list(
            snapshot.coaching_suggestion_history
        )
        self._call_state.coaching_transcript_revision = (
            snapshot.coaching_transcript_revision
        )
        self._displayed_fingerprints = set(snapshot.displayed_fingerprints)
        self._displayed_candidate_times = dict(snapshot.displayed_candidate_times)
        self._processed_revisions = set(snapshot.processed_revisions)

    def process(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> CoachingCoordinatorResult:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        ensure_same_tenant(self._tenant_config.context, self._call_state, event)
        ensure_same_call(self._call_state, event)
        if event.kind is TranscriptKind.PARTIAL:
            return CoachingCoordinatorResult(None, (), (), (), (), event.revision)
        if event.revision in self._processed_revisions:
            return CoachingCoordinatorResult(
                classification_event,
                (),
                (),
                (),
                ("duplicate_revision",),
                event.revision,
            )
        self._processed_revisions.add(event.revision)
        if classification_event is not None:
            ensure_same_tenant(event, classification_event)
            ensure_same_call(event, classification_event)

        labels_for_evaluation = active_labels or ()
        evaluation = self._rule_engine.evaluate(event, labels_for_evaluation)
        classification = classification_event or evaluation.classification_event
        evaluated_labels = (
            [label.name for label in evaluation.classification_event.labels]
            if evaluation.classification_event is not None
            else []
        )
        labels = list(dict.fromkeys((*labels_for_evaluation, *evaluated_labels)))
        if any(label != "no_action" for label in labels):
            labels = [label for label in labels if label != "no_action"]
        self._call_state.update_active_labels(labels)

        displayed: list[CoachingSuggestionEvent] = []
        suppressed: list[CoachingSuggestionEvent] = []
        reasons: list[str] = []
        replaced_ids: list[str] = []
        decisions: list[SafeSuggestionDecision] = []
        maximum = self._tenant_config.coaching.max_active_suggestions

        for suggestion in _merge_current_candidates(evaluation.suggestion_events):
            if suggestion.label_id is not None:
                self._call_state.record_detected_labels(
                    [suggestion.label_id],
                    transcript_revision=event.revision,
                    source=suggestion.source,
                    model_id=(
                        classification_event.model_id
                        if classification_event is not None
                        else None
                    ),
                    threshold_profile_id=(
                        classification_event.threshold_profile_id
                        if classification_event is not None
                        else None
                    ),
                )
            fingerprint = _suggestion_fingerprint(suggestion)
            candidate_key = _candidate_key(suggestion)
            if fingerprint in self._displayed_fingerprints:
                suppressed.append(suggestion)
                reasons.append("duplicate_same_revision")
                decisions.append(
                    _decision(
                        suggestion,
                        event.revision,
                        "duplicate_same_revision",
                        False,
                    )
                )
            elif not _cooldown_available(
                self._displayed_candidate_times.get(candidate_key),
                current_seconds,
                self._tenant_config.coaching.cooldown_seconds,
            ):
                suppressed.append(suggestion)
                reasons.append("cooldown_previously_displayed")
                decisions.append(
                    _decision(
                        suggestion,
                        event.revision,
                        "cooldown_previously_displayed",
                        False,
                    )
                )
            else:
                admitted, replaced = self._call_state.admit_coaching_suggestion(
                    suggestion,
                    transcript_revision=event.revision,
                    model_id=(
                        classification_event.model_id
                        if classification_event is not None
                        else None
                    ),
                    threshold_profile_id=(
                        classification_event.threshold_profile_id
                        if classification_event is not None
                        else None
                    ),
                    maximum_active_suggestions=maximum,
                )
                if not admitted:
                    suppressed.append(suggestion)
                    reasons.append("rejected_by_capacity")
                    decisions.append(
                        _decision(
                            suggestion,
                            event.revision,
                            "rejected_by_capacity",
                            False,
                        )
                    )
                    continue
                displayed.append(suggestion)
                self._displayed_fingerprints.add(fingerprint)
                self._displayed_candidate_times[candidate_key] = current_seconds
                self._call_state.mark_suggestion_shown(suggestion.suggestion_id)
                decisions.append(
                    _decision(suggestion, event.revision, "admitted", False)
                )
                if replaced is not None:
                    replaced_ids.append(replaced.suggestion_id)
                    decisions.append(
                        SafeSuggestionDecision(
                            transcript_revision=replaced.transcript_revision,
                            label_id=replaced.label_id,
                            priority=replaced.priority,
                            reason="replaced_by_newer_priority",
                            moved_to_history=True,
                        )
                    )

        if displayed:
            self._call_state.mark_coaching_triggered(current_seconds)

        return CoachingCoordinatorResult(
            classification_event=classification,
            displayed_suggestions=tuple(displayed),
            suppressed_suggestions=tuple(suppressed),
            matched_rule_ids=evaluation.matched_rule_ids,
            suppression_reasons=tuple(reasons),
            transcript_revision=event.revision,
            current_revision_labels=tuple(labels),
            replaced_suggestion_ids=tuple(replaced_ids),
            suggestion_decisions=tuple(decisions),
        )

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        if event.kind is TranscriptKind.PARTIAL:
            return StableCoachingOutcome(
                CoachingProcessingStatus.PARTIAL_SKIPPED, event.revision
            )
        if event.revision in self._processed_revisions:
            return StableCoachingOutcome(
                CoachingProcessingStatus.DUPLICATE_REVISION_SKIPPED, event.revision
            )
        try:
            result = self.process(
                event,
                current_seconds,
                classification_event=classification_event,
                active_labels=active_labels,
            )
            return StableCoachingOutcome(
                CoachingProcessingStatus.PROCESSED,
                event.revision,
                result=result,
            )
        except Exception as error:
            self._processed_revisions.add(event.revision)
            self._logger.error(
                "stable transcript coaching failed",
                extra={
                    "tenant_id": event.tenant_id,
                    "call_id": event.call_id,
                    "transcript_revision": event.revision,
                    "error_type": type(error).__name__,
                    "error_code": "coaching_failed",
                },
            )
            return StableCoachingOutcome(
                CoachingProcessingStatus.FAILED,
                event.revision,
                error_type=type(error).__name__,
                error_code="coaching_failed",
            )

    def clear(self) -> None:
        self._call_state.coaching_suggestion_history = [
            *self._call_state.coaching_suggestion_history,
            *self._call_state.active_coaching_suggestions,
        ]
        self._call_state.active_coaching_suggestions = []
        self._displayed_fingerprints.clear()
        self._displayed_candidate_times.clear()
        self._processed_revisions.clear()


def _suggestion_fingerprint(
    suggestion: CoachingSuggestionEvent,
) -> SuggestionFingerprint:
    return (
        suggestion.action.value,
        suggestion.title,
        suggestion.suggestion,
        tuple(suggestion.evidence_ids),
    )


def _candidate_key(suggestion: CoachingSuggestionEvent) -> str:
    label = canonical_label(suggestion.label_id) if suggestion.label_id else None
    return label or suggestion.label_id or repr(_suggestion_fingerprint(suggestion))


def _cooldown_available(
    displayed_seconds: float | None,
    current_seconds: float,
    cooldown_seconds: float,
) -> bool:
    return (
        displayed_seconds is None
        or current_seconds - displayed_seconds >= cooldown_seconds
    )


def _merge_current_candidates(
    suggestions: tuple[CoachingSuggestionEvent, ...],
) -> tuple[CoachingSuggestionEvent, ...]:
    by_key: dict[str, CoachingSuggestionEvent] = {}
    for suggestion in suggestions:
        key = _candidate_key(suggestion)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = suggestion
            continue
        sources = {existing.source, suggestion.source}
        source = (
            CoachingSuggestionSource.BOTH
            if len(sources) > 1 or CoachingSuggestionSource.BOTH in sources
            else existing.source
        )
        by_key[key] = existing.model_copy(update={"source": source})
    return tuple(by_key[key] for key in sorted(by_key, key=str.casefold))


def _decision(
    suggestion: CoachingSuggestionEvent,
    transcript_revision: int,
    reason: str,
    moved_to_history: bool,
) -> SafeSuggestionDecision:
    return SafeSuggestionDecision(
        transcript_revision=transcript_revision,
        label_id=suggestion.label_id,
        priority=suggestion.priority,
        reason=reason,
        moved_to_history=moved_to_history,
    )
