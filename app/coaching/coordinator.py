"""Coordinate deterministic coaching evaluation with per-call display state."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import logging

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, ValidationError

from app.calls.models import (
    CallCoachingMetadata,
    CallDetectedLabelMetadata,
    CallRevisionLabelDiagnostic,
    CallState,
)
from app.coaching.rule_engine import (
    RULE_ONLY_PARTIAL_MODEL_ID,
    RuleBasedCoachingEngine,
)
from app.events.labels import canonical_label
from app.events.models import (
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionLifecycle,
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
    withdrawn_suggestion_ids: tuple[str, ...] = ()
    suggestion_decisions: tuple["SafeSuggestionDecision", ...] = ()
    lifecycle: CoachingSuggestionLifecycle = CoachingSuggestionLifecycle.CONFIRMED


@dataclass(frozen=True, slots=True)
class SafeSuggestionDecision:
    transcript_revision: int
    label_id: str | None
    priority: SuggestionPriority
    reason: str
    moved_to_history: bool


class ExternalSuggestionAdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    SUPPRESSED = "suppressed"
    REJECTED = "rejected"


class ExternalSuggestionAdmissionReason(str, Enum):
    ADMITTED = "admitted"
    DUPLICATE_PREVIOUSLY_DISPLAYED = "duplicate_previously_displayed"
    COOLDOWN_PREVIOUSLY_DISPLAYED = "cooldown_previously_displayed"
    REJECTED_BY_CAPACITY = "rejected_by_capacity"
    INVALID_CANDIDATE = "invalid_candidate"
    UNKNOWN_LABEL = "unknown_label"
    INVALID_SCOPE = "invalid_scope"
    INVALID_REVISION = "invalid_revision"


class ExternalCoachingCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: StrictStr
    call_id: StrictStr
    transcript_revision: StrictInt
    label_id: StrictStr
    action: CoachingAction
    title: StrictStr
    suggestion: StrictStr
    priority: SuggestionPriority
    source: CoachingSuggestionSource = CoachingSuggestionSource.LLM


@dataclass(frozen=True, slots=True)
class ExternalSuggestionAdmissionResult:
    status: ExternalSuggestionAdmissionStatus
    reason: ExternalSuggestionAdmissionReason


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
    processed_external_revisions: frozenset[int]
    provisional_suggestions: tuple[tuple[str, CoachingSuggestionEvent], ...]
    processed_partial_keys: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SuggestionAdmission:
    displayed: tuple[CoachingSuggestionEvent, ...]
    suppressed: tuple[CoachingSuggestionEvent, ...]
    suppression_reasons: tuple[str, ...]
    replaced_suggestion_ids: tuple[str, ...]
    decisions: tuple[SafeSuggestionDecision, ...]


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
        self._processed_external_revisions: set[int] = set()
        self._provisional_suggestions: dict[str, CoachingSuggestionEvent] = {}
        self._processed_partial_keys: list[int] = []
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
            processed_external_revisions=frozenset(self._processed_external_revisions),
            provisional_suggestions=tuple(
                sorted(self._provisional_suggestions.items())
            ),
            processed_partial_keys=tuple(self._processed_partial_keys),
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
        self._processed_external_revisions = set(snapshot.processed_external_revisions)
        self._provisional_suggestions = dict(snapshot.provisional_suggestions)
        self._processed_partial_keys = list(snapshot.processed_partial_keys)

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
            return self._process_provisional(
                event,
                current_seconds,
                classification_event=classification_event,
                active_labels=active_labels or (),
            )
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

        candidates = _merge_current_candidates(evaluation.suggestion_events)
        promoted: list[CoachingSuggestionEvent] = []
        remaining: list[CoachingSuggestionEvent] = []
        confirmed_labels: set[str] = set()
        for suggestion in candidates:
            label = _candidate_key(suggestion)
            confirmed_labels.add(label)
            provisional = self._provisional_suggestions.pop(label, None)
            if provisional is None:
                remaining.append(suggestion)
                continue
            confirmed = suggestion.model_copy(
                update={
                    "suggestion_id": provisional.suggestion_id,
                    "lifecycle": CoachingSuggestionLifecycle.CONFIRMED,
                }
            )
            self._call_state.update_coaching_suggestion(
                confirmed,
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
            )
            promoted.append(confirmed)
        withdrawn_ids: list[str] = []
        for label, provisional in tuple(self._provisional_suggestions.items()):
            if label in confirmed_labels:
                continue
            if self._call_state.withdraw_coaching_suggestion(provisional.suggestion_id):
                withdrawn_ids.append(provisional.suggestion_id)
            del self._provisional_suggestions[label]

        admission = self._apply_suggestion_policy(
            event,
            tuple(remaining),
            current_seconds,
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

        return CoachingCoordinatorResult(
            classification_event=classification,
            displayed_suggestions=(*promoted, *admission.displayed),
            suppressed_suggestions=admission.suppressed,
            matched_rule_ids=evaluation.matched_rule_ids,
            suppression_reasons=admission.suppression_reasons,
            transcript_revision=event.revision,
            current_revision_labels=tuple(labels),
            replaced_suggestion_ids=admission.replaced_suggestion_ids,
            withdrawn_suggestion_ids=tuple(withdrawn_ids),
            suggestion_decisions=admission.decisions,
        )

    def _process_provisional(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None,
        active_labels: tuple[str, ...],
    ) -> CoachingCoordinatorResult:
        if (
            classification_event is None
            or not classification_event.provisional
            or not active_labels
        ):
            return CoachingCoordinatorResult(
                None,
                (),
                (),
                (),
                (),
                event.revision,
                lifecycle=CoachingSuggestionLifecycle.PROVISIONAL,
            )
        ensure_same_tenant(event, classification_event)
        ensure_same_call(event, classification_event)
        partial_key = (
            event.source_chunk_sequence
            if event.source_chunk_sequence is not None
            else event.revision
        )
        if partial_key in self._processed_partial_keys:
            return CoachingCoordinatorResult(
                classification_event,
                (),
                (),
                (),
                ("duplicate_partial_chunk",),
                event.revision,
                lifecycle=CoachingSuggestionLifecycle.PROVISIONAL,
            )
        self._processed_partial_keys.append(partial_key)
        del self._processed_partial_keys[:-64]
        evaluation = self._rule_engine.evaluate(
            event,
            active_labels,
            classification_labels_are_rules=(
                classification_event.model_id == RULE_ONLY_PARTIAL_MODEL_ID
            ),
        )
        candidates = _merge_current_candidates(evaluation.suggestion_events)
        fresh = tuple(
            suggestion
            for suggestion in candidates
            if _candidate_key(suggestion) not in self._provisional_suggestions
        )
        admission = self._apply_suggestion_policy(
            event,
            fresh,
            current_seconds,
            model_id=classification_event.model_id,
            threshold_profile_id=classification_event.threshold_profile_id,
        )
        for suggestion in admission.displayed:
            self._provisional_suggestions[_candidate_key(suggestion)] = suggestion
        return CoachingCoordinatorResult(
            classification_event=classification_event,
            displayed_suggestions=admission.displayed,
            suppressed_suggestions=admission.suppressed,
            matched_rule_ids=evaluation.matched_rule_ids,
            suppression_reasons=admission.suppression_reasons,
            transcript_revision=event.revision,
            current_revision_labels=active_labels,
            replaced_suggestion_ids=admission.replaced_suggestion_ids,
            suggestion_decisions=admission.decisions,
            lifecycle=CoachingSuggestionLifecycle.PROVISIONAL,
        )

    def process_external_suggestion(
        self,
        event: TranscriptEvent,
        suggestion: CoachingSuggestionEvent,
        current_seconds: float,
    ) -> CoachingCoordinatorResult:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        ensure_same_tenant(self._tenant_config.context, self._call_state, event)
        ensure_same_call(self._call_state, event)
        ensure_same_tenant(event, suggestion)
        ensure_same_call(event, suggestion)
        if event.kind is TranscriptKind.PARTIAL:
            raise ValueError("external suggestion requires a stable transcript")
        if event.revision != self._call_state.transcript_revision:
            raise ValueError("external suggestion transcript revision does not match")
        if suggestion.source_transcript_event_id != event.event_id:
            raise ValueError("external suggestion source transcript does not match")
        if suggestion.source is not CoachingSuggestionSource.LLM:
            raise ValueError("external suggestion source must be llm")
        if suggestion.label_id is not None and (
            canonical_label(suggestion.label_id) != suggestion.label_id
            or suggestion.label_id == "no_action"
        ):
            raise ValueError("external suggestion label must be canonical")
        if suggestion.action.value not in self._tenant_config.coaching.allowed_actions:
            return CoachingCoordinatorResult(
                classification_event=None,
                displayed_suggestions=(),
                suppressed_suggestions=(suggestion,),
                matched_rule_ids=(),
                suppression_reasons=("action_not_allowed",),
                transcript_revision=event.revision,
                suggestion_decisions=(
                    _decision(
                        suggestion,
                        event.revision,
                        "action_not_allowed",
                        False,
                    ),
                ),
            )
        if event.revision in self._processed_external_revisions:
            return CoachingCoordinatorResult(
                classification_event=None,
                displayed_suggestions=(),
                suppressed_suggestions=(suggestion,),
                matched_rule_ids=(),
                suppression_reasons=("duplicate_external_revision",),
                transcript_revision=event.revision,
                suggestion_decisions=(
                    _decision(
                        suggestion,
                        event.revision,
                        "duplicate_external_revision",
                        False,
                    ),
                ),
            )

        self._processed_external_revisions.add(event.revision)
        admission = self._apply_suggestion_policy(
            event,
            (suggestion,),
            current_seconds,
            model_id=None,
            threshold_profile_id=None,
        )
        return CoachingCoordinatorResult(
            classification_event=None,
            displayed_suggestions=admission.displayed,
            suppressed_suggestions=admission.suppressed,
            matched_rule_ids=(),
            suppression_reasons=admission.suppression_reasons,
            transcript_revision=event.revision,
            replaced_suggestion_ids=admission.replaced_suggestion_ids,
            suggestion_decisions=admission.decisions,
        )

    def _apply_suggestion_policy(
        self,
        event: TranscriptEvent,
        suggestions: tuple[CoachingSuggestionEvent, ...],
        current_seconds: float,
        *,
        model_id: str | None,
        threshold_profile_id: str | None,
    ) -> _SuggestionAdmission:
        displayed: list[CoachingSuggestionEvent] = []
        suppressed: list[CoachingSuggestionEvent] = []
        reasons: list[str] = []
        replaced_ids: list[str] = []
        decisions: list[SafeSuggestionDecision] = []

        for suggestion in suggestions:
            if (
                suggestion.label_id is not None
                and suggestion.lifecycle is CoachingSuggestionLifecycle.CONFIRMED
            ):
                self._call_state.record_detected_labels(
                    [suggestion.label_id],
                    transcript_revision=event.revision,
                    source=suggestion.source,
                    model_id=model_id,
                    threshold_profile_id=threshold_profile_id,
                )
            admission_reason, replaced = self._admit_candidate(
                suggestion,
                transcript_revision=event.revision,
                current_seconds=current_seconds,
                model_id=model_id,
                threshold_profile_id=threshold_profile_id,
            )
            if admission_reason == "duplicate_same_revision":
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
            elif admission_reason == "cooldown_previously_displayed":
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
            elif admission_reason == "rejected_by_capacity":
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
            else:
                displayed.append(suggestion)
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

        return _SuggestionAdmission(
            displayed=tuple(displayed),
            suppressed=tuple(suppressed),
            suppression_reasons=tuple(reasons),
            replaced_suggestion_ids=tuple(replaced_ids),
            decisions=tuple(decisions),
        )

    def admit_external_suggestion(
        self,
        candidate: object,
        *,
        current_seconds: float,
    ) -> ExternalSuggestionAdmissionResult:
        try:
            validated = ExternalCoachingCandidate.model_validate(candidate)
        except ValidationError:
            return ExternalSuggestionAdmissionResult(
                ExternalSuggestionAdmissionStatus.REJECTED,
                ExternalSuggestionAdmissionReason.INVALID_CANDIDATE,
            )
        if (
            not validated.tenant_id.strip()
            or not validated.call_id.strip()
            or not validated.label_id.strip()
            or not validated.title.strip()
            or not validated.suggestion.strip()
            or validated.transcript_revision < 0
            or current_seconds < 0
            or validated.source is not CoachingSuggestionSource.LLM
            or validated.action.value
            not in self._tenant_config.coaching.allowed_actions
        ):
            return ExternalSuggestionAdmissionResult(
                ExternalSuggestionAdmissionStatus.REJECTED,
                ExternalSuggestionAdmissionReason.INVALID_CANDIDATE,
            )
        if (
            validated.tenant_id != self._call_state.tenant_id
            or validated.call_id != self._call_state.call_id
        ):
            return ExternalSuggestionAdmissionResult(
                ExternalSuggestionAdmissionStatus.REJECTED,
                ExternalSuggestionAdmissionReason.INVALID_SCOPE,
            )
        if validated.transcript_revision != self._call_state.transcript_revision:
            return ExternalSuggestionAdmissionResult(
                ExternalSuggestionAdmissionStatus.REJECTED,
                ExternalSuggestionAdmissionReason.INVALID_REVISION,
            )
        if (
            canonical_label(validated.label_id) != validated.label_id
            or validated.label_id == "no_action"
        ):
            return ExternalSuggestionAdmissionResult(
                ExternalSuggestionAdmissionStatus.REJECTED,
                ExternalSuggestionAdmissionReason.UNKNOWN_LABEL,
            )

        suggestion = CoachingSuggestionEvent(
            tenant_id=validated.tenant_id,
            call_id=validated.call_id,
            suggestion_id=(f"llm:{validated.transcript_revision}:{validated.label_id}"),
            source_transcript_event_id=f"external:{validated.transcript_revision}",
            action=validated.action,
            priority=validated.priority,
            source=validated.source,
            label_id=validated.label_id,
            title=validated.title,
            suggestion=validated.suggestion,
            created_at_utc=datetime.now(UTC),
        )
        reason, _ = self._admit_candidate(
            suggestion,
            transcript_revision=validated.transcript_revision,
            current_seconds=current_seconds,
            model_id=None,
            threshold_profile_id=None,
        )
        if reason == "admitted":
            self._call_state.mark_coaching_triggered(current_seconds)
            return ExternalSuggestionAdmissionResult(
                ExternalSuggestionAdmissionStatus.ADMITTED,
                ExternalSuggestionAdmissionReason.ADMITTED,
            )
        reason_map = {
            "duplicate_same_revision": (
                ExternalSuggestionAdmissionReason.DUPLICATE_PREVIOUSLY_DISPLAYED
            ),
            "cooldown_previously_displayed": (
                ExternalSuggestionAdmissionReason.COOLDOWN_PREVIOUSLY_DISPLAYED
            ),
            "rejected_by_capacity": (
                ExternalSuggestionAdmissionReason.REJECTED_BY_CAPACITY
            ),
        }
        return ExternalSuggestionAdmissionResult(
            ExternalSuggestionAdmissionStatus.SUPPRESSED,
            reason_map[reason],
        )

    def _admit_candidate(
        self,
        suggestion: CoachingSuggestionEvent,
        *,
        transcript_revision: int,
        current_seconds: float,
        model_id: str | None,
        threshold_profile_id: str | None,
    ) -> tuple[str, CallCoachingMetadata | None]:
        fingerprint = _suggestion_fingerprint(suggestion)
        candidate_key = _candidate_key(suggestion)
        if fingerprint in self._displayed_fingerprints:
            return "duplicate_same_revision", None
        if not _cooldown_available(
            self._displayed_candidate_times.get(candidate_key),
            current_seconds,
            self._tenant_config.coaching.cooldown_seconds,
        ):
            return "cooldown_previously_displayed", None
        admitted, replaced = self._call_state.admit_coaching_suggestion(
            suggestion,
            transcript_revision=transcript_revision,
            model_id=model_id,
            threshold_profile_id=threshold_profile_id,
            maximum_active_suggestions=(
                self._tenant_config.coaching.max_active_suggestions
            ),
        )
        if not admitted:
            return "rejected_by_capacity", None
        self._displayed_fingerprints.add(fingerprint)
        self._displayed_candidate_times[candidate_key] = current_seconds
        self._call_state.mark_suggestion_shown(suggestion.suggestion_id)
        return "admitted", replaced

    def process_safely(
        self,
        event: TranscriptEvent,
        current_seconds: float,
        *,
        classification_event: ClassificationResultEvent | None = None,
        active_labels: tuple[str, ...] | None = None,
    ) -> StableCoachingOutcome:
        if event.kind is TranscriptKind.PARTIAL and (
            classification_event is None or not classification_event.provisional
        ):
            return StableCoachingOutcome(
                CoachingProcessingStatus.PARTIAL_SKIPPED, event.revision
            )
        if (
            event.kind is not TranscriptKind.PARTIAL
            and event.revision in self._processed_revisions
        ):
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
            if event.kind is not TranscriptKind.PARTIAL:
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
        self._processed_external_revisions.clear()
        self._provisional_suggestions.clear()
        self._processed_partial_keys.clear()


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
