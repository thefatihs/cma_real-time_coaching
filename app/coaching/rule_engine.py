"""Deterministic tenant-aware transcript coaching rules."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import unicodedata
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

from app.events.models import (
    ClassificationLabel,
    ClassificationResultEvent,
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.events.validation import ensure_same_tenant
from app.tenancy.models import TenantConfig


class CoachingRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    label: str
    include_any: tuple[str, ...] = ()
    include_all: tuple[str, ...] = ()
    exclude_any: tuple[str, ...] = ()
    action: CoachingAction
    priority: SuggestionPriority
    title: str
    suggestion: str
    evidence_ids: tuple[str, ...] = ()
    enabled: bool = True

    @field_validator("rule_id", "label", "title", "suggestion")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} cannot be empty")
        return cleaned

    @field_validator("include_any", "include_all", "exclude_any")
    @classmethod
    def validate_phrases(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _validated_unique(values, getattr(info, "field_name", "phrases"), True)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_unique(values, "evidence_ids", False)

    @model_validator(mode="after")
    def validate_conditions(self) -> Self:
        if not self.include_any and not self.include_all:
            raise ValueError("at least one include condition is required")
        normalized = [
            _tokens(phrase)
            for phrase in (*self.include_any, *self.include_all, *self.exclude_any)
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("rule phrases cannot be duplicated")
        return self


@dataclass(frozen=True, slots=True)
class RuleEvaluationResult:
    classification_event: ClassificationResultEvent | None
    suggestion_events: tuple[CoachingSuggestionEvent, ...]
    matched_rule_ids: tuple[str, ...]


class RuleBasedCoachingEngine:
    _ACTION_STRENGTH = {
        CoachingAction.NO_ACTION: 0,
        CoachingAction.TEMPLATE_ACTION: 1,
        CoachingAction.RAG_ACTION: 2,
        CoachingAction.ESCALATE: 3,
    }
    _PRIORITY_STRENGTH = {
        SuggestionPriority.LOW: 0,
        SuggestionPriority.MEDIUM: 1,
        SuggestionPriority.HIGH: 2,
        SuggestionPriority.CRITICAL: 3,
    }

    def __init__(
        self,
        tenant_config: TenantConfig,
        rules: tuple[CoachingRule, ...],
        event_id_factory: Callable[[], str] | None = None,
        utc_datetime_factory: Callable[[], datetime] | None = None,
    ) -> None:
        rule_ids = [rule.rule_id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule IDs must be unique")
        configured_labels = set(tenant_config.classification.labels)
        allowed_actions = set(tenant_config.coaching.allowed_actions)
        for rule in rules:
            if rule.label not in configured_labels:
                raise ValueError(f"Unknown tenant classification label: {rule.label}")
            if rule.action.value not in allowed_actions:
                raise ValueError(f"Rule action is not allowed: {rule.action.value}")

        self._tenant_config = tenant_config
        self._rules = rules
        self._event_id_factory = event_id_factory or (lambda: str(uuid4()))
        self._utc_datetime_factory = utc_datetime_factory or (lambda: datetime.now(UTC))

    @property
    def tenant_id(self) -> str:
        return self._tenant_config.context.tenant_id

    def evaluate(
        self,
        event: TranscriptEvent,
        classification_labels: tuple[str, ...] = (),
    ) -> RuleEvaluationResult:
        ensure_same_tenant(self._tenant_config.context, event)
        if event.kind is TranscriptKind.PARTIAL:
            return RuleEvaluationResult(None, (), ())

        transcript_tokens = _tokens(event.text)
        rule_matches = tuple(
            rule
            for rule in self._rules
            if rule.enabled and _matches(rule, transcript_tokens)
        )
        classification_label_set = set(classification_labels)
        classification_matches = tuple(
            rule
            for rule in self._rules
            if rule.enabled and rule.label in classification_label_set
        )
        matched_by_id = {
            rule.rule_id: rule for rule in (*rule_matches, *classification_matches)
        }
        matched = tuple(matched_by_id.values())
        if not matched:
            return RuleEvaluationResult(None, (), ())

        labels = sorted({rule.label for rule in matched}, key=str.casefold)
        strongest_action = max(
            (rule.action for rule in matched), key=self._ACTION_STRENGTH.__getitem__
        )
        classification = ClassificationResultEvent(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            transcript_event_id=event.event_id,
            labels=[ClassificationLabel(name=label, score=1.0) for label in labels],
            action=strongest_action,
            model_id=self._tenant_config.classification.model_id,
            created_at_utc=self._utc_datetime_factory(),
        )

        suggestion_rules: dict[str, CoachingRule] = {}
        for rule in matched:
            if rule.action not in {
                CoachingAction.TEMPLATE_ACTION,
                CoachingAction.ESCALATE,
            }:
                continue
            selected = suggestion_rules.get(rule.label)
            if selected is None or self._suggestion_strength(rule) > (
                self._suggestion_strength(selected)
            ):
                suggestion_rules[rule.label] = rule

        suggestions = tuple(
            self._suggestion_event(
                event,
                rule,
                source=_source_for_rule(rule, rule_matches, classification_matches),
            )
            for rule in suggestion_rules.values()
        )
        return RuleEvaluationResult(
            classification_event=classification,
            suggestion_events=suggestions,
            matched_rule_ids=tuple(rule.rule_id for rule in matched),
        )

    def _suggestion_strength(self, rule: CoachingRule) -> tuple[int, int]:
        return (
            self._ACTION_STRENGTH[rule.action],
            self._PRIORITY_STRENGTH[rule.priority],
        )

    def _suggestion_event(
        self,
        event: TranscriptEvent,
        rule: CoachingRule,
        *,
        source: CoachingSuggestionSource,
    ) -> CoachingSuggestionEvent:
        return CoachingSuggestionEvent(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            suggestion_id=self._event_id_factory(),
            source_transcript_event_id=event.event_id,
            action=rule.action,
            priority=rule.priority,
            source=source,
            title=rule.title,
            suggestion=rule.suggestion,
            evidence_ids=list(rule.evidence_ids),
            created_at_utc=self._utc_datetime_factory(),
        )


def _source_for_rule(
    rule: CoachingRule,
    rule_matches: tuple[CoachingRule, ...],
    classification_matches: tuple[CoachingRule, ...],
) -> CoachingSuggestionSource:
    matched_rule = rule in rule_matches
    matched_classification = rule in classification_matches
    if matched_rule and matched_classification:
        return CoachingSuggestionSource.BOTH
    if matched_classification:
        return CoachingSuggestionSource.CLASSIFICATION
    return CoachingSuggestionSource.RULE


def _validated_unique(
    values: tuple[str, ...], field_name: str, normalize_phrase: bool
) -> tuple[str, ...]:
    cleaned = tuple(value.strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{field_name} cannot contain empty values")
    keys: list[object] = (
        [_tokens(value) for value in cleaned] if normalize_phrase else list(cleaned)
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return cleaned


def _matches(rule: CoachingRule, transcript: tuple[str, ...]) -> bool:
    include_any = [_tokens(phrase) for phrase in rule.include_any]
    include_all = [_tokens(phrase) for phrase in rule.include_all]
    exclude_any = [_tokens(phrase) for phrase in rule.exclude_any]
    if include_any and not any(_contains(transcript, phrase) for phrase in include_any):
        return False
    if include_all and not all(_contains(transcript, phrase) for phrase in include_all):
        return False
    return not any(_contains(transcript, phrase) for phrase in exclude_any)


def _contains(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(
        words[index : index + size] == phrase for index in range(len(words) - size + 1)
    )


def _tokens(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for raw_word in text.split():
        start = 0
        end = len(raw_word)
        while start < end and _is_surrounding_punctuation(raw_word[start]):
            start += 1
        while end > start and _is_surrounding_punctuation(raw_word[end - 1]):
            end -= 1
        word = raw_word[start:end].casefold().replace("\N{COMBINING DOT ABOVE}", "")
        if word:
            result.append(word)
    return tuple(result)


def _is_surrounding_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith(("P", "S"))
