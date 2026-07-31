"""Deterministic construction of complete LLM coaching suggestion events."""

from collections.abc import Callable
from datetime import datetime, timedelta

from app.coaching.llm_result_gate import (
    LLMCoachingGateStatus,
    LLMCoachingResultGate,
)
from app.events.models import (
    CoachingAction,
    CoachingSuggestionEvent,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
)
from app.orchestration import OrchestrationResult


class DeterministicLLMCoachingSuggestionFactory:
    def __init__(
        self,
        *,
        title: str,
        action: CoachingAction,
        priority: SuggestionPriority,
        label_id: str | None,
        expires_after_seconds: float | None,
        suggestion_id_factory: Callable[[], str],
        utc_datetime_factory: Callable[[], datetime],
    ) -> None:
        self._title = _required_text(title, "title")
        self._action = action
        self._priority = priority
        self._label_id = (
            None if label_id is None else _required_text(label_id, "label_id")
        )
        if expires_after_seconds is not None and expires_after_seconds < 0:
            raise ValueError("expires_after_seconds cannot be negative")
        if not callable(suggestion_id_factory):
            raise ValueError("suggestion_id_factory must be callable")
        if not callable(utc_datetime_factory):
            raise ValueError("utc_datetime_factory must be callable")
        self._expires_after_seconds = expires_after_seconds
        self._suggestion_id_factory = suggestion_id_factory
        self._utc_datetime_factory = utc_datetime_factory
        self._result_gate = LLMCoachingResultGate()

    def create(
        self,
        *,
        event: TranscriptEvent,
        orchestration_result: OrchestrationResult,
        current_seconds: float,
    ) -> CoachingSuggestionEvent | None:
        if current_seconds < 0:
            raise ValueError("current_seconds cannot be negative")
        if (
            orchestration_result.tenant_id != event.tenant_id
            or orchestration_result.call_id != event.call_id
            or orchestration_result.transcript_revision != event.revision
            or not orchestration_result.generated_text.strip()
        ):
            return None

        gate_result = self._result_gate.evaluate(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            revision=event.revision,
            raw_output=orchestration_result.generated_text,
            allowed_citations={
                (citation.document_id, citation.chunk_id)
                for citation in orchestration_result.citations
            },
        )
        if (
            gate_result.status is not LLMCoachingGateStatus.VALID_SUGGESTION
            or gate_result.suggestion is None
        ):
            return None

        suggestion_id = self._suggestion_id_factory()
        if not isinstance(suggestion_id, str):
            raise ValueError("suggestion_id_factory must return a string")
        validated_id = _required_text(suggestion_id, "suggestion_id")
        created_at_utc = self._utc_datetime_factory()
        if not isinstance(created_at_utc, datetime):
            raise ValueError("utc_datetime_factory must return a datetime")
        if (
            created_at_utc.tzinfo is None
            or created_at_utc.utcoffset() is None
            or created_at_utc.utcoffset() != timedelta(0)
        ):
            raise ValueError(
                "utc_datetime_factory must return a timezone-aware UTC datetime"
            )

        return CoachingSuggestionEvent(
            tenant_id=event.tenant_id,
            call_id=event.call_id,
            suggestion_id=validated_id,
            source_transcript_event_id=event.event_id,
            action=self._action,
            priority=self._priority,
            source=CoachingSuggestionSource.LLM,
            label_id=self._label_id,
            title=self._title,
            suggestion=gate_result.suggestion.suggestion,
            evidence_ids=[],
            expires_after_seconds=self._expires_after_seconds,
            created_at_utc=created_at_utc,
        )


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
