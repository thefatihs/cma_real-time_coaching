import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import cast

import pytest

from app.events.models import (
    CoachingAction,
    CoachingSuggestionSource,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.integration import (
    CoachingSuggestionFactory,
    DeterministicLLMCoachingSuggestionFactory,
    OrchestrationRunner,
    RAGCoachingProcessorDecorator,
)
from app.orchestration import (
    OrchestrationCitationReference,
    OrchestrationResult,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


class Callback:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def transcript() -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="call_001",
        event_id="transcript_7",
        kind=TranscriptKind.STABLE,
        text="Synthetic user input.",
        start_seconds=0,
        end_seconds=7,
        revision=7,
        created_at_utc=NOW,
    )


def orchestration_result(**changes: object) -> OrchestrationResult:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "call_id": "call_001",
        "transcript_revision": 7,
        "generated_text": json.dumps(
            {
                "decision": "suggestion",
                "tenant_id": "tenant_alpha",
                "call_id": "call_001",
                "revision": 7,
                "action": "RAG_ACTION",
                "title": "Synthetic model title",
                "suggestion": "Synthetic generated coaching.",
                "priority": "HIGH",
                "citations": [{"document_id": "document_1", "chunk_id": "chunk_1"}],
                "source": "llm",
            }
        ),
        "citations": (
            OrchestrationCitationReference(
                document_id="document_1",
                chunk_id="chunk_1",
            ),
        ),
    }
    values.update(changes)
    return OrchestrationResult.model_validate(values)


def factory(
    *,
    title: str = "Synthetic policy title",
    action: CoachingAction = CoachingAction.RAG_ACTION,
    priority: SuggestionPriority = SuggestionPriority.HIGH,
    label_id: str | None = "product_information",
    expires_after_seconds: float | None = 30,
    id_callback: Callable[[], str] = lambda: "suggestion_7",
    clock_callback: Callable[[], datetime] = lambda: NOW,
) -> DeterministicLLMCoachingSuggestionFactory:
    return DeterministicLLMCoachingSuggestionFactory(
        title=title,
        action=action,
        priority=priority,
        label_id=label_id,
        expires_after_seconds=expires_after_seconds,
        suggestion_id_factory=id_callback,
        utc_datetime_factory=clock_callback,
    )


def test_structurally_satisfies_existing_factory_protocol() -> None:
    subject: CoachingSuggestionFactory = factory()

    assert isinstance(subject, DeterministicLLMCoachingSuggestionFactory)


def test_exact_trusted_mapping_and_explicit_policy() -> None:
    event = transcript()
    result = orchestration_result()

    suggestion = factory().create(
        event=event,
        orchestration_result=result,
        current_seconds=99,
    )

    assert suggestion is not None
    assert suggestion.tenant_id == event.tenant_id
    assert suggestion.call_id == event.call_id
    assert suggestion.source_transcript_event_id == event.event_id
    assert suggestion.suggestion_id == "suggestion_7"
    assert suggestion.created_at_utc is NOW
    assert suggestion.suggestion == "Synthetic generated coaching."
    assert suggestion.source is CoachingSuggestionSource.LLM
    assert suggestion.title == "Synthetic policy title"
    assert suggestion.action is CoachingAction.RAG_ACTION
    assert suggestion.priority is SuggestionPriority.HIGH
    assert suggestion.label_id == "product_information"
    assert suggestion.expires_after_seconds == 30
    assert suggestion.evidence_ids == []


def test_grounded_creation_retains_only_internal_document_order() -> None:
    grounded = factory().create_grounded(
        event=transcript(),
        orchestration_result=orchestration_result(),
        current_seconds=99,
    )

    assert grounded is not None
    assert grounded.citation_document_ids == ("document_1",)
    assert grounded.event.evidence_ids == []


def test_callbacks_are_called_exactly_once_on_success() -> None:
    id_callback = Callback("synthetic_id")
    clock_callback = Callback(NOW)

    suggestion = factory(
        id_callback=cast(Callable[[], str], id_callback),
        clock_callback=cast(Callable[[], datetime], clock_callback),
    ).create(
        event=transcript(),
        orchestration_result=orchestration_result(),
        current_seconds=7,
    )

    assert suggestion is not None
    assert id_callback.calls == clock_callback.calls == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "tenant_other"},
        {"call_id": "call_other"},
        {"transcript_revision": 8},
    ],
)
def test_scope_or_revision_mismatch_skips_callbacks(
    changes: dict[str, object],
) -> None:
    id_callback = Callback("synthetic_id")
    clock_callback = Callback(NOW)
    subject = factory(
        id_callback=cast(Callable[[], str], id_callback),
        clock_callback=cast(Callable[[], datetime], clock_callback),
    )

    suggestion = subject.create(
        event=transcript(),
        orchestration_result=orchestration_result(**changes),
        current_seconds=7,
    )

    assert suggestion is None
    assert id_callback.calls == clock_callback.calls == 0


def test_blank_generated_text_skips_callbacks_when_constructible() -> None:
    id_callback = Callback("synthetic_id")
    clock_callback = Callback(NOW)
    result = OrchestrationResult.model_construct(
        tenant_id="tenant_alpha",
        call_id="call_001",
        transcript_revision=7,
        generated_text=" ",
        citations=(),
    )

    suggestion = factory(
        id_callback=cast(Callable[[], str], id_callback),
        clock_callback=cast(Callable[[], datetime], clock_callback),
    ).create(
        event=transcript(),
        orchestration_result=result,
        current_seconds=7,
    )

    assert suggestion is None
    assert id_callback.calls == clock_callback.calls == 0


def test_blank_title_is_rejected() -> None:
    with pytest.raises(ValueError, match="title"):
        factory(title=" ")


def test_blank_optional_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="label_id"):
        factory(label_id=" ")


def test_negative_expiry_is_rejected() -> None:
    with pytest.raises(ValueError, match="expires_after_seconds"):
        factory(expires_after_seconds=-0.1)


@pytest.mark.parametrize(
    ("id_callback", "clock_callback", "message"),
    [
        (cast(Callable[[], str], object()), lambda: NOW, "suggestion_id_factory"),
        (lambda: "id", cast(Callable[[], datetime], object()), "utc_datetime_factory"),
    ],
)
def test_non_callable_callbacks_are_rejected_without_invocation(
    id_callback: Callable[[], str],
    clock_callback: Callable[[], datetime],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory(id_callback=id_callback, clock_callback=clock_callback)


@pytest.mark.parametrize("value", [" ", cast(str, 123)])
def test_invalid_id_callback_output_raises(value: str) -> None:
    with pytest.raises(ValueError, match="suggestion_id"):
        factory(id_callback=lambda: value).create(
            event=transcript(),
            orchestration_result=orchestration_result(),
            current_seconds=7,
        )


@pytest.mark.parametrize(
    "value",
    [
        cast(datetime, "not-a-datetime"),
        datetime(2026, 7, 27, 12, 0),
        datetime(
            2026,
            7,
            27,
            12,
            0,
            tzinfo=timezone(timedelta(hours=3)),
        ),
    ],
)
def test_invalid_or_non_utc_clock_output_raises(value: datetime) -> None:
    with pytest.raises(ValueError, match="utc_datetime_factory"):
        factory(clock_callback=lambda: value).create(
            event=transcript(),
            orchestration_result=orchestration_result(),
            current_seconds=7,
        )


@pytest.mark.parametrize(
    ("id_callback", "clock_callback", "message"),
    [
        (
            cast(Callable[[], str], Callback(RuntimeError("synthetic id failure"))),
            lambda: NOW,
            "id failure",
        ),
        (
            lambda: "id",
            cast(
                Callable[[], datetime],
                Callback(RuntimeError("synthetic clock failure")),
            ),
            "clock failure",
        ),
    ],
)
def test_callback_exceptions_propagate(
    id_callback: Callable[[], str],
    clock_callback: Callable[[], datetime],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        factory(
            id_callback=id_callback,
            clock_callback=clock_callback,
        ).create(
            event=transcript(),
            orchestration_result=orchestration_result(),
            current_seconds=7,
        )


def test_negative_current_seconds_is_rejected_before_callbacks() -> None:
    id_callback = Callback("synthetic_id")
    clock_callback = Callback(NOW)

    with pytest.raises(ValueError, match="current_seconds"):
        factory(
            id_callback=cast(Callable[[], str], id_callback),
            clock_callback=cast(Callable[[], datetime], clock_callback),
        ).create(
            event=transcript(),
            orchestration_result=orchestration_result(),
            current_seconds=-0.1,
        )

    assert id_callback.calls == clock_callback.calls == 0


def test_current_seconds_does_not_change_time_or_expiry() -> None:
    first = factory().create(
        event=transcript(),
        orchestration_result=orchestration_result(),
        current_seconds=1,
    )
    second = factory().create(
        event=transcript(),
        orchestration_result=orchestration_result(),
        current_seconds=999,
    )

    assert first == second
    assert first is not None
    assert first.created_at_utc == NOW
    assert first.expires_after_seconds == 30


def test_citations_are_not_encoded_and_inputs_are_unchanged() -> None:
    event = transcript()
    result = orchestration_result()
    before = (event.model_dump(), result.model_dump())

    suggestion = factory().create(
        event=event,
        orchestration_result=result,
        current_seconds=7,
    )

    assert suggestion is not None
    assert suggestion.evidence_ids == []
    assert before == (event.model_dump(), result.model_dump())


def test_optional_none_policy_is_preserved() -> None:
    suggestion = factory(
        label_id=None,
        expires_after_seconds=None,
    ).create(
        event=transcript(),
        orchestration_result=orchestration_result(),
        current_seconds=7,
    )

    assert suggestion is not None
    assert suggestion.label_id is None
    assert suggestion.expires_after_seconds is None


@pytest.mark.parametrize(
    "generated_text",
    [
        "Synthetic arbitrary non-empty text.",
        "{malformed",
        json.dumps(
            {
                "decision": "suggestion",
                "tenant_id": "tenant_alpha",
                "call_id": "call_001",
                "revision": 7,
            }
        ),
        json.dumps(
            {
                "decision": "suggestion",
                "tenant_id": "tenant_alpha",
                "call_id": "call_001",
                "revision": 7,
                "action": "RAG_ACTION",
                "title": "Synthetic title",
                "suggestion": "Synthetic guidance.",
                "priority": "HIGH",
                "citations": [{"document_id": "unknown", "chunk_id": "chunk_1"}],
                "source": "llm",
            }
        ),
    ],
)
def test_rejected_gate_output_skips_callbacks(generated_text: str) -> None:
    id_callback = Callback("synthetic_id")
    clock_callback = Callback(NOW)

    suggestion = factory(
        id_callback=cast(Callable[[], str], id_callback),
        clock_callback=cast(Callable[[], datetime], clock_callback),
    ).create(
        event=transcript(),
        orchestration_result=orchestration_result(generated_text=generated_text),
        current_seconds=7,
    )

    assert suggestion is None
    assert id_callback.calls == clock_callback.calls == 0


def test_valid_no_suggestion_output_is_not_admitted() -> None:
    raw_output = json.dumps(
        {
            "decision": "no_suggestion",
            "tenant_id": "tenant_alpha",
            "call_id": "call_001",
            "revision": 7,
        }
    )

    assert (
        factory().create(
            event=transcript(),
            orchestration_result=orchestration_result(generated_text=raw_output),
            current_seconds=7,
        )
        is None
    )


def test_existing_integration_exports_are_preserved() -> None:
    assert CoachingSuggestionFactory is not None
    assert DeterministicLLMCoachingSuggestionFactory is not None
    assert OrchestrationRunner is not None
    assert RAGCoachingProcessorDecorator is not None
