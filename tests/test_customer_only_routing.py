from copy import deepcopy
from datetime import UTC, datetime

from app.calls.models import CallState
from app.diarization.models import SpeakerRole
from app.diarization.role_resolver import RoleEvidenceCode
from app.diarization.routing import (
    CustomerProjectionReason,
    CustomerProjectionStatus,
    CustomerSpeechProjection,
    RoleTaggedWord,
)
from app.events.models import TranscriptEvent, TranscriptKind
from app.streaming.customer_routing import (
    CustomerOnlyClassificationRouter,
    CustomerRoutingReason,
    CustomerRoutingStatus,
)


NOW = datetime(2026, 7, 29, tzinfo=UTC)


def _event(
    *,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 1,
    text: str = "Temsilci ve müşteri karışık konuşması.",
) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id=tenant_id,
        call_id=call_id,
        event_id=f"event-{revision}",
        kind=TranscriptKind.STABLE,
        text=text,
        start_seconds=0,
        end_seconds=2,
        revision=revision,
        created_at_utc=NOW,
        source_chunk_sequence=revision,
    )


def _state(event: TranscriptEvent) -> CallState:
    state = CallState(tenant_id=event.tenant_id, call_id=event.call_id)
    state.apply_transcript(event)
    return state


def _projection(
    *,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 1,
    text: str = "İptal etmek istiyorum.",
) -> CustomerSpeechProjection:
    if not text:
        return CustomerSpeechProjection(
            tenant_id=tenant_id,
            call_id=call_id,
            transcript_revision=revision,
            customer_words=(),
            customer_text="",
            excluded_agent_word_count=1,
            excluded_unknown_word_count=0,
            excluded_overlap_word_count=0,
            excluded_below_confidence_word_count=0,
            status=CustomerProjectionStatus.EMPTY,
            reason=CustomerProjectionReason.NO_TRUSTED_CUSTOMER_SPEECH,
        )
    words = tuple(
        RoleTaggedWord(
            tenant_id=tenant_id,
            call_id=call_id,
            transcript_revision=revision,
            start_seconds=float(index),
            end_seconds=float(index + 1),
            text=word,
            local_speaker_ids=("local-customer",),
            global_speaker_id="CALL_SPEAKER_0002",
            global_speaker_ids=("CALL_SPEAKER_0002",),
            role=SpeakerRole.CUSTOMER,
            role_confidence=1.0,
            role_evidence=RoleEvidenceCode.STRONG_CUSTOMER,
        )
        for index, word in enumerate(text.split())
    )
    return CustomerSpeechProjection(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        customer_words=words,
        customer_text=text,
        customer_start_seconds=words[0].start_seconds,
        customer_end_seconds=words[-1].end_seconds,
        excluded_agent_word_count=1,
        excluded_unknown_word_count=0,
        excluded_overlap_word_count=0,
        excluded_below_confidence_word_count=0,
        status=CustomerProjectionStatus.READY,
        reason=CustomerProjectionReason.TRUSTED_CUSTOMER_SPEECH,
    )


class FakeProjectionProvider:
    def __init__(
        self,
        projection: CustomerSpeechProjection | None,
        *,
        fail: bool = False,
    ) -> None:
        self.projection = projection
        self.fail = fail
        self.calls: list[tuple[str, str, int]] = []

    def get_projection(
        self,
        *,
        tenant_id: str,
        call_id: str,
        transcript_revision: int,
    ) -> CustomerSpeechProjection | None:
        self.calls.append((tenant_id, call_id, transcript_revision))
        if self.fail:
            raise RuntimeError("private provider detail")
        return self.projection


def test_disabled_path_preserves_legacy_event_without_requesting_projection() -> None:
    event = _event()
    provider = FakeProjectionProvider(_projection(), fail=True)

    decision = CustomerOnlyClassificationRouter(
        enabled=False,
        projection_provider=provider,
    ).prepare(event, _state(event))

    assert decision.outcome.status is CustomerRoutingStatus.LEGACY_PATH
    assert decision.routed_event is event
    assert provider.calls == []


def test_enabled_path_routes_only_customer_projection_text() -> None:
    event = _event(text="Temsilci: iptal etmek istiyorum. Müşteri: devam.")
    provider = FakeProjectionProvider(_projection(text="Devam etmek istiyorum."))

    decision = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=provider,
    ).prepare(event, _state(event))

    assert decision.outcome.status is CustomerRoutingStatus.CUSTOMER_PROCESSED
    assert decision.routed_event is not None
    assert decision.routed_event.text == "Devam etmek istiyorum."
    assert "iptal" not in decision.routed_event.text.casefold()


def test_empty_projection_is_a_normal_safe_skip() -> None:
    event = _event()
    state = _state(event)
    state.active_labels = ["complaint"]
    snapshot = deepcopy(state)
    decision = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=FakeProjectionProvider(_projection(text="")),
    ).prepare(event, state)

    assert decision.outcome.status is CustomerRoutingStatus.NO_CUSTOMER_SPEECH
    assert decision.outcome.reason is CustomerRoutingReason.EMPTY_PROJECTION
    assert decision.routed_event is None
    assert state == snapshot


def test_missing_wrong_scope_and_wrong_revision_fail_closed() -> None:
    event = _event()
    state = _state(event)
    cases = (
        None,
        _projection(tenant_id="tenant-b"),
        _projection(call_id="call-b"),
        _projection(revision=2),
    )

    for projection in cases:
        decision = CustomerOnlyClassificationRouter(
            enabled=True,
            projection_provider=FakeProjectionProvider(projection),
        ).prepare(event, state)
        assert decision.outcome.status is CustomerRoutingStatus.REJECTED
        assert decision.routed_event is None


def test_provider_exception_fails_safe_without_state_mutation() -> None:
    event = _event()
    state = _state(event)
    snapshot = deepcopy(state)

    decision = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=FakeProjectionProvider(_projection(), fail=True),
    ).prepare(event, state)

    assert decision.outcome.status is CustomerRoutingStatus.FAILED_SAFE
    assert decision.outcome.reason is CustomerRoutingReason.PROVIDER_FAILURE
    assert state == snapshot


def test_same_revision_is_idempotent_and_out_of_order_is_rejected() -> None:
    second = _event(revision=2)
    provider = FakeProjectionProvider(_projection(revision=2))
    router = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=provider,
    )

    assert router.prepare(second, _state(second)).outcome.status is (
        CustomerRoutingStatus.CUSTOMER_PROCESSED
    )
    assert router.prepare(second, _state(second)).outcome.status is (
        CustomerRoutingStatus.ALREADY_PROCESSED
    )
    first = _event(revision=1)
    assert router.prepare(first, _state(first)).outcome.reason is (
        CustomerRoutingReason.OUT_OF_ORDER_REVISION
    )
    assert len(provider.calls) == 1


def test_tenant_and_call_idempotency_are_isolated() -> None:
    router = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=FakeProjectionProvider(_projection()),
    )
    first = _event()
    assert router.prepare(first, _state(first)).outcome.status is (
        CustomerRoutingStatus.CUSTOMER_PROCESSED
    )

    second = _event(tenant_id="tenant-b", call_id="call-b")
    router_b = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=FakeProjectionProvider(
            _projection(tenant_id="tenant-b", call_id="call-b")
        ),
    )
    assert router_b.prepare(second, _state(second)).outcome.status is (
        CustomerRoutingStatus.CUSTOMER_PROCESSED
    )


def test_outcome_and_decision_repr_never_expose_text() -> None:
    private_text = "özel müşteri iptal metni"
    event = _event(text="özel karışık metin")
    decision = CustomerOnlyClassificationRouter(
        enabled=True,
        projection_provider=FakeProjectionProvider(_projection(text=private_text)),
    ).prepare(event, _state(event))

    assert private_text not in repr(decision)
    assert event.text not in repr(decision)
