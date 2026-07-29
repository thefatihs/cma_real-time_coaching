from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.diarization.models import SpeakerRole
from app.diarization.role_resolver import (
    DirectRoleEvidenceOutcome,
    OppositeRoleInferenceBlockReason,
    RoleEvidenceCode,
    RoleConfidenceBucket,
    RuleBasedSpeakerRoleResolver,
    SpeakerAttributedTextSpan,
    SpeakerRoleAssignment,
    SpeakerRoleDiagnostic,
    SpeakerRoleResolutionError,
    SpeakerRoleResolutionErrorCategory,
    SpeakerRoleResolutionRequest,
    SpeakerRoleResolutionResult,
    SpeakerRoleResolverProtocol,
)


def _span(
    text: str,
    speaker_id: str | None,
    *,
    start: float = 0.0,
    end: float = 1.0,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 4,
    role: SpeakerRole = SpeakerRole.UNKNOWN,
    extra_speaker_id: str | None = None,
) -> SpeakerAttributedTextSpan:
    speaker_ids = () if speaker_id is None else (speaker_id,)
    if extra_speaker_id is not None:
        speaker_ids += (extra_speaker_id,)
    return SpeakerAttributedTextSpan(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        start_seconds=start,
        end_seconds=end,
        global_speaker_ids=speaker_ids,
        role=role,
        text=text,
    )


def _request(
    *spans: SpeakerAttributedTextSpan,
    tenant_id: str = "tenant-a",
    call_id: str = "call-a",
    revision: int = 4,
) -> SpeakerRoleResolutionRequest:
    return SpeakerRoleResolutionRequest(
        tenant_id=tenant_id,
        call_id=call_id,
        transcript_revision=revision,
        spans=spans,
    )


def _assignment(
    result: SpeakerRoleResolutionResult,
    speaker_id: str | None,
) -> SpeakerRoleAssignment:
    return next(
        assignment
        for assignment in result.assignments
        if assignment.global_speaker_id == speaker_id
    )


def _diagnostic(
    result: SpeakerRoleResolutionResult,
    speaker_id: str | None,
) -> SpeakerRoleDiagnostic:
    index = next(
        index
        for index, assignment in enumerate(result.assignments)
        if assignment.global_speaker_id == speaker_id
    )
    return result.diagnostics[index]


def test_strong_greeting_resolves_agent() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Merhaba, ben Ayşe.", "CALL_SPEAKER_0001"))
    )

    assignment = _assignment(result, "CALL_SPEAKER_0001")
    assert assignment.role is SpeakerRole.AGENT
    assert assignment.evidence is RoleEvidenceCode.STRONG_AGENT
    assert assignment.confidence == 1.0
    diagnostic = _diagnostic(result, "CALL_SPEAKER_0001")
    assert diagnostic.agent_evidence_hit_count == 1
    assert diagnostic.customer_evidence_hit_count == 0
    assert diagnostic.direct_evidence_outcome is DirectRoleEvidenceOutcome.AGENT
    assert diagnostic.agent_threshold_reached is True
    assert diagnostic.customer_threshold_reached is False
    assert diagnostic.final_role is SpeakerRole.AGENT
    assert diagnostic.confidence_bucket is RoleConfidenceBucket.HIGH
    assert diagnostic.final_decision_reason is RoleEvidenceCode.STRONG_AGENT


def test_strong_customer_request_resolves_customer() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Bir sorun yaşıyorum.", "CALL_SPEAKER_0001"))
    )

    assignment = _assignment(result, "CALL_SPEAKER_0001")
    assert assignment.role is SpeakerRole.CUSTOMER
    assert assignment.evidence is RoleEvidenceCode.STRONG_CUSTOMER
    diagnostic = _diagnostic(result, "CALL_SPEAKER_0001")
    assert diagnostic.agent_evidence_hit_count == 0
    assert diagnostic.customer_evidence_hit_count == 1
    assert diagnostic.direct_evidence_outcome is DirectRoleEvidenceOutcome.CUSTOMER
    assert diagnostic.agent_threshold_reached is False
    assert diagnostic.customer_threshold_reached is True


def test_first_speaker_position_alone_remains_unknown() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Günaydın.", "CALL_SPEAKER_0001", start=0, end=1))
    )

    assignment = _assignment(result, "CALL_SPEAKER_0001")
    assert assignment.role is SpeakerRole.UNKNOWN
    assert assignment.evidence is RoleEvidenceCode.WEAK_POSITIONAL
    assert assignment.confidence is None
    diagnostic = _diagnostic(result, "CALL_SPEAKER_0001")
    assert diagnostic.agent_evidence_hit_count == 0
    assert diagnostic.customer_evidence_hit_count == 0
    assert diagnostic.direct_evidence_outcome is DirectRoleEvidenceOutcome.NONE
    assert diagnostic.weak_opening_position_evidence_present is True
    assert diagnostic.confidence_bucket is RoleConfidenceBucket.NONE
    assert (
        diagnostic.inference_block_reason
        is OppositeRoleInferenceBlockReason.SPEAKER_COUNT_NOT_TWO
    )


def test_high_confidence_agent_allows_safe_opposite_customer_inference() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("Size nasıl yardımcı olabilirim?", "CALL_SPEAKER_0001"),
            _span(
                "Teşekkür ederim.",
                "CALL_SPEAKER_0002",
                start=1,
                end=2,
            ),
        )
    )

    inferred = _assignment(result, "CALL_SPEAKER_0002")
    assert inferred.role is SpeakerRole.CUSTOMER
    assert inferred.evidence is RoleEvidenceCode.INFERRED_OPPOSITE
    inferred_diagnostic = _diagnostic(result, "CALL_SPEAKER_0002")
    assert inferred_diagnostic.opposite_role_inference_attempted is True
    assert inferred_diagnostic.opposite_role_inference_applied is True
    assert (
        inferred_diagnostic.inference_block_reason
        is OppositeRoleInferenceBlockReason.NONE
    )
    assert inferred_diagnostic.final_role is SpeakerRole.CUSTOMER
    assert inferred_diagnostic.confidence_bucket is RoleConfidenceBucket.HIGH
    assert (
        inferred_diagnostic.final_decision_reason is RoleEvidenceCode.INFERRED_OPPOSITE
    )


def test_conflicting_evidence_remains_unknown_and_blocks_inference() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span(
                "Merhaba ben arıyorum ve sorun yaşıyorum.",
                "CALL_SPEAKER_0001",
            ),
            _span("Merhaba.", "CALL_SPEAKER_0002", start=1, end=2),
        )
    )

    conflicted = _assignment(result, "CALL_SPEAKER_0001")
    assert conflicted.role is SpeakerRole.UNKNOWN
    assert conflicted.evidence is RoleEvidenceCode.CONFLICTING
    assert _assignment(result, "CALL_SPEAKER_0002").role is SpeakerRole.UNKNOWN
    diagnostic = _diagnostic(result, "CALL_SPEAKER_0001")
    assert diagnostic.agent_evidence_hit_count == 1
    assert diagnostic.customer_evidence_hit_count == 1
    assert diagnostic.direct_evidence_outcome is DirectRoleEvidenceOutcome.CONFLICTING
    assert diagnostic.agent_threshold_reached is True
    assert diagnostic.customer_threshold_reached is True
    assert diagnostic.opposite_role_inference_attempted is False
    assert (
        diagnostic.inference_block_reason
        is OppositeRoleInferenceBlockReason.ROLE_CARDINALITY_MISMATCH
    )


def test_conflicting_unknown_blocks_opposite_inference_with_fixed_reason() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("Kontrol ediyorum.", "CALL_SPEAKER_0001"),
            _span(
                "Merhaba ben arıyorum ve sorun yaşıyorum.",
                "CALL_SPEAKER_0002",
                start=1,
                end=2,
            ),
        )
    )

    diagnostic = _diagnostic(result, "CALL_SPEAKER_0002")
    assert diagnostic.opposite_role_inference_attempted is True
    assert diagnostic.opposite_role_inference_applied is False
    assert (
        diagnostic.inference_block_reason
        is OppositeRoleInferenceBlockReason.CONFLICTING_DIRECT_EVIDENCE
    )


def test_subthreshold_opposite_evidence_blocks_inference_with_fixed_reason() -> None:
    resolver = RuleBasedSpeakerRoleResolver(
        agent_phrases=("agent one", "agent two"),
        customer_phrases=("customer one",),
        agent_threshold=2,
    )
    result = resolver.resolve(
        _request(
            _span("agent one agent two", "CALL_SPEAKER_0001"),
            _span("agent one", "CALL_SPEAKER_0002", start=1, end=2),
        )
    )

    diagnostic = _diagnostic(result, "CALL_SPEAKER_0002")
    assert diagnostic.agent_evidence_hit_count == 1
    assert diagnostic.agent_threshold_reached is False
    assert diagnostic.opposite_role_inference_attempted is True
    assert diagnostic.opposite_role_inference_applied is False
    assert (
        diagnostic.inference_block_reason
        is OppositeRoleInferenceBlockReason.OPPOSITE_ROLE_EVIDENCE_PRESENT
    )


def test_two_unknown_speakers_report_role_cardinality_block() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("Tamam.", "CALL_SPEAKER_0001"),
            _span("Peki.", "CALL_SPEAKER_0002", start=10, end=11),
        )
    )

    first = _diagnostic(result, "CALL_SPEAKER_0001")
    second = _diagnostic(result, "CALL_SPEAKER_0002")
    assert first.weak_opening_position_evidence_present is True
    assert second.weak_opening_position_evidence_present is False
    assert first.opposite_role_inference_attempted is False
    assert (
        first.inference_block_reason
        is OppositeRoleInferenceBlockReason.ROLE_CARDINALITY_MISMATCH
    )


def test_insufficient_evidence_and_more_than_two_speakers_remain_unknown() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("Tamam.", "CALL_SPEAKER_0001", start=10, end=11),
            _span("Peki.", "CALL_SPEAKER_0002", start=11, end=12),
            _span("Anladım.", "CALL_SPEAKER_0003", start=12, end=13),
        )
    )

    assert {assignment.role for assignment in result.assignments} == {
        SpeakerRole.UNKNOWN
    }


def test_overlap_text_is_ignored_as_independent_evidence() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span(
                "Merhaba ben müşteri temsilcisiyim.",
                "CALL_SPEAKER_0001",
                extra_speaker_id="CALL_SPEAKER_0002",
                role=SpeakerRole.OVERLAP,
            )
        )
    )

    assert result.assignments == ()
    assert result.ignored_evidence == (RoleEvidenceCode.OVERLAP_IGNORED,)


def test_two_speakers_receive_isolated_direct_evidence() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(
            _span("İşleminizi gerçekleştiriyorum.", "CALL_SPEAKER_0001"),
            _span(
                "İptal etmek istiyorum.",
                "CALL_SPEAKER_0002",
                start=1,
                end=2,
            ),
        )
    )

    assert _assignment(result, "CALL_SPEAKER_0001").role is SpeakerRole.AGENT
    assert _assignment(result, "CALL_SPEAKER_0002").role is SpeakerRole.CUSTOMER


@pytest.mark.parametrize(
    ("span", "category"),
    [
        (
            _span("Merhaba.", "speaker", tenant_id="tenant-b"),
            SpeakerRoleResolutionErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _span("Merhaba.", "speaker", call_id="call-b"),
            SpeakerRoleResolutionErrorCategory.SCOPE_MISMATCH,
        ),
        (
            _span("Merhaba.", "speaker", revision=5),
            SpeakerRoleResolutionErrorCategory.REVISION_MISMATCH,
        ),
    ],
)
def test_scope_and_revision_mismatch_rejected(
    span: SpeakerAttributedTextSpan,
    category: SpeakerRoleResolutionErrorCategory,
) -> None:
    with pytest.raises(SpeakerRoleResolutionError) as error:
        RuleBasedSpeakerRoleResolver().resolve(_request(span))

    assert error.value.category is category


def test_resolution_is_deterministic_and_input_is_not_mutated() -> None:
    resolver: SpeakerRoleResolverProtocol = RuleBasedSpeakerRoleResolver()
    spans = (
        _span("Sorun yaşıyorum.", "CALL_SPEAKER_0002", start=2, end=3),
        _span("Merhaba ben Deniz.", "CALL_SPEAKER_0001", start=0, end=1),
    )
    snapshot = deepcopy(spans)
    request = _request(*spans)

    first = resolver.resolve(request)
    second = resolver.resolve(request)

    assert first == second
    assert spans == snapshot
    with pytest.raises(ValidationError):
        spans[0].text = "değişti"  # type: ignore[misc]


def test_text_processing_is_bounded_fail_closed() -> None:
    resolver = RuleBasedSpeakerRoleResolver(max_text_characters=8)

    with pytest.raises(SpeakerRoleResolutionError) as error:
        resolver.resolve(_request(_span("uzun gizli metin", "speaker")))

    assert (
        error.value.category is SpeakerRoleResolutionErrorCategory.TEXT_LIMIT_EXCEEDED
    )


def test_missing_global_identity_returns_unknown() -> None:
    result = RuleBasedSpeakerRoleResolver().resolve(
        _request(_span("Sorun yaşıyorum.", None))
    )

    assignment = _assignment(result, None)
    assert assignment.role is SpeakerRole.UNKNOWN
    assert assignment.evidence is RoleEvidenceCode.MISSING_GLOBAL_ID


def test_sensitive_text_is_absent_from_result_repr_and_safe_errors() -> None:
    private_text = "özel müşteri metni ve dosya yolu"
    span = _span(private_text, "speaker")
    result = RuleBasedSpeakerRoleResolver().resolve(_request(span))

    assert private_text not in repr(span)
    assert private_text not in repr(result)
    serialized_diagnostic = result.diagnostics[0].model_dump_json()
    assert private_text not in serialized_diagnostic
    assert "CALL_SPEAKER_0001" not in serialized_diagnostic
    with pytest.raises(SpeakerRoleResolutionError) as error:
        RuleBasedSpeakerRoleResolver(max_text_characters=2).resolve(_request(span))
    assert str(error.value) == "text_limit_exceeded"
    assert private_text not in repr(error.value)
