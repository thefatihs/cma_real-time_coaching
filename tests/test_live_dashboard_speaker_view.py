from dataclasses import replace

from app.diarization.composition import (
    DiarizationCompositionOutcome,
    DiarizationCompositionReason,
    DiarizationCompositionStatus,
)
from app.diarization.models import DiarizationTurn, DiarizedWord, SpeakerRole
from app.diarization.role_resolver import (
    DirectRoleEvidenceOutcome,
    OppositeRoleInferenceBlockReason,
    RoleConfidenceBucket,
    RoleEvidenceCode,
    SpeakerRoleAssignment,
    SpeakerRoleDiagnostic,
    SpeakerRoleResolutionResult,
)
from app.diarization.routing import (
    CustomerProjectionReason,
    CustomerProjectionStatus,
    CustomerSpeechProjection,
    RoleTaggedWord,
)
from live_dashboard.demo_data import scenario_for, tenant_demos
from live_dashboard.view_models import (
    create_runtime,
    dashboard_tabs,
    speaker_dashboard_view,
)


TENANT_ID = "tenant_alpha"
CALL_ID = "synthetic-speaker-call"
REVISION = 0


def _diagnostic(
    role: SpeakerRole,
    *,
    bucket: RoleConfidenceBucket,
    reason: RoleEvidenceCode,
) -> SpeakerRoleDiagnostic:
    return SpeakerRoleDiagnostic(
        agent_evidence_hit_count=int(role is SpeakerRole.AGENT),
        customer_evidence_hit_count=int(role is SpeakerRole.CUSTOMER),
        direct_evidence_outcome=(
            DirectRoleEvidenceOutcome(role.value.lower())
            if role in {SpeakerRole.AGENT, SpeakerRole.CUSTOMER}
            else DirectRoleEvidenceOutcome.NONE
        ),
        agent_threshold_reached=role is SpeakerRole.AGENT,
        customer_threshold_reached=role is SpeakerRole.CUSTOMER,
        weak_opening_position_evidence_present=False,
        opposite_role_inference_attempted=False,
        opposite_role_inference_applied=False,
        inference_block_reason=OppositeRoleInferenceBlockReason.NONE,
        final_role=role,
        confidence_bucket=bucket,
        final_decision_reason=reason,
    )


def _outcome(
    roles: tuple[SpeakerRole, SpeakerRole],
    *,
    diagnostics: bool = True,
) -> DiarizationCompositionOutcome:
    speaker_ids = ("private-internal-a", "private-internal-b")
    assignments = tuple(
        SpeakerRoleAssignment(
            global_speaker_id=speaker_id,
            role=role,
            confidence=0.95 if role is not SpeakerRole.UNKNOWN else None,
            evidence=(
                RoleEvidenceCode.STRONG_AGENT
                if role is SpeakerRole.AGENT
                else RoleEvidenceCode.STRONG_CUSTOMER
                if role is SpeakerRole.CUSTOMER
                else RoleEvidenceCode.INSUFFICIENT
            ),
        )
        for speaker_id, role in zip(speaker_ids, roles, strict=True)
    )
    role_diagnostics = tuple(
        _diagnostic(
            role,
            bucket=(
                RoleConfidenceBucket.HIGH
                if role is not SpeakerRole.UNKNOWN
                else RoleConfidenceBucket.NONE
            ),
            reason=assignment.evidence,
        )
        for assignment, role in zip(assignments, roles, strict=True)
    )
    words = (
        DiarizedWord(
            tenant_id=TENANT_ID,
            call_id=CALL_ID,
            transcript_revision=REVISION,
            start_seconds=0.0,
            end_seconds=0.4,
            text="sentetik",
            local_speaker_ids=("local-a",),
            global_speaker_id=speaker_ids[0],
            global_speaker_ids=(speaker_ids[0],),
        ),
        DiarizedWord(
            tenant_id=TENANT_ID,
            call_id=CALL_ID,
            transcript_revision=REVISION,
            start_seconds=0.5,
            end_seconds=0.9,
            text="veri",
            local_speaker_ids=("local-a",),
            global_speaker_id=speaker_ids[0],
            global_speaker_ids=(speaker_ids[0],),
        ),
        DiarizedWord(
            tenant_id=TENANT_ID,
            call_id=CALL_ID,
            transcript_revision=REVISION,
            start_seconds=1.0,
            end_seconds=1.4,
            text="örneği",
            local_speaker_ids=("local-b",),
            global_speaker_id=speaker_ids[1],
            global_speaker_ids=(speaker_ids[1],),
        ),
    )
    customer_words = tuple(
        RoleTaggedWord(
            tenant_id=word.tenant_id,
            call_id=word.call_id,
            transcript_revision=word.transcript_revision,
            start_seconds=word.start_seconds,
            end_seconds=word.end_seconds,
            text=word.text,
            local_speaker_ids=word.local_speaker_ids,
            global_speaker_id=word.global_speaker_id,
            global_speaker_ids=word.global_speaker_ids,
            role=SpeakerRole.CUSTOMER,
            role_confidence=0.95,
            role_evidence=RoleEvidenceCode.STRONG_CUSTOMER,
        )
        for word in words
        if word.global_speaker_id == speaker_ids[1] and roles[1] is SpeakerRole.CUSTOMER
    )
    projection = CustomerSpeechProjection(
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        transcript_revision=REVISION,
        customer_words=customer_words,
        customer_text=" ".join(word.text for word in customer_words),
        customer_start_seconds=(
            customer_words[0].start_seconds if customer_words else None
        ),
        customer_end_seconds=(
            customer_words[-1].end_seconds if customer_words else None
        ),
        excluded_agent_word_count=2 if roles[0] is SpeakerRole.AGENT else 0,
        excluded_unknown_word_count=(
            sum(role is SpeakerRole.UNKNOWN for role in roles)
        ),
        excluded_overlap_word_count=0,
        excluded_below_confidence_word_count=0,
        status=(
            CustomerProjectionStatus.READY
            if customer_words
            else CustomerProjectionStatus.EMPTY
        ),
        reason=(
            CustomerProjectionReason.TRUSTED_CUSTOMER_SPEECH
            if customer_words
            else CustomerProjectionReason.NO_TRUSTED_CUSTOMER_SPEECH
        ),
    )
    return DiarizationCompositionOutcome(
        status=(
            DiarizationCompositionStatus.COMPLETED
            if customer_words
            else DiarizationCompositionStatus.EMPTY
        ),
        reason=(
            DiarizationCompositionReason.COMPOSED
            if customer_words
            else DiarizationCompositionReason.NO_CUSTOMER_SPEECH
        ),
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        transcript_revision=REVISION,
        tracked_turns=(
            DiarizationTurn(
                tenant_id=TENANT_ID,
                call_id=CALL_ID,
                start_seconds=0.0,
                end_seconds=0.9,
                local_speaker_ids=("local-a",),
                global_speaker_id=speaker_ids[0],
                global_speaker_ids=(speaker_ids[0],),
            ),
            DiarizationTurn(
                tenant_id=TENANT_ID,
                call_id=CALL_ID,
                start_seconds=1.0,
                end_seconds=1.4,
                local_speaker_ids=("local-b",),
                global_speaker_id=speaker_ids[1],
                global_speaker_ids=(speaker_ids[1],),
            ),
        ),
        diarized_words=words,
        role_resolution=SpeakerRoleResolutionResult(
            tenant_id=TENANT_ID,
            call_id=CALL_ID,
            transcript_revision=REVISION,
            assignments=assignments,
            diagnostics=role_diagnostics if diagnostics else (),
        ),
        customer_projection=projection,
    )


def test_known_roles_are_anonymized_and_aggregated() -> None:
    view = speaker_dashboard_view(
        _outcome((SpeakerRole.AGENT, SpeakerRole.CUSTOMER)),
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        transcript_revision=REVISION,
    )

    assert view is not None
    assert [(speaker.slot, speaker.role) for speaker in view.speakers] == [
        ("SPEAKER_1", "AGENT"),
        ("SPEAKER_2", "CUSTOMER"),
    ]
    assert [speaker.aligned_word_count for speaker in view.speakers] == [2, 1]
    assert (view.speaker_count, view.turn_count) == (2, 2)
    assert view.projected_customer_word_count == 1
    assert "private-internal" not in repr(view)


def test_unknown_roles_show_pending_without_guessing() -> None:
    view = speaker_dashboard_view(
        _outcome((SpeakerRole.UNKNOWN, SpeakerRole.UNKNOWN)),
        tenant_id=TENANT_ID,
        call_id=CALL_ID,
        transcript_revision=REVISION,
    )

    assert view is not None
    assert {speaker.role for speaker in view.speakers} == {"Rol belirleniyor"}
    assert {speaker.confidence_bucket for speaker in view.speakers} == {"NONE"}
    assert view.unknown_exclusion_count == 2


def test_missing_diagnostics_fail_closed_to_legacy_view() -> None:
    assert (
        speaker_dashboard_view(
            _outcome(
                (SpeakerRole.AGENT, SpeakerRole.CUSTOMER),
                diagnostics=False,
            ),
            tenant_id=TENANT_ID,
            call_id=CALL_ID,
            transcript_revision=REVISION,
        )
        is None
    )


def test_malformed_scope_fails_closed_to_legacy_view() -> None:
    malformed = replace(
        _outcome((SpeakerRole.AGENT, SpeakerRole.CUSTOMER)),
        call_id="other-call",
    )

    assert (
        speaker_dashboard_view(
            malformed,
            tenant_id=TENANT_ID,
            call_id=CALL_ID,
            transcript_revision=REVISION,
        )
        is None
    )


def test_dashboard_tabs_legacy_fallback_is_unchanged() -> None:
    tenant = tenant_demos()[TENANT_ID]
    runtime = create_runtime(
        tenant,
        scenario_for(TENANT_ID, "cancel"),
        CALL_ID,
    )

    assert dashboard_tabs(runtime) == dashboard_tabs(
        runtime,
        diarization_outcome=None,
    )
