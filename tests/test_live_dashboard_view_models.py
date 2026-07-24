from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.classification.streaming import (
    ClassificationProcessingStatus,
    StableClassificationOutcome,
)
from app.coaching.coordinator import (
    CoachingCoordinatorResult,
    CoachingProcessingStatus,
    StableCoachingOutcome,
)
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
from app.streaming.pipeline import (
    StreamingASRPlan,
    StreamingASRResult,
    StreamingASRStep,
)
from live_dashboard.demo_data import scenario_for, tenant_demos
from live_dashboard.uploaded_audio import (
    safe_upload_metadata,
    safe_upload_identity,
    temporary_uploaded_audio,
)
from live_dashboard.view_models import (
    advance_runtime,
    apply_feedback,
    create_local_execution,
    create_runtime,
    dashboard_tabs,
    execute_local_once,
    intent_chips,
    intent_label,
    ordered_suggestions,
    progress_view,
    reset_runtime,
    reset_local_execution,
    responsive_rows,
    status_cards,
    suggestion_card,
    suppression_reason_display,
    transcript_view,
    UploadedAudioSession,
)


def runtime(tenant_id: str = "tenant_alpha", scenario_id: str = "cancel"):
    tenant = tenant_demos()[tenant_id]
    return create_runtime(tenant, scenario_for(tenant_id, scenario_id), "test-call")


def test_partial_updates_transcript_without_coaching_card() -> None:
    subject = runtime()
    advance_runtime(subject)
    view = transcript_view(subject)
    assert view.partial_text == "Aboneliğimi"
    assert view.stable_text == ""
    assert view.partial_is_changeable
    assert subject.suggestions == []
    assert subject.latest_labels == ()


def test_stable_event_creates_formatted_coaching_card() -> None:
    subject = runtime()
    advance_runtime(subject)
    advance_runtime(subject)
    card = subject.suggestions[0]
    assert (card.priority_text, card.title, card.action) == (
        "HIGH",
        "İptal talebini doğrulayın",
        "Hazır öneri",
    )
    assert card.suggestion
    assert card.timestamp.count(":") == 2
    assert transcript_view(subject).latest_event_type == "STABLE"


def test_priority_ordering_places_critical_first() -> None:
    normal = runtime("tenant_alpha", "price")
    critical = runtime("tenant_alpha", "critical")
    for subject in (normal, critical):
        while advance_runtime(subject) is not None:
            pass
    cards = ordered_suggestions([*normal.suggestions, *critical.suggestions])
    assert [card.priority_text for card in cards] == ["CRITICAL", "HIGH"]


def test_timeline_is_chronological_and_has_required_event_types() -> None:
    subject = runtime()
    while advance_runtime(subject) is not None:
        pass
    assert subject.timeline == sorted(subject.timeline, key=lambda item: item.timestamp)
    assert {item.event_type for item in subject.timeline} >= {
        "Transkript",
        "Sınıflandırma",
        "Öneri gösterildi",
        "Öneri bastırıldı",
    }


def test_duplicate_suppression_reason_is_displayed() -> None:
    subject = runtime()
    while advance_runtime(subject) is not None:
        pass
    assert subject.suppression_reasons == ["yinelenen öneri"]
    assert suppression_reason_display("cooldown") == "bekleme süresi"


def test_tenant_labels_rules_and_state_are_isolated() -> None:
    alpha = runtime("tenant_alpha", "price")
    beta = runtime("tenant_beta", "price")
    for subject in (alpha, beta):
        advance_runtime(subject)
        advance_runtime(subject)
    assert alpha.call_state.tenant_id != beta.call_state.tenant_id
    assert alpha.latest_labels[0].name == "price_objection"
    assert beta.latest_labels[0].name == "price_objection"
    assert alpha.suggestions[0].title != beta.suggestions[0].title


def test_negated_cancellation_does_not_trigger() -> None:
    subject = runtime("tenant_alpha", "negated")
    while advance_runtime(subject) is not None:
        pass
    assert subject.suggestions == []
    assert subject.latest_labels == ()


def test_reset_creates_clean_call_state_and_coordinator() -> None:
    subject = runtime()
    advance_runtime(subject)
    cleaned = reset_runtime(subject)
    assert cleaned is not subject
    assert (
        cleaned.call_state.stable_transcript
        == cleaned.call_state.partial_transcript
        == ""
    )
    assert cleaned.next_event_index == 0
    assert cleaned.suggestions == cleaned.timeline == []


def test_view_models_contain_no_audio_or_private_data() -> None:
    subject = runtime("tenant_beta", "product")
    while advance_runtime(subject) is not None:
        pass
    rendered = repr(subject)
    assert "audio_bytes" not in rendered
    assert "CallMetricPrivate" not in rendered
    assert "müşteri adı" not in rendered.casefold()


def pipeline_event(kind: TranscriptKind, revision: int, text: str) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id="tenant_alpha",
        call_id="local-call",
        event_id=f"event-{revision}",
        kind=kind,
        text=text,
        start_seconds=float(revision - 1),
        end_seconds=float(revision),
        revision=revision,
        created_at_utc=datetime(2026, 7, 22, 10, revision, tzinfo=UTC),
    )


def fake_pipeline_result() -> StreamingASRResult:
    partial = pipeline_event(TranscriptKind.PARTIAL, 1, "Aboneliğimi")
    stable = pipeline_event(
        TranscriptKind.STABLE,
        2,
        ("Aboneliğimi bugün iptal ettirmek istiyorum. Lütfen iptal işlemini başlatın."),
    )
    final = pipeline_event(TranscriptKind.FINAL, 3, "İşlem bilgisini aldım.")
    classification = ClassificationResultEvent(
        tenant_id="tenant_alpha",
        call_id="local-call",
        transcript_event_id=stable.event_id,
        labels=[ClassificationLabel(name="cancellation_request", score=1.0)],
        action=CoachingAction.TEMPLATE_ACTION,
        model_id="common_turkish_setfit_v2",
        threshold_profile_id="common_turkish_setfit_v2:calibrated:v1",
        probabilities={"cancellation_request": 1.0},
        thresholds={"cancellation_request": 0.7},
        processing_time_ms=4.0,
        created_at_utc=stable.created_at_utc,
    )
    suggestion = CoachingSuggestionEvent(
        tenant_id="tenant_alpha",
        call_id="local-call",
        suggestion_id="live-suggestion-1",
        source_transcript_event_id=stable.event_id,
        action=CoachingAction.TEMPLATE_ACTION,
        priority=SuggestionPriority.HIGH,
        source=CoachingSuggestionSource.BOTH,
        label_id="cancellation_request",
        title="İptal talebini doğrulayın",
        suggestion="Müşterinin iptal talebini açıkça doğrulayın.",
        created_at_utc=stable.created_at_utc,
    )
    classification_outcome = StableClassificationOutcome(
        status=ClassificationProcessingStatus.CLASSIFIED,
        transcript_revision=stable.revision,
        source_sequence=None,
        classification_event=classification,
    )
    coaching_outcome = StableCoachingOutcome(
        status=CoachingProcessingStatus.PROCESSED,
        transcript_revision=stable.revision,
        result=CoachingCoordinatorResult(
            classification_event=classification,
            displayed_suggestions=(suggestion,),
            suppressed_suggestions=(),
            matched_rule_ids=("cancel-rule",),
            suppression_reasons=(),
            transcript_revision=stable.revision,
        ),
    )
    steps = (
        StreamingASRStep(
            tenant_id="tenant_alpha",
            call_id="local-call",
            sequence_number=0,
            chunk_start_seconds=0.0,
            chunk_end_seconds=1.0,
            window_start_seconds=0.0,
            window_end_seconds=1.0,
            window_duration_seconds=1.0,
            raw_window_text="synthetic partial",
            transcript_events=(partial,),
            stable_transcript="",
            partial_transcript=partial.text,
            transcription_time_seconds=0.2,
        ),
        StreamingASRStep(
            tenant_id="tenant_alpha",
            call_id="local-call",
            sequence_number=1,
            chunk_start_seconds=1.0,
            chunk_end_seconds=2.0,
            window_start_seconds=0.0,
            window_end_seconds=2.0,
            window_duration_seconds=2.0,
            raw_window_text="synthetic stable",
            transcript_events=(stable,),
            stable_transcript=stable.text,
            partial_transcript="",
            transcription_time_seconds=0.3,
            classification_outcomes=(classification_outcome,),
            coaching_outcomes=(coaching_outcome,),
        ),
    )
    return StreamingASRResult(
        tenant_id="tenant_alpha",
        call_id="local-call",
        steps=steps,
        final_event=final,
        stable_transcript=f"{stable.text} {final.text}",
        partial_transcript="",
        total_chunks=2,
        audio_duration_seconds=2.0,
        classification_outcomes=(classification_outcome,),
        coaching_outcomes=(coaching_outcome,),
    )


class FakePipeline:
    def __init__(self, result: StreamingASRResult | None = None) -> None:
        self.calls = 0
        self.result = result or fake_pipeline_result()

    def run(
        self,
        audio_path: Path,
        call_id: str,
        *,
        step_callback=None,
        plan_callback=None,
    ):
        self.calls += 1
        result = self.result
        if plan_callback is not None:
            plan_callback(
                StreamingASRPlan(
                    tenant_id=result.tenant_id,
                    call_id=result.call_id,
                    total_chunks=result.total_chunks,
                    audio_duration_seconds=result.audio_duration_seconds,
                )
            )
        for step in result.steps:
            if step_callback is not None:
                step_callback(step)
        return result


def test_local_mode_requires_start_and_pipeline_runs_once() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    pipeline = FakePipeline()
    assert not execute_local_once(state, pipeline, Path("fake.wav"))
    state.request_start()
    assert execute_local_once(
        state,
        pipeline,
        Path("fake.wav"),
        clock=iter((1.0, 1.2, 1.4, 1.5)).__next__,
    )
    state.request_start()
    assert not execute_local_once(state, pipeline, Path("fake.wav"))
    assert pipeline.calls == state.pipeline_calls == 1


def test_local_results_update_transcript_coaching_and_latency() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(
        state,
        FakePipeline(),
        Path("fake.wav"),
        clock=iter((2.0, 2.1, 2.3, 2.4)).__next__,
    )
    assert state.runtime.call_state.partial_transcript == ""
    assert "İşlem bilgisini aldım." in state.runtime.call_state.stable_transcript
    assert [label.name for label in state.runtime.latest_labels] == [
        "cancellation_request"
    ]
    assert len(state.runtime.suggestions) == 1
    assert state.runtime.latency is not None
    assert state.runtime.latency.asr_ms == 300
    assert state.asr_window_ms == [200, 300]
    assert state.processing_seconds == pytest.approx(0.4)
    assert state.real_time_factor == pytest.approx(0.2)


def test_partial_local_event_does_not_trigger_coaching_but_stable_does() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("fake.wav"))
    suggestion_events = [
        item for item in state.runtime.timeline if item.event_type == "Öneri gösterildi"
    ]
    assert len(suggestion_events) == 1
    assert suggestion_events[0].detail == "İptal talebini doğrulayın"


def test_exact_cancellation_runtime_outcome_reaches_dashboard_card() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("synthetic.wav"))
    tabs = dashboard_tabs(state.runtime, state)
    assert (
        "Aboneliğimi bugün iptal ettirmek istiyorum."
        in tabs.representative.transcript.stable_text
    )
    assert len(tabs.representative.suggestions) == 1
    assert tabs.representative.suggestions[0].source == "Kural + sınıflandırma"
    assert tabs.representative.suppressed_count == 0


def test_reset_clears_local_execution_state() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("fake.wav"))
    cleaned = reset_local_execution(state)
    assert cleaned.status == "idle"
    assert cleaned.pipeline_calls == cleaned.current_chunk == cleaned.total_chunks == 0
    assert cleaned.runtime.suggestions == []
    assert cleaned.asr_window_ms == []


def test_safe_pipeline_failure_logs_metadata_without_private_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_transcript = "PRIVATE_TRANSCRIPT_CONTENT"
    private_path = Path("C:/CallMetricPrivate/customer-name.wav")

    class FailingPipeline(FakePipeline):
        def run(self, *args: object, **kwargs: object) -> StreamingASRResult:
            plan_callback = kwargs.get("plan_callback")
            step_callback = kwargs.get("step_callback")
            result = fake_pipeline_result()
            if callable(plan_callback):
                plan_callback(
                    StreamingASRPlan(
                        tenant_id=result.tenant_id,
                        call_id=result.call_id,
                        total_chunks=3,
                        audio_duration_seconds=3,
                    )
                )
            if callable(step_callback):
                step_callback(result.steps[0])
                step_callback(result.steps[1])
            raise RuntimeError(f"{private_transcript} {private_path}")

    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.runtime.setfit_enabled = True
    state.runtime.coaching_enabled = True
    state.request_start()
    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError):
        execute_local_once(state, FailingPipeline(), private_path)
    assert state.safe_failure is not None
    assert state.safe_failure.chunk_sequence == 2
    assert state.safe_failure.error_code == "RuntimeError"
    assert state.safe_failure.classification_enabled
    assert state.safe_failure.coaching_enabled
    assert private_transcript not in caplog.text
    assert "CallMetricPrivate" not in caplog.text
    tabs = dashboard_tabs(state.runtime, state)
    assert ("Hata kodu", "RuntimeError") in tabs.technical.failure_details
    assert ("Parça", "3") in tabs.technical.failure_details
    assert "RuntimeError" not in repr(tabs.representative)
    assert "Ses işleme güvenli biçimde tamamlanamadı." in (
        tabs.representative.safe_messages
    )


def test_uploaded_file_switch_creates_fresh_call_without_manual_reset() -> None:
    session = UploadedAudioSession()
    tenant = tenant_demos()["tenant_alpha"]
    identity_a = safe_upload_identity(b"synthetic-file-a")
    identity_b = safe_upload_identity(b"synthetic-file-b")
    first, changed = session.select(
        identity=identity_a,
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    assert changed
    first.status = "completed"
    first.stage = "Tamamlandı"
    first.current_chunk = 3
    first.total_chunks = 3
    first.processing_seconds = 4.0
    first.audio_duration_seconds = 10.0
    first.runtime.call_state.active_labels = ["complaint"]
    coaching_result = fake_pipeline_result().coaching_outcomes[0].result
    assert coaching_result is not None
    first.runtime.suggestions.append(
        suggestion_card(coaching_result.displayed_suggestions[0])
    )

    same, changed = session.select(
        identity=identity_a,
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    assert same is first
    assert not changed

    second, changed = session.select(
        identity=identity_b,
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    assert changed
    assert second is not first
    assert second.runtime.call_id != first.runtime.call_id
    assert second.runtime.call_state.transcript_revision == 0
    assert second.runtime.call_state.active_labels == []
    assert second.runtime.suggestions == []
    assert second.current_chunk == second.total_chunks == 0
    assert second.safe_failure is None
    assert second.status == "idle"
    assert second.stage == "Başlatılmadı"
    assert progress_view(second).percentage == 0
    assert second.start_enabled
    assert not second.stop_enabled
    assert session.selected_file_identity == identity_b
    assert session.initialized_run_file_identity == identity_b


def test_manual_upload_reset_changes_widget_generation_once() -> None:
    session = UploadedAudioSession()
    tenant = tenant_demos()["tenant_alpha"]
    first, _ = session.select(
        identity=safe_upload_identity(b"first"),
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    session.reset()
    assert session.uploader_generation == 1
    assert session.execution is None
    assert session.selected_file_identity is None
    assert session.initialized_run_file_identity is None
    after_reset, changed = session.select(
        identity=safe_upload_identity(b"next"),
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    assert changed
    assert after_reset is not first
    assert session.uploader_generation == 1
    same, changed = session.select(
        identity=safe_upload_identity(b"next"),
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    assert same is after_reset
    assert not changed
    assert session.uploader_generation == 1


@pytest.mark.parametrize("prior_status", ["completed", "error", "stopped"])
def test_file_change_atomically_resets_any_terminal_run(prior_status: str) -> None:
    session = UploadedAudioSession()
    tenant = tenant_demos()["tenant_alpha"]
    first, _ = session.select(
        identity=safe_upload_identity(b"file-a"),
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    first.status = prior_status
    first.stage = "Eski durum"
    first.start_requested = True
    first.current_chunk = 2
    first.total_chunks = 2
    first.elapsed_seconds = 12
    first.processing_seconds = 8
    first.audio_duration_seconds = 10
    first.error_message = "old failure"
    first.failed_chunk = 2
    first.runtime.call_state.stable_transcript = "old stable"
    first.runtime.call_state.partial_transcript = "old partial"
    first.runtime.call_state.active_labels = ["complaint"]

    second, changed = session.select(
        identity=safe_upload_identity(b"file-b"),
        tenant=tenant,
        base_call_id="dashboard-call",
    )
    assert changed
    assert second.status == "idle"
    assert not second.start_requested
    assert second.stage == "Başlatılmadı"
    assert second.current_chunk == second.total_chunks == 0
    assert second.elapsed_seconds == 0
    assert second.processing_seconds is None
    assert second.audio_duration_seconds is None
    assert second.error_message is None
    assert second.failed_chunk is None
    assert second.runtime.call_state.stable_transcript == ""
    assert second.runtime.call_state.partial_transcript == ""
    assert second.runtime.call_state.active_labels == []
    assert second.start_enabled
    assert not second.stop_enabled


def test_responsive_rows_preserve_untruncated_status_values() -> None:
    subject = runtime("tenant_alpha", "product")
    cards = status_cards(subject, "Uzun fakat eksiksiz işlem hattı durumu")
    rows = responsive_rows(cards, 3)
    assert [card.value for row in rows for card in row] == [
        card.value for card in cards
    ]
    assert all(len(row) <= 3 for row in rows)


def test_uploaded_audio_uses_and_deletes_temporary_file() -> None:
    metadata = safe_upload_metadata("sentetik.wav", 4)
    assert (metadata.filename, metadata.format_name, metadata.size_bytes) == (
        "sentetik.wav",
        "WAV",
        4,
    )
    with temporary_uploaded_audio("sentetik.wav", bytes([1, 2, 3, 4])) as path:
        assert path.exists()
        assert "callmetric-upload-" in path.name
    assert not path.exists()
    assert "audio_bytes" not in repr(metadata)


def test_progress_total_is_known_before_first_completed_chunk() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    snapshots: list[tuple[int, int]] = []
    state.request_start()
    execute_local_once(
        state,
        FakePipeline(),
        Path("fake.wav"),
        plan_progress_callback=lambda plan: snapshots.append(
            (plan.total_chunks, state.current_chunk)
        ),
    )
    assert snapshots == [(2, 0)]


def test_progress_percentage_eta_and_completed_state() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    assert progress_view(state).eta == "Tahmin hazırlanıyor"
    captured = []
    state.request_start()
    execute_local_once(
        state,
        FakePipeline(),
        Path("fake.wav"),
        progress_callback=lambda step: captured.append(progress_view(state)),
    )
    assert captured[0].percentage == 50.0
    assert captured[0].eta == "Tahmini kalan süre: 0 sn"
    completed = progress_view(state)
    assert completed.percentage == 100.0
    assert completed.completed_chunks == completed.total_chunks == 2
    assert completed.stage == "Tamamlandı"


def test_eta_uses_human_readable_rolling_asr_average() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.total_chunks = 4
    state.current_chunk = 1
    state.asr_window_ms = [45_000.0]
    assert progress_view(state).eta == "Tahmini kalan süre: 2 dk 15 sn"


def test_tenant_display_names_change_but_ids_do_not() -> None:
    demos = tenant_demos()
    assert demos["tenant_alpha"].config.context.tenant_name == "Demo Telekom"
    assert demos["tenant_beta"].config.context.tenant_name == "Demo Yazılım"
    assert demos["tenant_alpha"].config.context.tenant_id == "tenant_alpha"
    assert demos["tenant_beta"].config.context.tenant_id == "tenant_beta"


def test_three_tab_data_has_clear_presentation_boundaries() -> None:
    subject = runtime("tenant_alpha", "price")
    advance_runtime(subject)
    advance_runtime(subject)
    tabs = dashboard_tabs(subject)
    assert tabs.representative.transcript.stable_text
    assert not hasattr(tabs.representative, "pipeline_statuses")
    assert tabs.technical.pipeline_statuses
    assert tabs.result.waiting_message


def test_coaching_card_includes_label_evidence_and_priority_symbol() -> None:
    subject = runtime("tenant_alpha", "cancel")
    advance_runtime(subject)
    advance_runtime(subject)
    card = dashboard_tabs(subject).representative.suggestions[0]
    assert card.related_label == "İptal Talebi"
    assert card.evidence_ids == ("synthetic-a-cancel",)
    assert (card.priority_text, card.priority_symbol) == ("HIGH", "▲")
    tabs = dashboard_tabs(subject)
    assert "iptal_riski" not in repr(tabs.representative.intent_chips)
    assert "iptal_riski" not in repr(tabs.representative.detected_intent_chips)
    assert tabs.representative.intent_chips[0].text == "İptal Talebi"
    assert tabs.technical.revision_label_timeline
    assert "text" not in type(tabs.technical.revision_label_timeline[0]).model_fields
    assert "probabilities" not in repr(tabs.technical.revision_label_timeline)


def test_intent_chip_formatting_is_compact_and_readable() -> None:
    subject = runtime("tenant_alpha", "cancel")
    advance_runtime(subject)
    advance_runtime(subject)
    chips = intent_chips(subject.latest_labels)
    assert [(chip.text, chip.score, chip.is_risk, chip.symbol) for chip in chips] == [
        ("İptal Talebi", "%100", True, "⚠")
    ]


def test_latency_chart_input_is_ordered_by_chunk() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.asr_window_ms = [310.0, 120.0, 205.0]
    chart = dashboard_tabs(state.runtime, state).technical.asr_chart
    assert chart == ((1, 310.0), (2, 120.0), (3, 205.0))


def test_completed_call_summary_contains_safe_product_data() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("fake.wav"))
    metadata = safe_upload_metadata("sentetik.wav", 2048)
    result = dashboard_tabs(state.runtime, state, metadata).result
    assert result.completed
    assert result.final_transcript
    assert result.model_name == "large-v3"
    assert result.language == "tr"
    assert result.audio_metadata == (
        ("Dosya", "sentetik.wav"),
        ("Biçim", "WAV"),
        ("Boyut", "2.0 KB"),
    )
    assert "İptal Talebi" in {chip.text for chip in result.detected_labels}


def test_feedback_is_session_only_and_does_not_change_coaching() -> None:
    subject = runtime("tenant_alpha", "cancel")
    advance_runtime(subject)
    advance_runtime(subject)
    before = tuple(subject.suggestions)
    feedback = apply_feedback({}, "card-1", "Uygulandı")
    assert feedback == {"card-1": "Uygulandı"}
    assert tuple(subject.suggestions) == before


def test_tab_models_expose_no_audio_bytes_or_private_paths() -> None:
    subject = runtime("tenant_beta", "product")
    advance_runtime(subject)
    rendered = repr(dashboard_tabs(subject))
    assert "audio_bytes" not in rendered
    assert "CallMetricPrivate" not in rendered


def test_general_setfit_labels_have_required_turkish_names() -> None:
    assert {
        label: intent_label(label)
        for label in (
            "product_information",
            "price_objection",
            "cancellation_request",
            "technical_issue",
            "complaint",
            "renewal_interest",
            "churn_risk",
            "no_action",
        )
    } == {
        "product_information": "Ürün Bilgisi",
        "price_objection": "Fiyat İtirazı",
        "cancellation_request": "İptal Talebi",
        "technical_issue": "Teknik Sorun",
        "complaint": "Şikâyet",
        "renewal_interest": "Yenileme İlgisi",
        "churn_risk": "Müşteri Kaybı Riski",
        "no_action": "Aksiyon Gerekmiyor",
    }


@pytest.mark.parametrize(
    ("source", "display"),
    [
        (CoachingSuggestionSource.RULE, "Kural"),
        (CoachingSuggestionSource.CLASSIFICATION, "Sınıflandırma"),
        (CoachingSuggestionSource.BOTH, "Kural + sınıflandırma"),
    ],
)
def test_suggestion_card_shows_priority_action_and_provenance(
    source: CoachingSuggestionSource, display: str
) -> None:
    event = fake_pipeline_result().coaching_outcomes[0].result
    assert event is not None
    source_event = event.displayed_suggestions[0].model_copy(update={"source": source})
    card = suggestion_card(source_event, transcript_revision=2)
    assert (card.priority_text, card.action, card.source) == (
        "HIGH",
        "Hazır öneri",
        display,
    )
    assert card.transcript_revision == 2
    assert card.is_new


def test_simultaneous_suggestions_keep_their_own_label_metadata() -> None:
    outcome = fake_pipeline_result().coaching_outcomes[0].result
    assert outcome is not None
    base = outcome.displayed_suggestions[0]
    product = base.model_copy(
        update={
            "suggestion_id": "product-card",
            "label_id": "product_information",
            "title": "Ürün bilgisini açıklayın",
            "suggestion": "Ürün bilgisini kısa biçimde açıklayın.",
            "priority": SuggestionPriority.MEDIUM,
        }
    )
    price = base.model_copy(
        update={
            "suggestion_id": "price-card",
            "label_id": "price_objection",
            "title": "Fiyat itirazını karşılayın",
            "suggestion": "Fiyat itirazını dikkatle karşılayın.",
            "priority": SuggestionPriority.HIGH,
        }
    )
    cards = ordered_suggestions(
        [
            suggestion_card(product),
            suggestion_card(price),
        ]
    )
    assert [(card.title, card.related_label, card.priority) for card in cards] == [
        ("Fiyat itirazını karşılayın", "Fiyat İtirazı", SuggestionPriority.HIGH),
        ("Ürün bilgisini açıklayın", "Ürün Bilgisi", SuggestionPriority.MEDIUM),
    ]


def test_live_outcomes_are_deduplicated_and_technical_metadata_is_separated() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("synthetic.wav"))
    tabs = dashboard_tabs(state.runtime, state)
    assert len(tabs.representative.suggestions) == 1
    assert "common_turkish_setfit_v2" not in repr(tabs.representative)
    assert "0.9" not in repr(tabs.representative)
    assert ("Model", "common_turkish_setfit_v2") in (
        tabs.technical.classification_metadata
    )
    assert tabs.technical.probabilities == (("cancellation_request", 1.0),)
    assert tabs.technical.rtf
    assert tabs.technical.last_asr == "300 ms"


def test_dashboard_separates_current_and_call_level_labels_without_context_text() -> (
    None
):
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("synthetic.wav"))
    state.runtime.call_state.record_detected_labels(
        ["technical_issue"],
        transcript_revision=1,
        source=CoachingSuggestionSource.CLASSIFICATION,
        model_id="synthetic-classifier",
    )

    tabs = dashboard_tabs(state.runtime, state)
    assert [chip.text for chip in tabs.representative.intent_chips] == ["İptal Talebi"]
    assert {chip.text for chip in tabs.representative.detected_intent_chips} == {
        "İptal Talebi",
        "Teknik Sorun",
    }
    assert tabs.technical.current_labels == ("cancellation_request",)
    assert set(tabs.technical.detected_labels) == {
        "cancellation_request",
        "technical_issue",
    }
    assert "synthetic.wav" not in repr(tabs.technical.classification_metadata)
    assert "Aboneliğimi" not in repr(tabs.technical.classification_metadata)


def test_disabled_modes_show_calm_empty_state() -> None:
    disabled_result = replace(
        fake_pipeline_result(),
        classification_outcomes=(),
        coaching_outcomes=(),
        steps=tuple(
            replace(step, classification_outcomes=(), coaching_outcomes=())
            for step in fake_pipeline_result().steps
        ),
    )
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(disabled_result), Path("synthetic.wav"))
    tabs = dashboard_tabs(state.runtime, state)
    assert tabs.representative.suggestions == ()
    assert (
        tabs.representative.empty_suggestion_message
        == "Şu anda gösterilecek yeni bir koçluk önerisi yok."
    )
    assert ("SetFit", "disabled") in tabs.technical.pipeline_statuses
    assert ("Coaching", "disabled") in tabs.technical.pipeline_statuses


def test_failure_outcomes_are_safe_and_do_not_expose_transcript_or_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    base = fake_pipeline_result()
    classification_failure = StableClassificationOutcome(
        status=ClassificationProcessingStatus.FAILED,
        transcript_revision=2,
        source_sequence=None,
    )
    coaching_failure = StableCoachingOutcome(
        status=CoachingProcessingStatus.FAILED,
        transcript_revision=2,
        error_type="RuntimeError",
        error_code="coaching_failed",
    )
    failed = replace(
        base,
        classification_outcomes=(classification_failure,),
        coaching_outcomes=(coaching_failure,),
        steps=tuple(
            replace(
                step,
                classification_outcomes=(
                    (classification_failure,) if step.sequence_number == 1 else ()
                ),
                coaching_outcomes=(
                    (coaching_failure,) if step.sequence_number == 1 else ()
                ),
            )
            for step in base.steps
        ),
    )
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(
        state,
        FakePipeline(failed),
        Path("C:/CallMetricPrivate/never-render-this.wav"),
    )
    tabs = dashboard_tabs(state.runtime, state)
    assert len(tabs.representative.safe_messages) == 2
    rendered = repr(tabs)
    assert "RuntimeError" not in rendered
    assert "CallMetricPrivate" not in rendered
    assert "never-render-this" not in rendered
    assert "Aboneliğimi bugün iptal ettirmek istiyorum." not in caplog.text
