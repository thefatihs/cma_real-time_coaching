from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.events.models import TranscriptEvent, TranscriptKind
from app.streaming.pipeline import (
    StreamingASRPlan,
    StreamingASRResult,
    StreamingASRStep,
)
from live_dashboard.demo_data import scenario_for, tenant_demos
from live_dashboard.uploaded_audio import (
    safe_upload_metadata,
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
    ordered_suggestions,
    progress_view,
    reset_runtime,
    reset_local_execution,
    responsive_rows,
    status_cards,
    suppression_reason_display,
    transcript_view,
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
    assert alpha.latest_labels[0].name == "fiyat_itirazi"
    assert beta.latest_labels[0].name == "butce_endisesi"
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
        TranscriptKind.STABLE, 2, "Aboneliğimi iptal etmek istiyorum."
    )
    final = pipeline_event(TranscriptKind.FINAL, 3, "İşlem bilgisini aldım.")
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
    )


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        audio_path: Path,
        call_id: str,
        *,
        step_callback=None,
        plan_callback=None,
    ):
        self.calls += 1
        result = fake_pipeline_result()
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
    assert state.runtime.latest_labels == ()
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


def test_reset_clears_local_execution_state() -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.request_start()
    execute_local_once(state, FakePipeline(), Path("fake.wav"))
    cleaned = reset_local_execution(state)
    assert cleaned.status == "idle"
    assert cleaned.pipeline_calls == cleaned.current_chunk == cleaned.total_chunks == 0
    assert cleaned.runtime.suggestions == []
    assert cleaned.asr_window_ms == []


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
    assert card.related_label == "İptal riski"
    assert card.evidence_ids == ("synthetic-a-cancel",)
    assert (card.priority_text, card.priority_symbol) == ("HIGH", "▲")


def test_intent_chip_formatting_is_compact_and_readable() -> None:
    subject = runtime("tenant_alpha", "critical")
    advance_runtime(subject)
    advance_runtime(subject)
    chips = intent_chips(subject.latest_labels)
    assert [(chip.text, chip.score, chip.is_risk, chip.symbol) for chip in chips] == [
        ("Kritik risk", "%100", True, "⚠")
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
    assert "İptal riski" in {chip.text for chip in result.detected_labels}


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
