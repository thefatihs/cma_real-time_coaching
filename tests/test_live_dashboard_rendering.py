from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import importlib
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.events.models import SuggestionPriority
from live_dashboard.demo_data import tenant_demos
from live_dashboard.presentation import ui_scope_identity
from live_dashboard.rag_runtime import DashboardRAGRuntimeController
from live_dashboard.runtime_wiring import DashboardExecutionResourceRegistry
from live_dashboard.runtime_wiring import (
    ArtifactAvailability,
    DashboardServiceSelection,
)
from live_dashboard.view_models import (
    create_local_execution,
    dashboard_tabs,
    SpeakerCardViewModel,
    SpeakerDashboardViewModel,
    SuggestionCardViewModel,
)


class _SessionState(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class _Column:
    def __init__(self, recorder: _RecordingStreamlit) -> None:
        self._recorder = recorder

    def __enter__(self) -> _Column:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def metric(self, *args: object, **_kwargs: object) -> None:
        self._recorder.metrics.append(tuple(str(value) for value in args))

    def button(self, *_args: object, **_kwargs: object) -> bool:
        return False


class _RecordingStreamlit:
    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.captions: list[str] = []
        self.infos: list[str] = []
        self.writes: list[str] = []
        self.metrics: list[tuple[str, ...]] = []
        self.markdown_kwargs: list[dict[str, object]] = []
        self.toggle_values: dict[str, bool] = {}
        self.sidebar = nullcontext()

    def container(self, **_kwargs: object) -> Any:
        return nullcontext()

    def expander(self, *_args: object, **_kwargs: object) -> Any:
        return nullcontext()

    def tabs(self, labels: tuple[str, ...]) -> list[Any]:
        return [nullcontext() for _ in labels]

    def columns(
        self,
        spec: int | list[float],
        **_kwargs: object,
    ) -> list[_Column]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_Column(self) for _ in range(count)]

    def cache_resource(self, **_kwargs: object) -> Any:
        return lambda function: function

    def fragment(self, **_kwargs: object) -> Any:
        return lambda function: function

    def selectbox(
        self,
        _label: str,
        options: list[str],
        *,
        key: str | None = None,
        **_kwargs: object,
    ) -> str:
        value = options[0]
        if key is not None:
            self.session_state[key] = value
        return value

    def text_input(
        self,
        _label: str,
        value: str = "",
        *,
        key: str | None = None,
        **_kwargs: object,
    ) -> str:
        if key is not None:
            self.session_state[key] = value
        return value

    def radio(self, _label: str, options: tuple[str, ...], **_kwargs: object) -> str:
        return options[0]

    def slider(self, _label: str, *_args: float, key: str | None = None) -> float:
        value = float(_args[2])
        if key is not None:
            self.session_state[key] = value
        return value

    def checkbox(self, _label: str, *, value: bool, **_kwargs: object) -> bool:
        return value

    def toggle(
        self,
        label: str,
        *,
        value: bool,
        **_kwargs: object,
    ) -> bool:
        return self.toggle_values.get(label, value)

    def button(self, *_args: object, **_kwargs: object) -> bool:
        return False

    def caption(self, value: object, **_kwargs: object) -> None:
        self.captions.append(str(value))

    def info(self, value: object, **_kwargs: object) -> None:
        self.infos.append(str(value))

    def write(self, value: object, **_kwargs: object) -> None:
        self.writes.append(str(value))

    def markdown(self, value: object, **kwargs: object) -> None:
        self.writes.append(str(value))
        self.markdown_kwargs.append(kwargs)

    def __getattr__(self, _name: str) -> Any:
        return lambda *_args, **_kwargs: None


def _load_dashboard_app(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _RecordingStreamlit,
) -> Any:
    monkeypatch.setitem(sys.modules, "streamlit", recorder)
    sys.modules.pop("live_dashboard.app", None)
    return importlib.import_module("live_dashboard.app")


def _history_card() -> SuggestionCardViewModel:
    return SuggestionCardViewModel(
        suggestion_id="history-card",
        priority=SuggestionPriority.MEDIUM,
        priority_text="Orta",
        title="Önceki güvenli öneri",
        suggestion="Güvenli sentetik öneri.",
        action="notify",
        timestamp="12:00:00",
        related_label="Şikâyet",
        evidence_ids=(),
        priority_symbol="●",
        source="Sınıflandırma",
        transcript_revision=2,
        is_new=False,
    )


def _scope(call_id: str = "local-call"):
    return ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id=call_id,
        source_mode="synthetic-test",
    )


def test_dashboard_execution_resource_reuses_scope_and_session_keeps_only_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    registry = DashboardExecutionResourceRegistry(capacity=2)
    controller = DashboardRAGRuntimeController(registry=registry, environment={})
    monkeypatch.setattr(app, "_rag_runtime_controller", lambda _capacity: controller)
    runtime = create_local_execution(
        tenant_demos()["tenant_alpha"],
        "synthetic-call",
    ).runtime

    first = app._execution_resource(runtime)
    second = app._execution_resource(runtime)

    assert second is first
    assert recorder.session_state["dashboard_execution_resource_key"] == (
        first.opaque_key
    )
    assert not any(
        isinstance(value, type(first)) for value in recorder.session_state.values()
    )

    app._close_execution_resource()
    assert first.closed
    assert "dashboard_execution_resource_key" not in recorder.session_state


def test_dashboard_scope_change_closes_and_removes_previous_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    registry = DashboardExecutionResourceRegistry(capacity=1)
    controller = DashboardRAGRuntimeController(registry=registry, environment={})
    monkeypatch.setattr(app, "_rag_runtime_controller", lambda _capacity: controller)
    first_runtime = create_local_execution(
        tenant_demos()["tenant_alpha"],
        "call-one",
    ).runtime
    second_runtime = create_local_execution(
        tenant_demos()["tenant_alpha"],
        "call-two",
    ).runtime

    first = app._execution_resource(first_runtime)
    second = app._execution_resource(second_runtime)

    assert first.closed
    assert not second.closed
    assert second.identity.call_id == "call-two"
    assert recorder.session_state["dashboard_execution_resource_key"] == (
        second.opaque_key
    )


def test_pipeline_construction_receives_resource_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    runtime = create_local_execution(
        tenant_demos()["tenant_alpha"],
        "synthetic-call",
    ).runtime
    integration = object()
    resource: Any = SimpleNamespace(integration=integration)
    captured: dict[str, object] = {}
    expected_pipeline = object()
    monkeypatch.setattr(app, "_load_asr_model", lambda *_args: object())
    monkeypatch.setattr(app, "WindowTranscriber", lambda engine: engine)

    def build(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected_pipeline

    monkeypatch.setattr(app, "build_live_pipeline", build)

    result = app._make_pipeline(
        runtime,
        DashboardServiceSelection(enable_setfit=False, enable_coaching=True),
        ArtifactAvailability(compatible=True),
        resource,
    )

    assert result is expected_pipeline
    captured_kwargs = cast(dict[str, object], captured["kwargs"])
    assert captured_kwargs["integration"] is integration
    assert captured_kwargs["execution_resource"] is resource


def test_representative_renderer_handles_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)

    app._render_representative(state.runtime, tabs, _scope())


def test_representative_renderer_shows_bounded_speaker_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    speaker_dashboard = SpeakerDashboardViewModel(
        speakers=(
            SpeakerCardViewModel(
                slot="SPEAKER_1",
                role="AGENT",
                aligned_word_count=8,
                confidence_bucket="HIGH",
                decision_reason="strong_agent",
            ),
            SpeakerCardViewModel(
                slot="SPEAKER_2",
                role="Rol belirleniyor",
                aligned_word_count=5,
                confidence_bucket="NONE",
                decision_reason="insufficient",
            ),
        ),
        speaker_count=2,
        turn_count=4,
        projected_customer_word_count=0,
        unknown_exclusion_count=5,
    )
    representative = replace(
        tabs.representative,
        speaker_dashboard=speaker_dashboard,
    )
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)

    app._render_representative(
        state.runtime,
        replace(tabs, representative=representative),
        _scope(),
    )

    rendered = " ".join([*recorder.writes, *recorder.captions])
    assert "SPEAKER_1" in rendered
    assert "SPEAKER_2" in rendered
    assert "Rol belirleniyor" in rendered
    assert "private" not in rendered
    assert ("Konuşmacı", "2") in recorder.metrics
    assert ("UNKNOWN dışlama", "5") in recorder.metrics


def test_representative_renderer_shows_non_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    representative = replace(
        tabs.representative,
        suggestion_history=(_history_card(),),
    )
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    recorder.toggle_values["Önceki önerileri göster (1)"] = True

    app._render_representative(
        state.runtime,
        replace(tabs, representative=representative),
        _scope(),
    )

    assert "Önceki güvenli öneri" in " ".join(recorder.captions)


def test_representative_renderer_uses_native_safe_dynamic_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(
        tenant_demos()["tenant_alpha"],
        "sensitive-call-1234",
    )
    tabs = dashboard_tabs(state.runtime, state)
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    recorder.markdown_kwargs.clear()

    app._render_representative(
        state.runtime,
        tabs,
        _scope("sensitive-call-1234"),
    )

    assert not recorder.markdown_kwargs
    rendered_metrics = " ".join(
        value for metric in recorder.metrics for value in metric
    )
    assert "sensitive-call" not in rendered_metrics
    assert "••••1234" in rendered_metrics


def test_representative_renderer_has_safe_empty_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    recorder.infos.clear()

    app._render_representative(state.runtime, tabs, _scope())

    assert any("Henüz kesinleşen konuşma yok" in item for item in recorder.infos)
    assert any("Henüz güncel bir niyet" in item for item in recorder.infos)
    assert any("aktif bir koçluk önerisi yok" in item for item in recorder.infos)


def test_representative_renderer_preserves_card_order_and_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    high = replace(
        _history_card(),
        suggestion_id="high",
        priority=SuggestionPriority.HIGH,
        priority_text="HIGH",
        title="Yüksek öncelik",
    )
    critical = replace(
        _history_card(),
        suggestion_id="critical",
        priority=SuggestionPriority.CRITICAL,
        priority_text="CRITICAL",
        title="Kritik öncelik",
    )
    representative = replace(
        tabs.representative,
        active_suggestions=(critical, high),
    )
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    recorder.writes.clear()
    runtime_before = repr(state.runtime)

    app._render_representative(
        state.runtime,
        replace(tabs, representative=representative),
        _scope(),
    )

    assert recorder.writes.index("Kritik öncelik") < recorder.writes.index(
        "Yüksek öncelik"
    )
    assert repr(state.runtime) == runtime_before


def test_technical_renderer_is_collapsed_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    runtime_before = repr(state.runtime)
    view_before = repr(tabs)

    app._render_technical(tabs, _scope())

    assert any("varsayılan olarak kapalıdır" in item for item in recorder.infos)
    assert repr(state.runtime) == runtime_before
    assert repr(tabs) == view_before


def test_technical_renderer_hides_raw_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    technical = replace(
        tabs.technical,
        error="PRIVATE exception C:/secret/model-cache",
        failure_details=(
            ("Hata kodu", "PRIVATE_EXCEPTION"),
            ("Bileşen", "ASR"),
        ),
    )
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    recorder.toggle_values["Teknik ayrıntıları göster"] = True

    app._render_technical(replace(tabs, technical=technical), _scope())

    rendered = " ".join([*recorder.writes, *recorder.captions, *recorder.infos])
    assert "PRIVATE" not in rendered
    assert "secret" not in rendered
    assert "asr_processing_failed" in rendered


def test_representative_renderer_bounds_transcript_and_history_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    state.runtime.suggestion_history = [
        replace(
            _history_card(),
            suggestion_id=f"history-{index}",
            title=f"Geçmiş {index}",
            transcript_revision=index,
        )
        for index in range(8)
    ]
    tabs = dashboard_tabs(state.runtime, state)
    representative = replace(
        tabs.representative,
        transcript=replace(
            tabs.representative.transcript,
            stable_text="".join(str(index % 10) for index in range(7_000)),
        ),
    )
    scoped_tabs = replace(tabs, representative=representative)
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    recorder.toggle_values["Önceki önerileri göster (8)"] = True
    runtime_before = repr(state.runtime)

    app._render_representative(state.runtime, scoped_tabs, _scope())

    rendered = " ".join([*recorder.writes, *recorder.captions])
    assert "Geçmiş 7" in rendered
    assert "Geçmiş 3" in rendered
    assert "Geçmiş 2" not in rendered
    assert "3 eski öneri görünüm dışında" in rendered
    assert "1000 eski karakter" in rendered
    assert len(state.runtime.suggestion_history) == 8
    assert repr(state.runtime) == runtime_before
