from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import importlib
import sys
from typing import Any

import pytest

from app.events.models import CoachingSuggestionSource, SuggestionPriority
from live_dashboard.demo_data import tenant_demos
from live_dashboard.presentation import rag_runtime_status_text, ui_scope_identity
from live_dashboard.rag_runtime import DashboardRAGRuntimeStatus
from live_dashboard.view_models import (
    create_local_execution,
    dashboard_tabs,
    SpeakerCardViewModel,
    SpeakerDashboardViewModel,
    SuggestionCardViewModel,
)


UNAVAILABLE_TEXT = (
    "RAG geçici olarak kullanılamıyor; temel görüşme analizi devam ediyor"
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

    def markdown(self, value: object, **_kwargs: object) -> None:
        self.writes.append(str(value))

    def clear_rendered(self) -> None:
        self.captions.clear()
        self.infos.clear()
        self.writes.clear()
        self.metrics.clear()

    def __getattr__(self, _name: str) -> Any:
        return lambda *_args, **_kwargs: None


def _load_dashboard_app(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _RecordingStreamlit,
) -> Any:
    monkeypatch.setitem(sys.modules, "streamlit", recorder)
    sys.modules.pop("live_dashboard.app", None)
    app = importlib.import_module("live_dashboard.app")
    recorder.clear_rendered()
    return app


def _scope():
    return ui_scope_identity(
        tenant_id="tenant_alpha",
        call_id="synthetic-call",
        source_mode="synthetic-test",
    )


def _state_and_tabs():
    state = create_local_execution(
        tenant_demos()["tenant_alpha"],
        "synthetic-call",
    )
    return state, dashboard_tabs(state.runtime, state)


def _card(*, source: CoachingSuggestionSource) -> SuggestionCardViewModel:
    return SuggestionCardViewModel(
        suggestion_id="safe-suggestion",
        priority=SuggestionPriority.MEDIUM,
        priority_text="MEDIUM",
        title="Güvenli koçluk önerisi",
        suggestion="Görüşmeyi güvenli biçimde sürdürün.",
        action="Bilgilendir",
        timestamp="12:00:00",
        related_label=None,
        evidence_ids=("raw-citation-id-must-not-render",),
        priority_symbol="●",
        source=source.value,
        transcript_revision=0,
        is_new=True,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (DashboardRAGRuntimeStatus.READY, "RAG hazır"),
        (DashboardRAGRuntimeStatus.DISABLED, "RAG devre dışı"),
        (DashboardRAGRuntimeStatus.UNAVAILABLE, UNAVAILABLE_TEXT),
    ],
)
def test_exact_status_text(
    status: DashboardRAGRuntimeStatus,
    expected: str,
) -> None:
    assert rag_runtime_status_text(status) == expected
    assert rag_runtime_status_text(status) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "READY secret-dsn token endpoint model-name",
        {"exception": "provider response body", "document_id": "private"},
        object(),
    ],
)
def test_unexpected_state_fails_closed_without_sensitive_content(
    invalid: object,
) -> None:
    rendered = rag_runtime_status_text(invalid)

    assert rendered == UNAVAILABLE_TEXT
    assert all(
        marker not in rendered
        for marker in (
            "secret",
            "dsn",
            "token",
            "endpoint",
            "model",
            "provider",
            "exception",
            "document_id",
        )
    )


@pytest.mark.parametrize("status", tuple(DashboardRAGRuntimeStatus))
def test_base_dashboard_renders_in_every_state(
    monkeypatch: pytest.MonkeyPatch,
    status: DashboardRAGRuntimeStatus,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    state, tabs = _state_and_tabs()

    app._render_dashboard(state.runtime, state, None, _scope(), status)

    assert rag_runtime_status_text(status) in recorder.captions
    assert any("Henüz kesinleşen konuşma yok" in item for item in recorder.infos)
    assert ("Kesin transkript", "Bekleniyor") in recorder.metrics


def test_ready_preserves_llm_suggestion_speakers_metrics_and_hides_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    state, tabs = _state_and_tabs()
    representative = replace(
        tabs.representative,
        active_suggestions=(_card(source=CoachingSuggestionSource.LLM),),
        speaker_dashboard=SpeakerDashboardViewModel(
            speakers=(
                SpeakerCardViewModel("SPEAKER_1", "AGENT", 8, "HIGH", "strong_agent"),
                SpeakerCardViewModel(
                    "SPEAKER_2",
                    "CUSTOMER",
                    6,
                    "HIGH",
                    "strong_customer",
                ),
            ),
            speaker_count=2,
            turn_count=5,
            projected_customer_word_count=6,
            unknown_exclusion_count=0,
        ),
    )
    monkeypatch.setattr(
        app,
        "dashboard_tabs",
        lambda *_args, **_kwargs: replace(tabs, representative=representative),
    )

    app._render_dashboard(
        state.runtime,
        state,
        None,
        _scope(),
        DashboardRAGRuntimeStatus.READY,
    )

    rendered = " ".join([*recorder.captions, *recorder.writes])
    assert "Güvenli koçluk önerisi" in rendered
    assert "SPEAKER_1" in rendered and "SPEAKER_2" in rendered
    assert "AGENT" in rendered and "CUSTOMER" in rendered
    assert "raw-citation-id-must-not-render" not in rendered
    assert ("Konuşmacı", "2") in recorder.metrics
    assert ("Aktif koçluk", "1") in recorder.metrics


@pytest.mark.parametrize(
    "status",
    (
        DashboardRAGRuntimeStatus.DISABLED,
        DashboardRAGRuntimeStatus.UNAVAILABLE,
    ),
)
def test_non_ready_status_preserves_base_coaching(
    monkeypatch: pytest.MonkeyPatch,
    status: DashboardRAGRuntimeStatus,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    state, tabs = _state_and_tabs()
    representative = replace(
        tabs.representative,
        active_suggestions=(_card(source=CoachingSuggestionSource.RULE),),
    )
    monkeypatch.setattr(
        app,
        "dashboard_tabs",
        lambda *_args, **_kwargs: replace(tabs, representative=representative),
    )

    app._render_dashboard(
        state.runtime,
        state,
        None,
        _scope(),
        status,
    )

    assert "Güvenli koçluk önerisi" in recorder.writes
    assert rag_runtime_status_text(status) in recorder.captions


def test_rendering_is_rerun_consistent_and_has_no_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)
    state, _ = _state_and_tabs()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime lifecycle call occurred during rendering")

    monkeypatch.setattr(app, "_rag_runtime_controller", forbidden)
    monkeypatch.setattr(app, "_execution_resource", forbidden)
    monkeypatch.setattr(app, "_close_execution_resource", forbidden)

    for _ in range(2):
        app._render_dashboard(
            state.runtime,
            state,
            None,
            _scope(),
            DashboardRAGRuntimeStatus.READY,
        )

    assert recorder.captions.count("RAG hazır") == 2
