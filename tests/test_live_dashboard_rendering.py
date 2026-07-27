from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import importlib
import sys
from typing import Any

import pytest

from app.events.models import SuggestionPriority
from live_dashboard.demo_data import tenant_demos
from live_dashboard.view_models import (
    create_local_execution,
    dashboard_tabs,
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
    def __enter__(self) -> _Column:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def metric(self, *_args: object, **_kwargs: object) -> None:
        return None

    def button(self, *_args: object, **_kwargs: object) -> bool:
        return False


class _RecordingStreamlit:
    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.captions: list[str] = []
        self.sidebar = nullcontext()

    def container(self, **_kwargs: object) -> Any:
        return nullcontext()

    def expander(self, *_args: object, **_kwargs: object) -> Any:
        return nullcontext()

    def tabs(self, labels: tuple[str, ...]) -> list[Any]:
        return [nullcontext() for _ in labels]

    def columns(self, spec: int | list[float]) -> list[_Column]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_Column() for _ in range(count)]

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

    def button(self, *_args: object, **_kwargs: object) -> bool:
        return False

    def caption(self, value: object, **_kwargs: object) -> None:
        self.captions.append(str(value))

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


def test_representative_renderer_handles_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_local_execution(tenant_demos()["tenant_alpha"], "local-call")
    tabs = dashboard_tabs(state.runtime, state)
    recorder = _RecordingStreamlit()
    app = _load_dashboard_app(monkeypatch, recorder)

    app._render_representative(state.runtime, tabs)


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

    app._render_representative(
        state.runtime,
        replace(tabs, representative=representative),
    )

    assert "Önceki Öneriler" in recorder.captions
