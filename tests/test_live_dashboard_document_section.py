"""Delegated document-section rendering tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.document_background import (
    DocumentSubmissionResult,
    DocumentSubmissionStatus,
)
from live_dashboard.document_section import render_document_section
from live_dashboard.document_view_models import (
    DocumentListItemViewModel,
    DocumentRuntimeStatus,
    DocumentSectionViewModel,
)


class _Streamlit:
    def __init__(self) -> None:
        self.rendered: list[str] = []
        self.buttons: dict[str, bool] = {}
        self.upload: _FakeUpload | None = None

    def subheader(self, value: object) -> None:
        self.rendered.append(str(value))

    def caption(self, value: object) -> None:
        self.rendered.append(str(value))

    def success(self, value: object) -> None:
        self.rendered.append(str(value))

    def info(self, value: object) -> None:
        self.rendered.append(str(value))

    def warning(self, value: object) -> None:
        self.rendered.append(str(value))

    def error(self, value: object) -> None:
        self.rendered.append(str(value))

    def write(self, value: object) -> None:
        self.rendered.append(str(value))

    def markdown(self, value: object) -> None:
        self.rendered.append(str(value))

    def progress(self, value: object, **kwargs: object) -> None:
        self.rendered.append(str(kwargs.get("text", value)))

    def file_uploader(self, *args: object, **kwargs: object):
        return self.upload

    def button(self, label: str, **kwargs: object) -> bool:
        return self.buttons.get(label, False)


@dataclass(frozen=True, slots=True)
class _FakeUpload:
    name: str
    type: str
    size: int
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


class _Resource:
    def __init__(self, view: DocumentSectionViewModel) -> None:
        self._view = view
        self.submissions: list[dict[str, object]] = []

    def view(self) -> DocumentSectionViewModel:
        return self._view

    def submit_upload(self, **kwargs: object) -> DocumentSubmissionResult:
        self.submissions.append(kwargs)
        return DocumentSubmissionResult(DocumentSubmissionStatus.ACCEPTED)


def _view(
    status: DocumentRuntimeStatus = DocumentRuntimeStatus.READY,
) -> DocumentSectionViewModel:
    return DocumentSectionViewModel(
        runtime_status=status,
        runtime_message={
            DocumentRuntimeStatus.READY: "Bilgi tabanı hazır",
            DocumentRuntimeStatus.DISABLED: "Bilgi tabanı yapılandırılmadı",
            DocumentRuntimeStatus.UNAVAILABLE: (
                "Bilgi tabanı geçici olarak kullanılamıyor; "
                "temel görüşme analizi devam ediyor"
            ),
        }[status],
        manager_busy=False,
        progress=None,
        documents=(),
    )


def test_unavailable_section_preserves_base_coaching_and_has_no_uploader() -> None:
    st = _Streamlit()
    resource = _Resource(_view(DocumentRuntimeStatus.UNAVAILABLE))
    render_document_section(st, resource=resource, session_state={})  # type: ignore[arg-type]
    rendered = " ".join(st.rendered)
    assert "temel görüşme analizi devam ediyor" in rendered
    assert resource.submissions == []


def test_valid_upload_renders_only_safe_metadata_and_delegates_once() -> None:
    st = _Streamlit()
    st.upload = _FakeUpload(
        name="rehber.md",
        type="text/markdown",
        size=18,
        content=b"Synthetic content",
    )
    st.buttons["Bilgi tabanına ekle"] = True
    resource = _Resource(_view())
    session: dict[str, object] = {}
    render_document_section(st, resource=resource, session_state=session)  # type: ignore[arg-type]
    assert len(resource.submissions) == 1
    rendered = " ".join(st.rendered)
    assert "rehber.md" in rendered
    assert "Markdown" in rendered
    assert "Synthetic content" not in rendered
    assert "sha256" not in rendered.casefold()
    assert "document_id" not in rendered


def test_document_list_never_renders_opaque_action_token() -> None:
    st = _Streamlit()
    document = DocumentListItemViewModel(
        filename="rehber.pdf",
        media_label="PDF",
        formatted_size="2.0 KiB",
        readiness_label="Hazır",
        created_at_utc="2026-08-05 10:30 UTC",
        action_token="opaque-private-action",
        active=False,
    )
    resource = _Resource(
        DocumentSectionViewModel(
            runtime_status=DocumentRuntimeStatus.READY,
            runtime_message="Bilgi tabanı hazır",
            manager_busy=False,
            progress=None,
            documents=(document,),
        )
    )
    render_document_section(st, resource=resource, session_state={})  # type: ignore[arg-type]
    assert "opaque-private-action" not in " ".join(st.rendered)


def test_app_adds_fourth_tab_and_only_delegates_document_rendering() -> None:
    source = (Path(__file__).parents[1] / "live_dashboard" / "app.py").read_text(
        encoding="utf-8"
    )
    assert '"Bilgi Tabanı"' in source
    assert "render_document_section(" in source
    assert "PersistentDocumentStorage" not in source
    assert "PsycopgDocumentRegistryRepository" not in source
    assert "embed_documents(" not in source
    assert source.count("_close_document_resource()") == 3  # definition + two resets
