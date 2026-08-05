"""Delegated Streamlit renderer for the tenant-scoped knowledge-base tab."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import uuid4

from app.ingestion.document_background import DocumentSubmissionStatus
from live_dashboard.document_runtime import DashboardDocumentResource
from live_dashboard.document_view_models import (
    DocumentRuntimeStatus,
    safe_upload_selection,
)

_UPLOAD_GENERATION_KEY = "document_upload_generation"
_SUBMISSION_TOKEN_KEY = "document_submission_token"
_CONFIRMATION_TOKEN_KEY = "document_delete_confirmation_token"


class DocumentSessionState(Protocol):
    """Minimal safe session-state operations used by this renderer."""

    def __contains__(self, key: object, /) -> bool: ...

    def __getitem__(self, key: str) -> object: ...

    def __delitem__(self, key: str) -> None: ...

    def __setitem__(self, key: str, value: object) -> None: ...


def render_document_section(
    st: Any,
    *,
    resource: DashboardDocumentResource,
    session_state: DocumentSessionState,
) -> None:
    """Render safe projections and delegate every operation to the resource."""
    view = resource.view()
    st.subheader("Bilgi Tabanı")
    st.caption(
        "PDF, TXT ve Markdown belgeleri görüşmelerde kullanılabilecek RAG bilgisine "
        "dönüştürülür. Belgenin hazır olması, bilginin bir görüşmede kullanıldığı "
        "anlamına gelmez."
    )
    if view.runtime_status is DocumentRuntimeStatus.READY:
        st.success(view.runtime_message)
    elif view.runtime_status is DocumentRuntimeStatus.DISABLED:
        st.info(view.runtime_message)
    else:
        st.warning(view.runtime_message)
    if view.runtime_status is not DocumentRuntimeStatus.READY:
        return

    stored_generation = _setdefault(session_state, _UPLOAD_GENERATION_KEY, 0)
    if type(stored_generation) is not int or stored_generation < 0:
        raise ValueError("document upload generation is invalid")
    generation = stored_generation
    uploader = st.file_uploader(
        "Bilgi tabanı belgesi",
        type=["pdf", "txt", "md"],
        disabled=view.manager_busy,
        key=f"document_upload_{generation}",
    )
    selection = None
    if uploader is not None:
        try:
            selection = safe_upload_selection(
                filename=uploader.name,
                media_type=uploader.type,
                byte_size=uploader.size,
            )
            st.caption(
                f"{selection.filename} · {selection.media_label} · "
                f"{selection.formatted_size}"
            )
        except ValueError as error:
            st.error(str(error))
    if st.button(
        "Bilgi tabanına ekle",
        disabled=selection is None or view.manager_busy,
        use_container_width=True,
        key=f"document_submit_{generation}",
    ):
        token = _setdefault(session_state, _SUBMISSION_TOKEN_KEY, uuid4().hex)
        if not isinstance(token, str) or selection is None or uploader is None:
            st.error("Belge yükleme isteği doğrulanamadı.")
        else:
            result = resource.submit_upload(
                submission_token=token,
                content=uploader.getvalue(),
                original_filename=selection.filename,
                media_type=selection.media_type,
            )
            if result.status is DocumentSubmissionStatus.ACCEPTED:
                st.success("Belge işleme kuyruğuna alındı.")
                session_state[_UPLOAD_GENERATION_KEY] = generation + 1
                _discard(session_state, _SUBMISSION_TOKEN_KEY)
            elif result.status is DocumentSubmissionStatus.BUSY:
                st.warning("Belge işleme kapasitesi dolu; daha sonra yeniden deneyin.")
            else:
                st.error("Belge yükleme güvenli biçimde başlatılamadı.")

    if view.progress is not None:
        st.info(view.progress.label)
        total_chunks = view.progress.total_chunks
        processed_chunks = view.progress.processed_chunks
        if (
            total_chunks is not None
            and total_chunks > 0
            and processed_chunks is not None
            and 0 <= processed_chunks <= total_chunks
        ):
            st.progress(
                min(
                    processed_chunks / total_chunks,
                    1.0,
                ),
                text=f"{processed_chunks}/{total_chunks} parça",
            )
        if view.progress.active and st.button("İşlemi iptal et"):
            if resource.cancel_active_submission():
                st.warning("Belge işleme iptal ediliyor.")
            else:
                st.warning("Belge işleme şu anda iptal edilemedi.")

    confirmation_token = _get(session_state, _CONFIRMATION_TOKEN_KEY)
    if isinstance(confirmation_token, str):
        filename = resource.confirmation_filename(confirmation_token=confirmation_token)
        if filename is None:
            _discard(session_state, _CONFIRMATION_TOKEN_KEY)
        else:
            st.warning(f"{filename} kalıcı olarak silinecek. Bu işlem geri alınamaz.")
            if st.button("Belgeyi kalıcı olarak sil"):
                result = resource.confirm_delete(confirmation_token=confirmation_token)
                _discard(session_state, _CONFIRMATION_TOKEN_KEY)
                if result.succeeded:
                    st.success("Belge silindi.")
                else:
                    st.error("Belge silinemedi.")

    st.markdown("#### Belgeler")
    if not view.documents:
        st.info("Henüz bilgi tabanı belgesi yok.")
    for document in view.documents:
        st.write(document.filename)
        st.caption(
            f"{document.media_label} · {document.formatted_size} · "
            f"{document.readiness_label} · {document.created_at_utc}"
        )
        if st.button(
            "Silme onayı iste",
            key=f"document_delete_{document.action_token}",
            disabled=document.active,
        ):
            confirmation = resource.begin_delete(action_token=document.action_token)
            if confirmation is None:
                st.warning("İşlenen belge silinemez.")
            else:
                session_state[_CONFIRMATION_TOKEN_KEY] = confirmation.token


def _get(state: DocumentSessionState, key: str) -> object | None:
    return state[key] if key in state else None


def _setdefault(state: DocumentSessionState, key: str, default: object) -> object:
    if key not in state:
        state[key] = default
    return state[key]


def _discard(state: DocumentSessionState, key: str) -> None:
    if key in state:
        del state[key]
