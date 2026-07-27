from concurrent.futures import ThreadPoolExecutor
from math import inf, nan
from threading import Lock

import pytest

from app.embeddings import (
    DocumentEmbedder,
    QueryEmbedder,
    SentenceTransformerBackend,
    SentenceTransformerQueryEmbedder,
    SentenceTransformerQueryEmbedderConfig,
)


class RecordingBackend:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = outputs
        self._lock = Lock()
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> object:
        with self._lock:
            self.calls.append((list(texts), normalize_embeddings))
            if not self._outputs:
                return tuple((0.25, 0.75) for _ in texts)
            return self._outputs.pop(0)


class RaisingBackend:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> object:
        self.calls += 1
        raise self.error


class FactorySpy:
    def __init__(self, backend: SentenceTransformerBackend) -> None:
        self.backend = backend
        self.calls: list[SentenceTransformerQueryEmbedderConfig] = []
        self._lock = Lock()

    def __call__(
        self,
        config: SentenceTransformerQueryEmbedderConfig,
    ) -> SentenceTransformerBackend:
        with self._lock:
            self.calls.append(config)
        return self.backend


def config(
    *,
    normalize_embeddings: bool = True,
) -> SentenceTransformerQueryEmbedderConfig:
    return SentenceTransformerQueryEmbedderConfig(
        expected_tenant_id="tenant_alpha",
        expected_knowledge_base_id="kb_support",
        model_name_or_path="local_models/synthetic-embedding",
        device="cpu",
        normalize_embeddings=normalize_embeddings,
        local_files_only=True,
    )


def provider(
    outputs: list[object] | None = None,
    *,
    normalize_embeddings: bool = True,
) -> tuple[SentenceTransformerQueryEmbedder, RecordingBackend, FactorySpy]:
    backend = RecordingBackend(outputs or [])
    factory = FactorySpy(backend)
    subject = SentenceTransformerQueryEmbedder(
        config(normalize_embeddings=normalize_embeddings),
        backend_factory=factory,
    )
    return subject, backend, factory


def embed_documents(
    subject: DocumentEmbedder,
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    texts: tuple[str, ...] = ("Synthetic first chunk.", "Synthetic second chunk."),
) -> tuple[tuple[float, ...], ...]:
    return subject.embed_documents(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        texts=texts,
    )


def test_structural_document_and_query_protocol_compatibility() -> None:
    concrete = SentenceTransformerQueryEmbedder(
        config(),
        backend_factory=FactorySpy(RecordingBackend([])),
    )

    document_embedder: DocumentEmbedder = concrete
    query_embedder: QueryEmbedder = concrete

    assert document_embedder is concrete
    assert query_embedder is concrete


def test_query_and_documents_share_one_lazy_backend_load() -> None:
    subject, backend, factory = provider(
        [
            ((0.25, 0.75),),
            ((0.75, 0.25), (0.5, 0.5)),
        ]
    )

    query = subject.embed_query(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        text="Synthetic query.",
    )
    documents = embed_documents(subject)

    assert query == (0.25, 0.75)
    assert documents == ((0.75, 0.25), (0.5, 0.5))
    assert factory.calls == [config()]
    assert len(backend.calls) == 2


def test_complete_batch_is_passed_in_one_call_with_exact_normalized_texts() -> None:
    subject, backend, _ = provider([((0.1, 0.9), (0.2, 0.8), (0.3, 0.7))])
    source = (
        "  Synthetic first chunk. ",
        "\tSynthetic second chunk.\n",
        "Synthetic third chunk.",
    )

    result = embed_documents(subject, texts=source)

    assert result == ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7))
    assert backend.calls == [
        (
            [
                "Synthetic first chunk.",
                "Synthetic second chunk.",
                "Synthetic third chunk.",
            ],
            True,
        )
    ]
    assert source == (
        "  Synthetic first chunk. ",
        "\tSynthetic second chunk.\n",
        "Synthetic third chunk.",
    )


def test_normalization_flag_is_forwarded_exactly() -> None:
    subject, backend, _ = provider(
        [((0.1, 0.9), (0.2, 0.8))],
        normalize_embeddings=False,
    )

    embed_documents(subject)

    assert backend.calls == [
        (
            ["Synthetic first chunk.", "Synthetic second chunk."],
            False,
        )
    ]


def test_backend_row_order_is_preserved() -> None:
    subject, _, _ = provider([((0.1, 0.9), (0.2, 0.8), (0.3, 0.7))])

    result = embed_documents(
        subject,
        texts=("Synthetic A.", "Synthetic B.", "Synthetic C."),
    )

    assert result == ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7))


def test_empty_batch_is_rejected_before_backend_loading() -> None:
    subject, backend, factory = provider()

    with pytest.raises(ValueError, match="texts cannot be empty"):
        embed_documents(subject, texts=())

    assert factory.calls == []
    assert backend.calls == []


def test_blank_text_is_rejected_before_backend_loading() -> None:
    subject, backend, factory = provider()

    with pytest.raises(ValueError, match=r"texts\[1\] cannot be empty"):
        embed_documents(subject, texts=("Synthetic valid chunk.", " "))

    assert factory.calls == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("tenant_id", "knowledge_base_id", "message"),
    [
        ("tenant_beta", "kb_support", "tenant_id"),
        ("tenant_alpha", "kb_other", "knowledge_base_id"),
        (" ", "kb_support", "tenant_id"),
        ("tenant_alpha", " ", "knowledge_base_id"),
    ],
)
def test_scope_is_rejected_before_backend_loading(
    tenant_id: str,
    knowledge_base_id: str,
    message: str,
) -> None:
    subject, backend, factory = provider()

    with pytest.raises(ValueError, match=message):
        embed_documents(
            subject,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )

    assert factory.calls == []
    assert backend.calls == []


@pytest.mark.parametrize(
    "output",
    [
        (),
        ((0.1, 0.9),),
        ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7)),
    ],
)
def test_wrong_output_row_count_is_rejected(output: object) -> None:
    subject, _, _ = provider([output])

    with pytest.raises(ValueError, match="row count"):
        embed_documents(subject)


@pytest.mark.parametrize(
    "output",
    [
        0.25,
        (0.25, 0.75),
        (0.25, (0.5, 0.5)),
        ((), (0.5, 0.5)),
        (((0.25, 0.75),), ((0.5, 0.5),)),
    ],
)
def test_scalar_nested_or_empty_output_is_rejected(output: object) -> None:
    subject, _, _ = provider([output])

    with pytest.raises(ValueError):
        embed_documents(subject)


@pytest.mark.parametrize(
    "value",
    [True, "0.25", object(), 1 + 2j, nan, inf, -inf],
)
def test_invalid_vector_values_are_rejected(value: object) -> None:
    subject, _, _ = provider([((value, 0.5), (0.25, 0.75))])

    with pytest.raises(ValueError):
        embed_documents(subject)


def test_inconsistent_batch_dimensions_are_rejected() -> None:
    subject, _, _ = provider([((0.1, 0.9), (0.2, 0.3, 0.5))])

    with pytest.raises(ValueError, match="equal dimensions"):
        embed_documents(subject)


def test_query_to_document_dimension_mismatch_is_rejected() -> None:
    subject, _, _ = provider(
        [
            ((0.25, 0.75),),
            ((0.2, 0.3, 0.5), (0.1, 0.4, 0.5)),
        ]
    )
    subject.embed_query(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        text="Synthetic query.",
    )

    with pytest.raises(ValueError, match="provider dimension"):
        embed_documents(subject)


def test_document_to_query_dimension_mismatch_is_rejected() -> None:
    subject, _, _ = provider(
        [
            ((0.2, 0.3, 0.5), (0.1, 0.4, 0.5)),
            ((0.25, 0.75),),
        ]
    )
    embed_documents(subject)

    with pytest.raises(ValueError, match="provider dimension"):
        subject.embed_query(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
            text="Synthetic query.",
        )


def test_failed_validation_does_not_establish_or_change_dimension() -> None:
    subject, _, _ = provider(
        [
            ((0.1, 0.9),),
            ((0.2, 0.3, 0.5), (0.1, 0.4, 0.5)),
            ((0.25, 0.75),),
            ((0.3, 0.3, 0.4),),
        ]
    )

    with pytest.raises(ValueError, match="row count"):
        embed_documents(subject)
    assert embed_documents(subject) == (
        (0.2, 0.3, 0.5),
        (0.1, 0.4, 0.5),
    )
    with pytest.raises(ValueError, match="provider dimension"):
        subject.embed_query(
            tenant_id="tenant_alpha",
            knowledge_base_id="kb_support",
            text="Synthetic mismatched query.",
        )
    assert subject.embed_query(
        tenant_id="tenant_alpha",
        knowledge_base_id="kb_support",
        text="Synthetic matching query.",
    ) == (0.3, 0.3, 0.4)


def test_repeated_document_calls_are_deterministic() -> None:
    subject, _, _ = provider(
        [
            ((0.1, 0.9), (0.2, 0.8)),
            ((0.1, 0.9), (0.2, 0.8)),
        ]
    )

    assert embed_documents(subject) == embed_documents(subject)


def test_query_and_document_calls_share_thread_safe_loading() -> None:
    subject, backend, factory = provider()

    def invoke(index: int) -> tuple[tuple[float, ...], ...]:
        if index % 2:
            return (
                subject.embed_query(
                    tenant_id="tenant_alpha",
                    knowledge_base_id="kb_support",
                    text="Synthetic concurrent query.",
                ),
            )
        return embed_documents(
            subject,
            texts=("Synthetic concurrent document.",),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(invoke, range(20)))

    assert results == (((0.25, 0.75),),) * 20
    assert factory.calls == [config()]
    assert len(backend.calls) == 20


def test_backend_exception_propagates_unchanged() -> None:
    error = RuntimeError("synthetic document embedding failure")
    backend = RaisingBackend(error)
    subject = SentenceTransformerQueryEmbedder(
        config(),
        backend_factory=FactorySpy(backend),
    )

    with pytest.raises(RuntimeError, match="document embedding failure"):
        embed_documents(subject)

    assert backend.calls == 1
