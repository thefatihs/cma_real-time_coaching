from typing import Any

import pytest
from pydantic import ValidationError

from app.vector_store import (
    AtomicVectorBatchWriter,
    EmbeddingDistanceMetric,
    InMemoryVectorStore,
    KnowledgeBaseEmbeddingProfile,
    SearchRequest,
    SearchResult,
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
    VectorSearchHit,
    VectorStore,
)


def profile(**changes: object) -> KnowledgeBaseEmbeddingProfile:
    values: dict[str, object] = {
        "tenant_id": "tenant_alpha",
        "knowledge_base_id": "kb_support",
        "model_id": "synthetic-embedding-v1",
        "vector_dimension": 384,
        "normalize_embeddings": True,
        "distance_metric": EmbeddingDistanceMetric.COSINE,
    }
    values.update(changes)
    return KnowledgeBaseEmbeddingProfile.model_validate(values)


def test_exact_enum_members_and_values() -> None:
    assert list(EmbeddingDistanceMetric) == [
        EmbeddingDistanceMetric.COSINE,
        EmbeddingDistanceMetric.DOT_PRODUCT,
        EmbeddingDistanceMetric.EUCLIDEAN,
    ]
    assert [metric.value for metric in EmbeddingDistanceMetric] == [
        "cosine",
        "dot_product",
        "euclidean",
    ]


@pytest.mark.parametrize(
    "field",
    [
        "tenant_id",
        "knowledge_base_id",
        "model_id",
        "vector_dimension",
        "normalize_embeddings",
        "distance_metric",
    ],
)
def test_all_fields_are_required(field: str) -> None:
    values = profile().model_dump()
    del values[field]

    with pytest.raises(ValidationError, match="Field required"):
        KnowledgeBaseEmbeddingProfile.model_validate(values)


def test_model_is_frozen() -> None:
    value = profile()

    with pytest.raises(ValidationError):
        value.model_id = "changed"


def test_required_text_is_stripped_without_case_rewriting() -> None:
    value = profile(
        tenant_id=" Tenant_Alpha ",
        knowledge_base_id=" KB_Support ",
        model_id=" Model-V1 ",
    )

    assert value.tenant_id == "Tenant_Alpha"
    assert value.knowledge_base_id == "KB_Support"
    assert value.model_id == "Model-V1"


@pytest.mark.parametrize("field", ["tenant_id", "knowledge_base_id", "model_id"])
def test_blank_required_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} cannot be empty"):
        profile(**{field: " \n "})


def test_slash_containing_model_id_is_preserved_as_opaque_text() -> None:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    assert profile(model_id=model_id).model_id == model_id


def test_positive_dimension_is_accepted() -> None:
    assert profile(vector_dimension=1).vector_dimension == 1


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_dimension_is_rejected(value: int) -> None:
    with pytest.raises(ValidationError, match="positive"):
        profile(vector_dimension=value)


@pytest.mark.parametrize("value", [True, False, 384.0, "384"])
def test_non_integer_dimension_is_rejected(value: object) -> None:
    with pytest.raises(ValidationError, match="must be an integer"):
        profile(vector_dimension=value)


@pytest.mark.parametrize("value", [True, False])
def test_strict_normalize_embeddings_boolean_is_accepted(value: bool) -> None:
    assert profile(normalize_embeddings=value).normalize_embeddings is value


@pytest.mark.parametrize("value", [0, 1, "true", "false"])
def test_non_boolean_normalization_value_is_rejected(value: object) -> None:
    with pytest.raises(ValidationError, match="must be a boolean"):
        profile(normalize_embeddings=value)


@pytest.mark.parametrize("metric", list(EmbeddingDistanceMetric))
def test_each_metric_member_is_accepted(metric: EmbeddingDistanceMetric) -> None:
    value = profile(distance_metric=metric)

    assert value.distance_metric is metric


@pytest.mark.parametrize("metric", list(EmbeddingDistanceMetric))
def test_each_metric_string_value_is_parsed(metric: EmbeddingDistanceMetric) -> None:
    value = profile(distance_metric=metric.value)

    assert value.distance_metric is metric


def test_unsupported_metric_is_rejected() -> None:
    with pytest.raises(ValidationError):
        profile(distance_metric="manhattan")


def test_json_serialization_uses_exact_metric_value() -> None:
    value = profile(distance_metric=EmbeddingDistanceMetric.DOT_PRODUCT)

    assert value.model_dump(mode="json")["distance_metric"] == "dot_product"
    assert '"distance_metric":"dot_product"' in value.model_dump_json()


def test_normalized_models_are_equal_and_serialization_is_deterministic() -> None:
    first = profile(
        tenant_id=" tenant_alpha ",
        knowledge_base_id=" kb_support ",
        model_id=" synthetic-embedding-v1 ",
    )
    second = profile()

    assert first == second
    assert first.model_dump() == first.model_dump() == second.model_dump()
    assert (
        first.model_dump_json() == first.model_dump_json() == second.model_dump_json()
    )


def test_serialized_fields_are_exact_and_contain_no_operational_configuration() -> None:
    value = profile()
    expected_fields = {
        "tenant_id",
        "knowledge_base_id",
        "model_id",
        "vector_dimension",
        "normalize_embeddings",
        "distance_metric",
    }

    assert set(value.model_dump()) == expected_fields
    forbidden_fragments = (
        "database",
        "credential",
        "device",
        "endpoint",
        "password",
        "path",
        "provider",
    )
    assert not any(
        fragment in field
        for field in value.model_dump()
        for fragment in forbidden_fragments
    )


def test_existing_vector_store_exports_remain_available() -> None:
    exports: tuple[Any, ...] = (
        AtomicVectorBatchWriter,
        InMemoryVectorStore,
        SearchRequest,
        SearchResult,
        VectorBatchWriteRequest,
        VectorBatchWriteResult,
        VectorRecord,
        VectorRecordIdentity,
        VectorSearchHit,
        VectorStore,
    )

    assert all(export is not None for export in exports)
