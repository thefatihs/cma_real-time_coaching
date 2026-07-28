"""Profile-bound PostgreSQL atomic vector batch admission."""

from app.vector_store.embedding_profile import (
    EmbeddingDistanceMetric,
    KnowledgeBaseEmbeddingProfile,
)
from app.vector_store.models import (
    VectorBatchWriteRequest,
    VectorBatchWriteResult,
    VectorRecord,
    VectorRecordIdentity,
)
from app.vector_store.postgres.codecs import (
    canonicalize_float32_embedding,
    decode_ordered_metadata,
    encode_ordered_metadata,
)
from app.vector_store.postgres.contracts import (
    PostgreSQLStoredVectorRow,
    PostgreSQLVectorTransaction,
    PostgreSQLVectorTransactionRunner,
)

ProfileSignature = tuple[str, str, str, int, bool, EmbeddingDistanceMetric]
RecordIdentityKey = tuple[str, str]


class ProfileBoundPostgreSQLVectorStore:
    """Admit immutable vector batches against one pre-registered profile."""

    def __init__(
        self,
        *,
        expected_profile: KnowledgeBaseEmbeddingProfile,
        transaction_runner: PostgreSQLVectorTransactionRunner,
    ) -> None:
        profile_signature = _profile_signature(expected_profile)
        if expected_profile.distance_metric is not EmbeddingDistanceMetric.COSINE:
            raise ValueError("expected_profile distance_metric must be cosine")
        runner = getattr(transaction_runner, "run_in_transaction", None)
        if not callable(runner):
            raise ValueError("transaction_runner.run_in_transaction must be callable")
        self._expected_profile = expected_profile
        self._expected_profile_signature = profile_signature
        self._transaction_runner = transaction_runner

    def admit_batch(
        self,
        request: VectorBatchWriteRequest,
    ) -> VectorBatchWriteResult:
        candidates = tuple(
            sorted(
                _canonical_batch_rows(
                    request,
                    expected_profile=self._expected_profile,
                ),
                key=_row_identity,
            )
        )
        identities = tuple(
            VectorRecordIdentity(
                document_id=row.document_id,
                chunk_id=row.chunk_id,
            )
            for row in candidates
        )

        def admit(
            transaction: PostgreSQLVectorTransaction,
        ) -> VectorBatchWriteResult:
            tenant_id = self._expected_profile.tenant_id
            knowledge_base_id = self._expected_profile.knowledge_base_id
            transaction.acquire_scope_lock(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
            stored_profile = transaction.get_profile(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                for_update=True,
            )
            if stored_profile is None:
                raise ValueError("embedding profile is not registered")
            if _profile_signature(stored_profile) != self._expected_profile_signature:
                raise ValueError(
                    "stored embedding profile conflicts with expected profile"
                )

            stored_rows = transaction.get_records(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                identities=identities,
            )
            stored_by_identity = _validated_stored_rows(
                stored_rows,
                requested_identities={
                    (identity.document_id, identity.chunk_id) for identity in identities
                },
                expected_profile=self._expected_profile,
            )

            inserted: list[VectorRecordIdentity] = []
            unchanged: list[VectorRecordIdentity] = []
            rows_to_insert: list[PostgreSQLStoredVectorRow] = []
            for candidate, identity in zip(candidates, identities, strict=True):
                key = (identity.document_id, identity.chunk_id)
                stored = stored_by_identity.get(key)
                if stored is None:
                    inserted.append(identity)
                    rows_to_insert.append(candidate)
                elif _rows_are_equal(stored, candidate):
                    unchanged.append(identity)
                else:
                    raise ValueError(
                        "existing PostgreSQL vector record conflicts with batch record"
                    )

            rows_to_insert.sort(key=_row_identity)
            inserted.sort(key=_identity_key)
            unchanged.sort(key=_identity_key)
            if rows_to_insert:
                transaction.insert_records(tuple(rows_to_insert))
            return VectorBatchWriteResult(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                inserted_identities=tuple(inserted),
                unchanged_identities=tuple(unchanged),
            )

        return self._transaction_runner.run_in_transaction(admit)


def _canonical_batch_rows(
    request: object,
    *,
    expected_profile: KnowledgeBaseEmbeddingProfile,
) -> tuple[PostgreSQLStoredVectorRow, ...]:
    if not isinstance(request, VectorBatchWriteRequest):
        raise ValueError("request must be a VectorBatchWriteRequest")
    try:
        tenant_id = _canonical_text(request.tenant_id, "request tenant_id")
        knowledge_base_id = _canonical_text(
            request.knowledge_base_id,
            "request knowledge_base_id",
        )
        records = request.records
    except AttributeError as error:
        raise ValueError("request is malformed") from error
    if tenant_id != expected_profile.tenant_id:
        raise ValueError("request tenant_id does not match expected profile")
    if knowledge_base_id != expected_profile.knowledge_base_id:
        raise ValueError("request knowledge_base_id does not match expected profile")
    if not isinstance(records, tuple) or not records:
        raise ValueError("request records must be a non-empty tuple")

    rows: list[PostgreSQLStoredVectorRow] = []
    identities: set[RecordIdentityKey] = set()
    for record in records:
        row = _canonical_record_row(record, expected_profile=expected_profile)
        identity = _row_identity(row)
        if identity in identities:
            raise ValueError("vector batch record identities must be unique")
        identities.add(identity)
        rows.append(row)
    return tuple(rows)


def _canonical_record_row(
    record: object,
    *,
    expected_profile: KnowledgeBaseEmbeddingProfile,
) -> PostgreSQLStoredVectorRow:
    if not isinstance(record, VectorRecord):
        raise ValueError("batch records must be VectorRecord instances")
    try:
        tenant_id = _canonical_text(record.tenant_id, "record tenant_id")
        knowledge_base_id = _canonical_text(
            record.knowledge_base_id,
            "record knowledge_base_id",
        )
        document_id = _canonical_text(record.document_id, "record document_id")
        chunk_id = _canonical_text(record.chunk_id, "record chunk_id")
        text = _canonical_text(record.text, "record text")
        raw_embedding = record.embedding
        metadata = record.metadata
    except AttributeError as error:
        raise ValueError("vector record is malformed") from error
    if tenant_id != expected_profile.tenant_id:
        raise ValueError("vector record tenant_id does not match expected profile")
    if knowledge_base_id != expected_profile.knowledge_base_id:
        raise ValueError(
            "vector record knowledge_base_id does not match expected profile"
        )
    embedding = canonicalize_float32_embedding(
        raw_embedding,
        expected_dimension=expected_profile.vector_dimension,
    )
    metadata_json = encode_ordered_metadata(metadata)
    return PostgreSQLStoredVectorRow(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        metadata_json=metadata_json,
    )


def _validated_stored_rows(
    rows: object,
    *,
    requested_identities: set[RecordIdentityKey],
    expected_profile: KnowledgeBaseEmbeddingProfile,
) -> dict[RecordIdentityKey, PostgreSQLStoredVectorRow]:
    if not isinstance(rows, tuple):
        raise ValueError("stored vector rows must be a tuple")
    validated: dict[RecordIdentityKey, PostgreSQLStoredVectorRow] = {}
    for row in rows:
        if not isinstance(row, PostgreSQLStoredVectorRow):
            raise ValueError("stored vector row is malformed")
        tenant_id = _canonical_text(row.tenant_id, "stored row tenant_id")
        knowledge_base_id = _canonical_text(
            row.knowledge_base_id,
            "stored row knowledge_base_id",
        )
        if tenant_id != expected_profile.tenant_id:
            raise ValueError("stored vector row tenant_id does not match scope")
        if knowledge_base_id != expected_profile.knowledge_base_id:
            raise ValueError("stored vector row knowledge_base_id does not match scope")
        document_id = _canonical_text(row.document_id, "stored row document_id")
        chunk_id = _canonical_text(row.chunk_id, "stored row chunk_id")
        _canonical_text(row.text, "stored row text")
        identity = (document_id, chunk_id)
        if identity not in requested_identities:
            raise ValueError("transaction returned an unexpected stored vector row")
        if identity in validated:
            raise ValueError("transaction returned a duplicate stored vector row")
        embedding = canonicalize_float32_embedding(
            row.embedding,
            expected_dimension=expected_profile.vector_dimension,
        )
        if (
            not isinstance(row.embedding, tuple)
            or any(type(value) is not float for value in row.embedding)
            or embedding != row.embedding
        ):
            raise ValueError("stored vector row embedding is not canonical float32")
        decode_ordered_metadata(row.metadata_json)
        validated[identity] = row
    return validated


def _rows_are_equal(
    stored: PostgreSQLStoredVectorRow,
    candidate: PostgreSQLStoredVectorRow,
) -> bool:
    return (
        stored.tenant_id == candidate.tenant_id
        and stored.knowledge_base_id == candidate.knowledge_base_id
        and stored.document_id == candidate.document_id
        and stored.chunk_id == candidate.chunk_id
        and stored.text == candidate.text
        and stored.embedding == candidate.embedding
        and decode_ordered_metadata(stored.metadata_json)
        == decode_ordered_metadata(candidate.metadata_json)
    )


def _profile_signature(value: object) -> ProfileSignature:
    if not isinstance(value, KnowledgeBaseEmbeddingProfile):
        raise ValueError("expected_profile must be a KnowledgeBaseEmbeddingProfile")
    try:
        tenant_id = _canonical_text(value.tenant_id, "profile tenant_id")
        knowledge_base_id = _canonical_text(
            value.knowledge_base_id,
            "profile knowledge_base_id",
        )
        model_id = _canonical_text(value.model_id, "profile model_id")
        vector_dimension = value.vector_dimension
        normalize_embeddings = value.normalize_embeddings
        distance_metric = value.distance_metric
    except AttributeError as error:
        raise ValueError("embedding profile is malformed") from error
    if type(vector_dimension) is not int or vector_dimension <= 0:
        raise ValueError("profile vector_dimension must be a positive integer")
    if type(normalize_embeddings) is not bool:
        raise ValueError("profile normalize_embeddings must be a boolean")
    if not isinstance(distance_metric, EmbeddingDistanceMetric):
        raise ValueError("profile distance_metric must be an EmbeddingDistanceMetric")
    return (
        tenant_id,
        knowledge_base_id,
        model_id,
        vector_dimension,
        normalize_embeddings,
        distance_metric,
    )


def _canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    if cleaned != value:
        raise ValueError(f"{field_name} must be canonical")
    return cleaned


def _identity_key(identity: VectorRecordIdentity) -> RecordIdentityKey:
    return (identity.document_id, identity.chunk_id)


def _row_identity(row: PostgreSQLStoredVectorRow) -> RecordIdentityKey:
    return (row.document_id, row.chunk_id)
