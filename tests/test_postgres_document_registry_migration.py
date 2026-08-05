"""Static contract checks for the immutable document-registry migration."""

from pathlib import Path


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "migrations"
    / "postgres"
    / "0002_document_registry.sql"
)
EPHEMERAL_MIGRATION_PATH = MIGRATION_PATH.with_name(
    "0003_ephemeral_document_sources.sql"
)


def _migration_sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_document_registry_defines_exact_scoped_ownership_and_idempotency() -> None:
    sql = _migration_sql()

    assert "CREATE TABLE callmetric_vector.documents" in sql
    assert "PRIMARY KEY (tenant_id, knowledge_base_id, document_id)" in sql
    assert "UNIQUE (tenant_id, knowledge_base_id, sha256_hex)" in sql
    assert "FOREIGN KEY (tenant_id, knowledge_base_id)" in sql
    assert "REFERENCES callmetric_vector.embedding_profiles" in sql
    assert "documents_sha256_lowercase_hex" in sql
    assert "sha256_hex ~ '^[0-9a-f]{64}$'" in sql
    assert "documents_original_filename_bounded_basename" in sql
    assert "documents_storage_object_key_server_owned" in sql


def test_document_registry_rejects_path_semantics_and_limits_media_types() -> None:
    sql = _migration_sql()

    assert "strpos(original_filename, '/') = 0" in sql
    assert "strpos(original_filename, chr(92)) = 0" in sql
    assert "strpos(original_filename, ':') = 0" in sql
    assert "original_filename NOT IN ('.', '..')" in sql
    assert "storage_object_key !~ '(^|/)\\.\\.(/|$)'" in sql
    assert "'application/pdf'" in sql
    assert "'text/plain'" in sql
    assert "'text/markdown'" in sql


def test_job_registry_defines_scoped_cascade_and_bounded_lifecycle() -> None:
    sql = _migration_sql()

    assert "CREATE TABLE callmetric_vector.document_ingestion_jobs" in sql
    assert "PRIMARY KEY (tenant_id, knowledge_base_id, job_id)" in sql
    assert "UNIQUE (tenant_id, knowledge_base_id, document_id)" in sql
    assert "FOREIGN KEY (tenant_id, knowledge_base_id, document_id)" in sql
    assert "REFERENCES callmetric_vector.documents" in sql
    assert "ON DELETE CASCADE" in sql
    assert (
        "state IN ('QUEUED', 'PROCESSING', 'SUCCEEDED', 'FAILED', 'CANCELLED')" in sql
    )
    assert "document_ingestion_jobs_phase_supported" in sql
    assert "CHECK (attempt_count BETWEEN 0 AND 10)" in sql
    assert "CHECK (total_chunks = 0 OR processed_chunks <= total_chunks)" in sql
    assert "document_ingestion_jobs_state_timestamps_consistent" in sql


def test_requested_indexes_and_additive_only_behavior_are_explicit() -> None:
    sql = _migration_sql()
    upper_sql = sql.upper()

    assert "documents_scope_created_document_index" in sql
    assert "created_at_utc DESC" in sql
    assert "document_ingestion_jobs_scope_state_updated_index" in sql
    assert (
        "tenant_id,\n        knowledge_base_id,\n        state,\n        updated_at_utc"
        in sql
    )
    assert "VALUES ('0002')" in sql
    assert "DELETE FROM" not in upper_sql
    assert "UPDATE CALLMETRIC_VECTOR.VECTOR_RECORDS" not in upper_sql
    assert "ALTER TABLE CALLMETRIC_VECTOR.VECTOR_RECORDS" not in upper_sql
    assert "REFERENCES CALLMETRIC_VECTOR.VECTOR_RECORDS" not in upper_sql


def test_ephemeral_source_migration_only_relaxes_source_key_nullability() -> None:
    sql = EPHEMERAL_MIGRATION_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    upper_sql = sql.upper()

    assert "ALTER TABLE callmetric_vector.documents" in sql
    assert "ALTER COLUMN storage_object_key DROP NOT NULL" in sql
    assert "VALUES ('0003')" in sql
    assert "DELETE FROM" not in upper_sql
    assert "UPDATE CALLMETRIC_VECTOR.DOCUMENTS" not in upper_sql
    assert "ALTER TABLE CALLMETRIC_VECTOR.VECTOR_RECORDS" not in upper_sql
