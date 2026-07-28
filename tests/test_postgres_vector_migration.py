"""Static contract tests for the first PostgreSQL vector-store migration."""

import re
from pathlib import Path

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "migrations"
    / "postgres"
    / "0001_vector_store.sql"
)


def _migration_bytes() -> bytes:
    return MIGRATION_PATH.read_bytes()


def _migration() -> str:
    return _migration_bytes().decode("utf-8")


def _normalized_sql() -> str:
    return re.sub(r"\s+", " ", _migration()).strip()


def _statements() -> tuple[str, ...]:
    return tuple(
        statement.strip() for statement in _migration().split(";") if statement.strip()
    )


def test_migration_exists_at_exact_repository_path() -> None:
    assert MIGRATION_PATH.relative_to(
        Path(__file__).resolve().parents[1]
    ).as_posix() == ("migrations/postgres/0001_vector_store.sql")
    assert MIGRATION_PATH.is_file()


def test_migration_content_is_deterministic_utf8() -> None:
    content = _migration()

    assert content.encode("utf-8") == _migration_bytes()
    assert content == _migration()


def test_transaction_boundaries_are_first_and_last_statements() -> None:
    statements = _statements()

    assert statements[0].upper() == "BEGIN"
    assert statements[-1].upper() == "COMMIT"
    assert sum(statement.upper() == "BEGIN" for statement in statements) == 1
    assert sum(statement.upper() == "COMMIT" for statement in statements) == 1


def test_vector_extension_and_fixed_schema_are_created() -> None:
    sql = _normalized_sql()

    assert "CREATE EXTENSION IF NOT EXISTS vector;" in _migration()
    assert "CREATE SCHEMA callmetric_vector;" in _migration()
    assert sql.count("CREATE SCHEMA callmetric_vector") == 1


def test_migration_ledger_has_exact_required_contract() -> None:
    sql = _normalized_sql()

    assert "CREATE TABLE callmetric_vector.schema_migrations (" in sql
    assert "version text PRIMARY KEY" in sql
    assert "applied_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP" in sql
    assert (
        "INSERT INTO callmetric_vector.schema_migrations (version) VALUES ('0001')"
        in sql
    )


def test_embedding_profile_has_exact_required_columns() -> None:
    sql = _normalized_sql()

    for declaration in (
        "tenant_id text NOT NULL",
        "knowledge_base_id text NOT NULL",
        "model_id text NOT NULL",
        "vector_dimension integer NOT NULL",
        "normalize_embeddings boolean NOT NULL",
        "distance_metric text NOT NULL",
    ):
        assert declaration in sql


def test_embedding_profile_keys_and_dimension_are_constrained() -> None:
    sql = _normalized_sql()

    assert "PRIMARY KEY (tenant_id, knowledge_base_id)" in sql
    assert "UNIQUE (tenant_id, knowledge_base_id, vector_dimension)" in sql
    assert "CHECK (vector_dimension > 0)" in sql


def test_embedding_profile_is_cosine_only() -> None:
    assert "CHECK (distance_metric = 'cosine')" in _normalized_sql()


def test_embedding_profile_required_text_is_nonblank() -> None:
    sql = _normalized_sql()

    for column in ("tenant_id", "knowledge_base_id", "model_id"):
        assert f"CHECK (btrim({column}) <> '')" in sql


def test_vector_record_has_exact_required_columns() -> None:
    sql = _normalized_sql()

    for declaration in (
        "tenant_id text NOT NULL",
        "knowledge_base_id text NOT NULL",
        "document_id text NOT NULL",
        "chunk_id text NOT NULL",
        "text text NOT NULL",
        "vector_dimension integer NOT NULL",
        "embedding vector NOT NULL",
        "metadata_json jsonb NOT NULL",
    ):
        assert declaration in sql


def test_vector_record_has_complete_identity_primary_key() -> None:
    sql = _normalized_sql()

    assert "PRIMARY KEY ( tenant_id, knowledge_base_id, document_id, chunk_id )" in sql


def test_vector_record_has_restrictive_dimension_bearing_foreign_key() -> None:
    sql = _normalized_sql()

    assert (
        "FOREIGN KEY ( tenant_id, knowledge_base_id, vector_dimension ) "
        "REFERENCES callmetric_vector.embedding_profiles "
        "( tenant_id, knowledge_base_id, vector_dimension )" in sql
    )
    assert "ON UPDATE RESTRICT ON DELETE RESTRICT" in sql


def test_vector_record_dimension_and_nonzero_checks_are_present() -> None:
    sql = _normalized_sql()

    assert "CHECK (vector_dimension > 0)" in sql
    assert "CHECK (vector_dims(embedding) = vector_dimension)" in sql
    assert "CHECK (vector_norm(embedding) > 0)" in sql


def test_ordered_metadata_is_stored_as_jsonb_array() -> None:
    sql = _normalized_sql()

    assert "metadata_json jsonb NOT NULL" in sql
    assert "CHECK (jsonb_typeof(metadata_json) = 'array')" in sql


def test_vector_record_required_text_is_nonblank() -> None:
    sql = _normalized_sql()

    for column in (
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "text",
    ):
        assert f"CHECK (btrim({column}) <> '')" in sql


def test_embedding_uses_unbounded_vector_without_ann_indexes() -> None:
    sql = _migration()

    assert re.search(r"\bembedding\s+vector\s+NOT\s+NULL\b", sql, re.IGNORECASE)
    assert not re.search(r"\bembedding\s+vector\s*\(", sql, re.IGNORECASE)
    assert not re.search(r"\b(?:hnsw|ivfflat)\b", sql, re.IGNORECASE)
    assert not re.search(r"\bCREATE\s+INDEX\b", sql, re.IGNORECASE)


def test_extension_is_only_if_not_exists_usage() -> None:
    sql = _migration()

    assert len(re.findall(r"\bIF\s+NOT\s+EXISTS\b", sql, re.IGNORECASE)) == 1
    assert re.search(
        r"\bCREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+vector\b",
        sql,
        re.IGNORECASE,
    )
    assert not re.search(
        r"\bCREATE\s+(?:SCHEMA|TABLE)\s+IF\s+NOT\s+EXISTS\b",
        sql,
        re.IGNORECASE,
    )


def test_migration_has_no_destructive_or_dynamic_sql() -> None:
    sql = _migration()

    for forbidden in (
        r"\bROLLBACK\b",
        r"\bDROP\b",
        r"\bTRUNCATE\b",
        r"\bDELETE\s+FROM\b",
        r"\bEXECUTE\b",
        r"\bFORMAT\s*\(",
        r"\bDO\s+\$",
    ):
        assert not re.search(forbidden, sql, re.IGNORECASE)


def test_migration_contains_no_credentials_or_connection_strings() -> None:
    sql = _migration()

    for forbidden in (
        "password",
        "username",
        "user=",
        "host=",
        "port=",
        "dbname=",
        "postgresql://",
        "postgres://",
        "dsn",
    ):
        assert forbidden not in sql.lower()


def test_migration_contains_no_conflict_markers() -> None:
    marker_characters = ("<", "=", ">")

    assert all(
        not line.startswith(character * 7)
        for line in _migration().splitlines()
        for character in marker_characters
    )


def test_static_test_module_has_no_database_execution_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")

    for forbidden in ("psycopg", "sqlalchemy", "subprocess", "socket"):
        assert f"import {forbidden}" not in source
