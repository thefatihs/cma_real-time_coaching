"""Read-only readiness verification for the fixed PostgreSQL vector schema."""

from collections.abc import Callable
from typing import Any, NoReturn

from psycopg import Connection

PostgreSQLReadinessConnectionFactory = Callable[[], Connection[Any]]

_SCHEMA_NAME = "callmetric_vector"
_EXTENSION_NAME = "vector"
_REQUIRED_EXTENSION_VERSION = "0.8.5"
_REQUIRED_MIGRATION_VERSIONS = ("0001", "0002", "0003")
_REQUIRED_MIGRATION_VERSION = _REQUIRED_MIGRATION_VERSIONS[-1]
_REQUIRED_TABLES = (
    "document_ingestion_jobs",
    "documents",
    "embedding_profiles",
    "schema_migrations",
    "vector_records",
)
_REQUIRED_COLUMNS = {
    "embedding_profiles": frozenset(
        {
            "tenant_id",
            "knowledge_base_id",
            "model_id",
            "vector_dimension",
            "normalize_embeddings",
            "distance_metric",
        }
    ),
    "documents": frozenset(
        {
            "tenant_id",
            "knowledge_base_id",
            "document_id",
            "original_filename",
            "media_type",
            "byte_size",
            "sha256_hex",
            "storage_object_key",
            "created_at_utc",
            "ready_at_utc",
        }
    ),
    "document_ingestion_jobs": frozenset(
        {
            "tenant_id",
            "knowledge_base_id",
            "job_id",
            "document_id",
            "state",
            "phase",
            "processed_chunks",
            "total_chunks",
            "attempt_count",
            "created_at_utc",
            "started_at_utc",
            "updated_at_utc",
            "finished_at_utc",
        }
    ),
    "vector_records": frozenset(
        {
            "tenant_id",
            "knowledge_base_id",
            "document_id",
            "chunk_id",
            "text",
            "vector_dimension",
            "embedding",
            "metadata_json",
        }
    ),
}
_REQUIRED_CONSTRAINTS = {
    "documents": frozenset(
        {
            "documents_primary_key",
            "documents_scope_sha256_unique",
            "documents_profile_foreign_key",
            "documents_tenant_id_bounded",
            "documents_knowledge_base_id_bounded",
            "documents_document_id_bounded",
            "documents_original_filename_bounded_basename",
            "documents_media_type_supported",
            "documents_byte_size_positive",
            "documents_sha256_lowercase_hex",
            "documents_storage_object_key_server_owned",
            "documents_ready_timestamp_ordered",
        }
    ),
    "document_ingestion_jobs": frozenset(
        {
            "document_ingestion_jobs_primary_key",
            "document_ingestion_jobs_document_unique",
            "document_ingestion_jobs_document_foreign_key",
            "document_ingestion_jobs_tenant_id_bounded",
            "document_ingestion_jobs_knowledge_base_id_bounded",
            "document_ingestion_jobs_job_id_bounded",
            "document_ingestion_jobs_document_id_bounded",
            "document_ingestion_jobs_state_supported",
            "document_ingestion_jobs_phase_supported",
            "document_ingestion_jobs_processed_chunks_nonnegative",
            "document_ingestion_jobs_total_chunks_nonnegative",
            "document_ingestion_jobs_chunk_progress_bounded",
            "document_ingestion_jobs_attempt_count_bounded",
            "document_ingestion_jobs_updated_timestamp_ordered",
            "document_ingestion_jobs_started_timestamp_ordered",
            "document_ingestion_jobs_finished_timestamp_ordered",
            "document_ingestion_jobs_state_timestamps_consistent",
            "document_ingestion_jobs_success_consistent",
        }
    ),
    "embedding_profiles": frozenset(
        {
            "embedding_profiles_primary_key",
            "embedding_profiles_scope_dimension_unique",
            "embedding_profiles_vector_dimension_positive",
            "embedding_profiles_cosine_only",
            "embedding_profiles_tenant_id_nonblank",
            "embedding_profiles_knowledge_base_id_nonblank",
            "embedding_profiles_model_id_nonblank",
        }
    ),
    "vector_records": frozenset(
        {
            "vector_records_primary_key",
            "vector_records_profile_foreign_key",
            "vector_records_vector_dimension_positive",
            "vector_records_embedding_dimension_matches",
            "vector_records_embedding_nonzero",
            "vector_records_metadata_is_array",
            "vector_records_tenant_id_nonblank",
            "vector_records_knowledge_base_id_nonblank",
            "vector_records_document_id_nonblank",
            "vector_records_chunk_id_nonblank",
            "vector_records_text_nonblank",
        }
    ),
}
_REQUIRED_INDEXES = {
    "documents": frozenset({"documents_scope_created_document_index"}),
    "document_ingestion_jobs": frozenset(
        {"document_ingestion_jobs_scope_state_updated_index"}
    ),
}

_READ_ONLY_SQL = "SET TRANSACTION READ ONLY"
_USABILITY_SQL = "SELECT 1"
_EXTENSION_SQL = """
    SELECT extversion
    FROM pg_catalog.pg_extension
    WHERE extname = %s
    """
_SCHEMA_SQL = """
    SELECT nspname
    FROM pg_catalog.pg_namespace
    WHERE nspname = %s
    """
_TABLES_SQL = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = %s AND tablename = ANY(%s)
    ORDER BY tablename
    """
_MIGRATION_SQL = """
    SELECT version
    FROM callmetric_vector.schema_migrations
    ORDER BY version
    """
_COLUMNS_SQL = """
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = ANY(%s)
    ORDER BY table_name, ordinal_position
    """
_NULLABILITY_SQL = """
    SELECT is_nullable
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s AND column_name = %s
    """
_CONSTRAINTS_SQL = """
    SELECT table_name, constraint_name
    FROM information_schema.table_constraints
    WHERE table_schema = %s AND table_name = ANY(%s)
    ORDER BY table_name, constraint_name
    """
_INDEXES_SQL = """
    SELECT tablename, indexname
    FROM pg_catalog.pg_indexes
    WHERE schemaname = %s AND tablename = ANY(%s)
    ORDER BY tablename, indexname
    """


class PostgreSQLSchemaReadinessChecker:
    """Verify the repository-owned PostgreSQL schema without mutating it."""

    def __init__(
        self,
        *,
        connection_factory: PostgreSQLReadinessConnectionFactory,
    ) -> None:
        if not callable(connection_factory):
            raise ValueError("connection_factory must be callable")
        self._connection_factory = connection_factory

    def verify(self) -> None:
        connection = self._connection_factory()
        try:
            if connection.autocommit is not False:
                raise ValueError("connection.autocommit must be exactly False")
            with connection.cursor() as cursor:
                cursor.execute(_READ_ONLY_SQL)
                _execute_and_expect_exact_rows(
                    cursor,
                    _USABILITY_SQL,
                    expected=((1,),),
                    check_name="connection usability",
                )
                _execute_and_expect_exact_rows(
                    cursor,
                    _EXTENSION_SQL,
                    (_EXTENSION_NAME,),
                    expected=((_REQUIRED_EXTENSION_VERSION,),),
                    check_name="pgvector extension",
                )
                _execute_and_expect_exact_rows(
                    cursor,
                    _SCHEMA_SQL,
                    (_SCHEMA_NAME,),
                    expected=((_SCHEMA_NAME,),),
                    check_name="PostgreSQL schema",
                )
                _execute_and_expect_exact_rows(
                    cursor,
                    _TABLES_SQL,
                    (_SCHEMA_NAME, list(_REQUIRED_TABLES)),
                    expected=tuple((name,) for name in sorted(_REQUIRED_TABLES)),
                    check_name="PostgreSQL tables",
                )
                _execute_and_expect_exact_rows(
                    cursor,
                    _MIGRATION_SQL,
                    expected=tuple(
                        (version,) for version in _REQUIRED_MIGRATION_VERSIONS
                    ),
                    check_name="migration ledger",
                )
                cursor.execute(
                    _COLUMNS_SQL,
                    (_SCHEMA_NAME, list(_REQUIRED_COLUMNS)),
                )
                _validate_required_members(
                    cursor.fetchall(),
                    required=_REQUIRED_COLUMNS,
                    check_name="PostgreSQL columns",
                )
                _execute_and_expect_exact_rows(
                    cursor,
                    _NULLABILITY_SQL,
                    (_SCHEMA_NAME, "documents", "storage_object_key"),
                    expected=(("YES",),),
                    check_name="document source nullability",
                )
                cursor.execute(
                    _CONSTRAINTS_SQL,
                    (_SCHEMA_NAME, list(_REQUIRED_CONSTRAINTS)),
                )
                _validate_required_members(
                    cursor.fetchall(),
                    required=_REQUIRED_CONSTRAINTS,
                    check_name="PostgreSQL constraints",
                )
                cursor.execute(
                    _INDEXES_SQL,
                    (_SCHEMA_NAME, list(_REQUIRED_INDEXES)),
                )
                _validate_required_members(
                    cursor.fetchall(),
                    required=_REQUIRED_INDEXES,
                    check_name="PostgreSQL indexes",
                )
        except BaseException as primary:
            _raise_after_cleanup(connection, primary)

        try:
            connection.rollback()
        except BaseException as primary:
            _raise_after_close(connection, primary)
        connection.close()


def _execute_and_expect_exact_rows(
    cursor: Any,
    query: str,
    parameters: tuple[object, ...] | None = None,
    *,
    expected: tuple[tuple[object, ...], ...],
    check_name: str,
) -> None:
    if parameters is None:
        cursor.execute(query)
    else:
        cursor.execute(query, parameters)
    rows = _validated_rows(cursor.fetchall(), arity=1, check_name=check_name)
    if rows != expected:
        raise ValueError(f"{check_name} is not ready")


def _validate_required_members(
    raw_rows: object,
    *,
    required: dict[str, frozenset[str]],
    check_name: str,
) -> None:
    rows = _validated_rows(raw_rows, arity=2, check_name=check_name)
    observed: dict[str, set[str]] = {name: set() for name in required}
    identities: set[tuple[str, str]] = set()
    for table_name, member_name in rows:
        if not isinstance(table_name, str) or not isinstance(member_name, str):
            raise ValueError(f"{check_name} returned a non-text value")
        if table_name not in required:
            raise ValueError(f"{check_name} returned an unexpected table")
        identity = (table_name, member_name)
        if identity in identities:
            raise ValueError(f"{check_name} returned a duplicate value")
        identities.add(identity)
        observed[table_name].add(member_name)
    if any(not names.issubset(observed[table]) for table, names in required.items()):
        raise ValueError(f"{check_name} are incomplete")


def _validated_rows(
    value: object,
    *,
    arity: int,
    check_name: str,
) -> tuple[tuple[object, ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{check_name} returned a malformed row collection")
    rows: list[tuple[object, ...]] = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != arity:
            raise ValueError(f"{check_name} returned a malformed row")
        if any(item is None for item in row):
            raise ValueError(f"{check_name} returned a null value")
        rows.append(row)
    return tuple(rows)


def _raise_after_cleanup(
    connection: Connection[Any],
    primary: BaseException,
) -> NoReturn:
    cleanup_failures: list[Exception] = []
    try:
        connection.rollback()
    except Exception as error:
        cleanup_failures.append(error)
    try:
        connection.close()
    except Exception as error:
        cleanup_failures.append(error)
    if cleanup_failures:
        raise primary from ExceptionGroup(
            "PostgreSQL readiness cleanup failed",
            cleanup_failures,
        )
    raise primary


def _raise_after_close(
    connection: Connection[Any],
    primary: BaseException,
) -> NoReturn:
    try:
        connection.close()
    except Exception as close_error:
        raise primary from ExceptionGroup(
            "PostgreSQL readiness cleanup failed",
            [close_error],
        )
    raise primary
