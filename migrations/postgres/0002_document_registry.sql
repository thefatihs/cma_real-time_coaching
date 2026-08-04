BEGIN;

CREATE TABLE callmetric_vector.documents (
    tenant_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    document_id text NOT NULL,
    original_filename text NOT NULL,
    media_type text NOT NULL,
    byte_size bigint NOT NULL,
    sha256_hex char(64) NOT NULL,
    storage_object_key text NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ready_at_utc timestamptz,
    CONSTRAINT documents_primary_key
        PRIMARY KEY (tenant_id, knowledge_base_id, document_id),
    CONSTRAINT documents_scope_sha256_unique
        UNIQUE (tenant_id, knowledge_base_id, sha256_hex),
    CONSTRAINT documents_profile_foreign_key
        FOREIGN KEY (tenant_id, knowledge_base_id)
        REFERENCES callmetric_vector.embedding_profiles (
            tenant_id,
            knowledge_base_id
        )
        ON UPDATE RESTRICT
        ON DELETE RESTRICT,
    CONSTRAINT documents_tenant_id_bounded
        CHECK (
            tenant_id = btrim(tenant_id)
            AND char_length(tenant_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT documents_knowledge_base_id_bounded
        CHECK (
            knowledge_base_id = btrim(knowledge_base_id)
            AND char_length(knowledge_base_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT documents_document_id_bounded
        CHECK (
            document_id = btrim(document_id)
            AND char_length(document_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT documents_original_filename_bounded_basename
        CHECK (
            original_filename = btrim(original_filename)
            AND char_length(original_filename) BETWEEN 1 AND 255
            AND original_filename NOT IN ('.', '..')
            AND strpos(original_filename, '/') = 0
            AND strpos(original_filename, chr(92)) = 0
            AND strpos(original_filename, ':') = 0
        ),
    CONSTRAINT documents_media_type_supported
        CHECK (
            media_type IN (
                'application/pdf',
                'text/plain',
                'text/markdown'
            )
        ),
    CONSTRAINT documents_byte_size_positive
        CHECK (byte_size > 0),
    CONSTRAINT documents_sha256_lowercase_hex
        CHECK (sha256_hex ~ '^[0-9a-f]{64}$'),
    CONSTRAINT documents_storage_object_key_server_owned
        CHECK (
            storage_object_key = btrim(storage_object_key)
            AND char_length(storage_object_key) BETWEEN 1 AND 512
            AND storage_object_key ~ '^[A-Za-z0-9][A-Za-z0-9/_-]*$'
            AND storage_object_key !~ '(^|/)\.\.(/|$)'
            AND storage_object_key !~ '//'
            AND right(storage_object_key, 1) <> '/'
        ),
    CONSTRAINT documents_ready_timestamp_ordered
        CHECK (ready_at_utc IS NULL OR ready_at_utc >= created_at_utc)
);

CREATE TABLE callmetric_vector.document_ingestion_jobs (
    tenant_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    job_id text NOT NULL,
    document_id text NOT NULL,
    state text NOT NULL,
    phase text NOT NULL,
    processed_chunks integer NOT NULL DEFAULT 0,
    total_chunks integer NOT NULL DEFAULT 0,
    attempt_count integer NOT NULL DEFAULT 0,
    created_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at_utc timestamptz,
    updated_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at_utc timestamptz,
    CONSTRAINT document_ingestion_jobs_primary_key
        PRIMARY KEY (tenant_id, knowledge_base_id, job_id),
    CONSTRAINT document_ingestion_jobs_document_unique
        UNIQUE (tenant_id, knowledge_base_id, document_id),
    CONSTRAINT document_ingestion_jobs_document_foreign_key
        FOREIGN KEY (tenant_id, knowledge_base_id, document_id)
        REFERENCES callmetric_vector.documents (
            tenant_id,
            knowledge_base_id,
            document_id
        )
        ON UPDATE RESTRICT
        ON DELETE CASCADE,
    CONSTRAINT document_ingestion_jobs_tenant_id_bounded
        CHECK (
            tenant_id = btrim(tenant_id)
            AND char_length(tenant_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT document_ingestion_jobs_knowledge_base_id_bounded
        CHECK (
            knowledge_base_id = btrim(knowledge_base_id)
            AND char_length(knowledge_base_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT document_ingestion_jobs_job_id_bounded
        CHECK (
            job_id = btrim(job_id)
            AND char_length(job_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT document_ingestion_jobs_document_id_bounded
        CHECK (
            document_id = btrim(document_id)
            AND char_length(document_id) BETWEEN 1 AND 255
        ),
    CONSTRAINT document_ingestion_jobs_state_supported
        CHECK (
            state IN ('QUEUED', 'PROCESSING', 'SUCCEEDED', 'FAILED', 'CANCELLED')
        ),
    CONSTRAINT document_ingestion_jobs_phase_supported
        CHECK (
            phase IN (
                'VALIDATION',
                'STORAGE',
                'EXTRACTION',
                'CHUNKING',
                'EMBEDDING',
                'VECTOR_WRITE',
                'FINALIZE'
            )
        ),
    CONSTRAINT document_ingestion_jobs_processed_chunks_nonnegative
        CHECK (processed_chunks >= 0),
    CONSTRAINT document_ingestion_jobs_total_chunks_nonnegative
        CHECK (total_chunks >= 0),
    CONSTRAINT document_ingestion_jobs_chunk_progress_bounded
        CHECK (total_chunks = 0 OR processed_chunks <= total_chunks),
    CONSTRAINT document_ingestion_jobs_attempt_count_bounded
        CHECK (attempt_count BETWEEN 0 AND 10),
    CONSTRAINT document_ingestion_jobs_updated_timestamp_ordered
        CHECK (
            updated_at_utc >= created_at_utc
            AND updated_at_utc >= COALESCE(started_at_utc, created_at_utc)
        ),
    CONSTRAINT document_ingestion_jobs_started_timestamp_ordered
        CHECK (started_at_utc IS NULL OR started_at_utc >= created_at_utc),
    CONSTRAINT document_ingestion_jobs_finished_timestamp_ordered
        CHECK (
            finished_at_utc IS NULL
            OR (
                finished_at_utc >= COALESCE(started_at_utc, created_at_utc)
                AND finished_at_utc >= updated_at_utc
            )
        ),
    CONSTRAINT document_ingestion_jobs_state_timestamps_consistent
        CHECK (
            (state = 'QUEUED' AND started_at_utc IS NULL AND finished_at_utc IS NULL)
            OR (
                state = 'PROCESSING'
                AND started_at_utc IS NOT NULL
                AND finished_at_utc IS NULL
            )
            OR (
                state IN ('SUCCEEDED', 'FAILED', 'CANCELLED')
                AND finished_at_utc IS NOT NULL
            )
        ),
    CONSTRAINT document_ingestion_jobs_success_consistent
        CHECK (
            state <> 'SUCCEEDED'
            OR (
                phase = 'FINALIZE'
                AND started_at_utc IS NOT NULL
                AND total_chunks > 0
                AND processed_chunks = total_chunks
            )
        )
);

CREATE INDEX documents_scope_created_document_index
    ON callmetric_vector.documents (
        tenant_id,
        knowledge_base_id,
        created_at_utc DESC,
        document_id
    );

CREATE INDEX document_ingestion_jobs_scope_state_updated_index
    ON callmetric_vector.document_ingestion_jobs (
        tenant_id,
        knowledge_base_id,
        state,
        updated_at_utc
    );

INSERT INTO callmetric_vector.schema_migrations (version)
VALUES ('0002');

COMMIT;
