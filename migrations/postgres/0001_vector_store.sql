BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA callmetric_vector;

CREATE TABLE callmetric_vector.schema_migrations (
    version text PRIMARY KEY,
    applied_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE callmetric_vector.embedding_profiles (
    tenant_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    model_id text NOT NULL,
    vector_dimension integer NOT NULL,
    normalize_embeddings boolean NOT NULL,
    distance_metric text NOT NULL,
    CONSTRAINT embedding_profiles_primary_key
        PRIMARY KEY (tenant_id, knowledge_base_id),
    CONSTRAINT embedding_profiles_scope_dimension_unique
        UNIQUE (tenant_id, knowledge_base_id, vector_dimension),
    CONSTRAINT embedding_profiles_vector_dimension_positive
        CHECK (vector_dimension > 0),
    CONSTRAINT embedding_profiles_cosine_only
        CHECK (distance_metric = 'cosine'),
    CONSTRAINT embedding_profiles_tenant_id_nonblank
        CHECK (btrim(tenant_id) <> ''),
    CONSTRAINT embedding_profiles_knowledge_base_id_nonblank
        CHECK (btrim(knowledge_base_id) <> ''),
    CONSTRAINT embedding_profiles_model_id_nonblank
        CHECK (btrim(model_id) <> '')
);

CREATE TABLE callmetric_vector.vector_records (
    tenant_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    document_id text NOT NULL,
    chunk_id text NOT NULL,
    text text NOT NULL,
    vector_dimension integer NOT NULL,
    embedding vector NOT NULL,
    metadata_json jsonb NOT NULL,
    CONSTRAINT vector_records_primary_key
        PRIMARY KEY (
            tenant_id,
            knowledge_base_id,
            document_id,
            chunk_id
        ),
    CONSTRAINT vector_records_profile_foreign_key
        FOREIGN KEY (
            tenant_id,
            knowledge_base_id,
            vector_dimension
        )
        REFERENCES callmetric_vector.embedding_profiles (
            tenant_id,
            knowledge_base_id,
            vector_dimension
        )
        ON UPDATE RESTRICT
        ON DELETE RESTRICT,
    CONSTRAINT vector_records_vector_dimension_positive
        CHECK (vector_dimension > 0),
    CONSTRAINT vector_records_embedding_dimension_matches
        CHECK (vector_dims(embedding) = vector_dimension),
    CONSTRAINT vector_records_embedding_nonzero
        CHECK (vector_norm(embedding) > 0),
    CONSTRAINT vector_records_metadata_is_array
        CHECK (jsonb_typeof(metadata_json) = 'array'),
    CONSTRAINT vector_records_tenant_id_nonblank
        CHECK (btrim(tenant_id) <> ''),
    CONSTRAINT vector_records_knowledge_base_id_nonblank
        CHECK (btrim(knowledge_base_id) <> ''),
    CONSTRAINT vector_records_document_id_nonblank
        CHECK (btrim(document_id) <> ''),
    CONSTRAINT vector_records_chunk_id_nonblank
        CHECK (btrim(chunk_id) <> ''),
    CONSTRAINT vector_records_text_nonblank
        CHECK (btrim(text) <> '')
);

INSERT INTO callmetric_vector.schema_migrations (version)
VALUES ('0001');

COMMIT;
