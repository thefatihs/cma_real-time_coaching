BEGIN;

ALTER TABLE callmetric_vector.documents
    ALTER COLUMN storage_object_key DROP NOT NULL;

INSERT INTO callmetric_vector.schema_migrations (version)
VALUES ('0003');

COMMIT;
