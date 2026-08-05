# PostgreSQL RAG Smoke Runbook

This runbook enables one synthetic dashboard tenant only when an explicit
server-side override is present. Normal dashboard defaults remain disabled.

## Safe committed examples

`docs/examples/dashboard-rag-smoke-tenant.json` contains only synthetic,
non-secret tenant activation values. Provider-settings and integration-policy
examples may contain tenant/knowledge-base policy and local model identity, but
must never contain connection or API credentials.

## Environment-only settings and secrets

Set the absolute path to the committed tenant example:

```powershell
$env:CALLMETRIC_DASHBOARD_SMOKE_TENANT_OVERRIDE_PATH = (
  Resolve-Path "docs/examples/dashboard-rag-smoke-tenant.json"
)
```

Set the existing strict dashboard activation variables to absolute,
server-controlled JSON paths:

```powershell
$env:CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH = "<ABSOLUTE_PROVIDER_JSON>"
$env:CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH = "<ABSOLUTE_POLICY_JSON>"
$env:CALLMETRIC_DASHBOARD_RAG_MAX_WORKERS = "1"
$env:CALLMETRIC_DASHBOARD_RAG_CAPACITY = "2"
```

PostgreSQL connection values remain exclusively in
`CALLMETRIC_POSTGRES_*`; migration credentials remain in
`CALLMETRIC_POSTGRES_MIGRATION_*`. vLLM endpoint, model and optional token
remain exclusively in `CALLMETRIC_VLLM_*`. Never commit their values.

## Windows-capable preparation

From a uv-managed Python 3.12 environment, apply the fixed migration, provision
the exact embedding profile, and ingest a small trusted synthetic TXT document
using the existing scripts. Pre-stage the embedding model under ignored
`local_artifacts/`; use `local_files_only=true`, a CPU device, normalization
matching the profile, and an actual 384-dimensional model output.

The canonical embedding identity remains
`sentence-transformers/all-MiniLM-L6-v2` in `model_id`, profile persistence and
compatibility checks. `model_name_or_path` may retain that canonical identifier
for a pre-populated offline Hugging Face cache, or it may be an operator-supplied
absolute snapshot directory beneath ignored `local_artifacts/`. The local path
option must identify immutable revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and match the deterministic
`minilm-1110a243.sha256` manifest. Validation rejects missing, changed,
symlink-escaping or non-canonical artifacts before model construction. Never
commit or display the machine-specific absolute path.

The model is not downloaded during dashboard startup.
`trust_remote_code` remains false. Deterministic fake embeddings validate
wiring only and do not constitute a real embedding smoke test.
The approved MiniLM model is English-oriented; this smoke verifies technical
384-dimensional ingestion and retrieval, not Turkish retrieval quality.

## Document-registry schema

The forward-only PostgreSQL migrator uses a fixed, ordered, digest-pinned
manifest. It applies `0001_vector_store.sql` before
`0002_document_registry.sql` and `0003_ephemeral_document_sources.sql`, records
all three versions in
`callmetric_vector.schema_migrations`, rejects altered migration content, and
performs no arbitrary migration-file discovery or downgrade execution.

Migration `0002` adds tenant- and knowledge-base-scoped `documents` and
`document_ingestion_jobs` tables. Document SHA-256 uniqueness is scoped to one
tenant and knowledge base. Jobs have fixed states and phases, bounded progress
and attempts, and are deleted automatically with their exact parent document.
The migration does not alter, delete, rewrite, or add a document foreign key to
legacy `vector_records`; the existing embedding profile and vector behavior is
unchanged. These tables provide persistence only. Dashboard document upload,
extraction, embedding workers, object storage, and document services are not
implemented by this migration.

Migration `0003` only makes `documents.storage_object_key` nullable so new
documents can follow the ephemeral-source policy. Existing non-null keys remain
valid legacy metadata; neither migration nor readiness rewrites them.

## Phase-one document preparation

The in-memory preparation layer accepts PDF, UTF-8 TXT and UTF-8 Markdown only.
Uploads are limited to 10 MiB, PDFs to 100 pages, and normalized extracted text
to 1,000,000 Unicode characters. Encrypted PDFs, OCR, archives, external URLs,
empty documents, invalid UTF-8, NUL content and filename path semantics are
rejected with fixed non-sensitive failures. PDF parsing uses lock-pinned
PyMuPDF from in-memory bytes; no client-controlled filesystem path is used.

Preparation calculates SHA-256 over accepted source bytes, normalizes text and
builds deterministic chunks with page metadata where applicable. It does not
create embeddings or perform storage operations.

## Synchronous registry and ingestion lifecycle

The document-registry repository accepts validated domain models only. Every
operation is scoped by trusted `tenant_id` and `knowledge_base_id`; document and
job IDs are server-owned inputs. New documents use a null storage object key.
Creation inserts
the document and its `QUEUED` job atomically. SHA-256 duplicates are resolved by
the database uniqueness constraint within that exact scope, without a
check-before-insert race. A ready duplicate returns its existing document/job
identity without embedding or rewriting vectors. A failed duplicate remains
failed until the bounded retry operation explicitly returns it to `QUEUED`.

Synchronous ingestion claims the queued job, embeds all prepared chunks outside
the final database transaction, and verifies the exact row count and registered
vector dimension. The final transaction locks the document/job, admits the full
vector batch through the caller-owned transaction, marks the job
`SUCCEEDED`/`FINALIZE`, and sets `ready_at_utc`. Any vector or finalization error
rolls back that complete transaction, then records only a fixed failure phase.
No exception details, digest, source text, filename, object key, or scoped IDs
are stored as failure text.

Exact deletion locks one scoped document, deletes only its tenant/knowledge-base
and document-matched vector rows, then deletes the registry row and lets the job
foreign key cascade. Fresh documents have no storage object to clean up. Legacy
non-null keys remain operator-owned metadata; dashboard deletion performs no
filesystem operation. Embedding profiles and unrelated legacy vectors are
untouched.

The Streamlit dashboard exposes a tenant-scoped `Bilgi Tabanı` tab before a
call starts. It supports PDF, UTF-8 TXT and Markdown uploads up to 10 MiB,
authoritative bounded job progress, a maximum 50-item document list,
cancellation, and confirmed exact-document deletion. Tenant and knowledge-base
scope are never editable upload fields.

## Ephemeral document source policy

The background manager owns exactly one non-daemon worker and reserves one of
1-to-8 configured capacity slots before validation. Each accepted, validated
upload is held only in its bounded in-memory work envelope until that work
succeeds, fails, is cancelled, or is released during close. Source bytes are
never written to a repository or server storage path. Submission tokens are
idempotent for the manager lifetime, and duplicate documents do not retain a
second envelope. PostgreSQL remains authoritative for durable registry, job,
chunk, ordered metadata, and embedding state.

Because source bytes do not survive process exit, startup marks only that exact
tenant and knowledge base's interrupted `QUEUED` or `PROCESSING` jobs as
`FAILED` with the fixed `FINALIZE` phase. Terminal jobs and other scopes are
unchanged. Retrying after interruption requires re-uploading the source;
scope-local SHA-256 uniqueness resolves it to the existing registry identity.

Extraction, chunking, embedding, and vector finalization run only after the
worker claims a queued job. The production composition accepts only
`sentence-transformers/all-MiniLM-L6-v2` on CPU with 384 normalized dimensions
and `local_files_only=true`. Construction opens no database connection and
does not load the model; model loading occurs lazily only for accepted worker
work and has no download fallback.

## Dashboard document configuration

No document storage root or orphan reconciliation is configured or required.
Configure both bounded execution values together; missing, partial, or invalid
configuration leaves the document tab unavailable while base coaching remains
operational:

```text
CALLMETRIC_DASHBOARD_DOCUMENT_MAX_WORKERS=1
CALLMETRIC_DASHBOARD_DOCUMENT_CAPACITY=<1..8>
```

The tab displays only filenames, media labels, formatted sizes, fixed progress
or readiness states, and UTC creation times. It never displays content,
digests, storage keys, paths, tenant/knowledge-base values, or internal IDs.
Deletion requires a short-lived one-use confirmation, refuses active ingestion,
and commits exact-scope vector/job/registry deletion. It performs no storage
cleanup and makes no orphan-reconciliation claim.

`Belge hazır`, retrieved context used, and LLM generation used are three
different facts. Document readiness alone must not be reported as retrieval or
generation. PostgreSQL and vLLM remain externally managed services; the
dashboard does not start, stop, migrate, or reconfigure them.

## External PostgreSQL requirement

Real smoke testing requires an externally managed PostgreSQL/pgvector endpoint
with server-side TLS. Use `sslmode=verify-full`, a trusted root CA configured
outside Git, and a certificate whose hostname matches the connection hostname.
Migration and application credentials should be distinct and environment-only.

Before smoke execution, verify on that exact connection:

```sql
SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid();
```

The result must be `true`. `compose.postgres-integration.yml` does not configure
or prove TLS and remains limited to non-TLS integration tests. Do not weaken
`PostgreSQLVectorStoreSettings` or use `sslmode=disable`.

## Repository-owned TLS smoke

PR52 adds a separate, opt-in environment in
`compose.postgres-tls-smoke.yml`. It does not modify or reuse the ordinary
non-TLS integration environment. Run it only through the bounded controller:

```powershell
uv run python scripts/run_postgres_tls_smoke.py
```

The controller uses the immutable pgvector 0.8.5 image digest, creates a fresh
ephemeral CA and server certificate outside the repository, and gives the
server certificate both `DNS:localhost` and `IP:127.0.0.1` subject alternative
names. Application and migration connections use `localhost`,
`sslmode=verify-full`, and that run's private trust root. The test also proves
that missing trust and a mismatched hostname fail closed.

The TLS smoke applies ordered migrations through `0003` and verifies an
idempotent second application, schema readiness, exact profile registration,
deterministic synthetic ingestion and tenant-scoped retrieval. Its synthetic
embedding backend validates database wiring only; it is not a real
embedding-model smoke test.

All passwords, DSNs, private keys, certificates and temporary paths are
generated for one invocation and remain outside Git. The controller always
removes its project-scoped container, network, volume and complete certificate
directory. A failure must be reported with fixed secret-free output.

## External vLLM requirement

Real generation requires a separately managed Linux/GPU vLLM service exposing
a canonical HTTPS `/v1/completions` API. Certificate-chain and hostname
verification are mandatory and `CALLMETRIC_VLLM_VERIFY_TLS=true` is required.
Token, endpoint and model values remain environment-only.

Do not treat a Windows RTX 3050 4 GB machine as an approved vLLM host. This
runbook provides no local model download or start command.

## READY verification

Start the dashboard only after migration, exact profile provisioning, and
document ingestion:

```powershell
uv run streamlit run live_dashboard/app.py
```

Select the explicitly overridden synthetic tenant and start one trusted
synthetic audio call. `RAG hazır` confirms schema readiness, exact registered
profile verification, and bounded manager startup. It does not prove embedding
or vLLM generation because both providers remain lazy.

## Real generation verification

Use a stable synthetic utterance whose enabled classification label has
nonempty retrieved context. A real smoke succeeds only when retrieval returns
evidence and the external HTTPS vLLM response produces an LLM coaching
suggestion. Empty retrieval must not contact vLLM. Provider failure must leave
base coaching operational without exposing identifiers, prompts or errors.

## Cleanup

Use the dashboard reset action to close and remove the call-scoped execution
resource exactly once, then stop Streamlit. Stop external services through
their owning deployment system and verify its project-scoped connections,
containers, networks and volumes are gone. Never delete shared infrastructure
or certificates from this dashboard process.
