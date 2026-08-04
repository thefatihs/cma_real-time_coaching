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

The model is not downloaded during dashboard startup.
`trust_remote_code` remains false. Deterministic fake embeddings validate
wiring only and do not constitute a real embedding smoke test.

## Document-registry schema

The forward-only PostgreSQL migrator uses a fixed, ordered, digest-pinned
manifest. It applies `0001_vector_store.sql` before
`0002_document_registry.sql`, records both versions in
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
job IDs and opaque storage object keys are server-owned inputs. Creation inserts
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
foreign key cascade. After commit it returns the opaque object key to a future
storage layer; this code performs no filesystem deletion. Embedding profiles and
unrelated legacy vectors are untouched.

Dashboard upload UI, HTTP APIs, and progress polling UI are not implemented by
this change. The repository exposes scoped list and job-status operations plus
an application-owned background manager for later server-side integration.

## Private persistent document storage prerequisites

The backend storage primitive requires an operator-created absolute directory
outside this repository. It never creates the configured root or changes its
ACL. The root, every path component and every managed object must be a regular
non-link object without a symlink, junction or Windows reparse point. Filesystem
roots, the user-profile root, broad shared roots, this repository and its
descendants are rejected.

On POSIX, the effective user must own the directory and its mode must be `0700`
or stricter. On Windows, owner and DACL checks use security APIs rather than
localized command output: the running account must own and control the root and
no unrelated principal may have writable access. Validation failure is closed
and emits only a fixed storage category. Operators remain responsible for
creating and maintaining this owner-only directory before application startup.

Accepted source bytes are stored under random server-owned direct-child keys;
filenames, tenant/knowledge-base identifiers, digests and content never form a
path. Writes are bounded to 10 MiB, flushed and synced before exclusive atomic
publication. Successful ingestion retains the source for retry or audit.
Registry deletion commits before source deletion, so a later storage failure
creates an orphan and never restores the database row. Duplicate registry
resolution deletes only the new attempt's object.

The background manager owns exactly one non-daemon worker and reserves one of
1-to-8 configured capacity slots before validation or storage. Accepted bytes
are released from queued memory after persistent storage succeeds; only opaque
server metadata enters the bounded queue. Submission tokens are idempotent for
the manager lifetime. Closing refuses new work, cooperatively cancels queued or
running work, and a waiting close has a fixed upper bound. PostgreSQL remains
authoritative for durable state and progress. Successful and failed source
objects are retained for explicit retry; duplicate-attempt objects alone are
removed immediately.

Extraction, chunking, embedding, and vector finalization run only after the
worker claims a queued job. The production composition accepts only
`sentence-transformers/all-MiniLM-L6-v2` on CPU with 384 normalized dimensions
and `local_files_only=true`. Construction opens no database connection and
does not load the model; model loading occurs lazily only for accepted worker
work and has no download fallback.

Orphan reconciliation is explicit, never scheduled automatically. The composed
runtime first obtains a complete tenant/knowledge-base registry-key snapshot;
snapshot failure deletes nothing. Cleanup ignores unrelated names, applies the
configured 300-to-604800-second grace period, removes at most 100 deterministic
candidates, and returns safe counts only. Dashboard document UI and automatic
orphan scheduling remain unimplemented.

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

The TLS smoke applies the ordered `0001` and `0002` migrations and verifies an
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
