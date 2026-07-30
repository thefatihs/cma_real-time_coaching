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
