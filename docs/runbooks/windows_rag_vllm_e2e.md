# Windows synthetic RAG-to-vLLM E2E

This phase-4 artifact is a later, operator-invoked proof for the existing
dashboard RAG contracts. It uses only the fixed `tenant_alpha` / `kb_smoke` /
`urun_bilgisi` synthetic scenario. It does not start the dashboard, Docker,
AWS vLLM controller, model server, certificates, or SSH tunnel.

## Safety boundary

Before use, the operator separately establishes and validates the Windows SSH
tunnel that exposes the already-running HTTPS endpoint as
`https://localhost:9443/v1`. The controller never owns that tunnel or any AWS
lifecycle. It never downloads the pinned
`sentence-transformers/all-MiniLM-L6-v2` embedding model; that 384-dimensional
model must already exist in the local cache.

PostgreSQL must use `CALLMETRIC_POSTGRES_SSL_MODE=verify-full`. Supply its DSN
through the existing secret-managed `CALLMETRIC_POSTGRES_DSN` setting. Supply
the existing vLLM base URL, model ID, custom CA, TLS verification, and bounded
timeout/output settings through their `CALLMETRIC_VLLM_*` settings. Set
`CALLMETRIC_VLLM_API_TOKEN_FILE` to an absolute, regular token file using the
Windows secret/session mechanism. The controller reads that file internally
into the existing secret `api_token` setting; do not set the token on a command
line. Do not put token, DSN, CA path, or other private paths in command
arguments, transcripts, screenshots, or logs.

## Preflight

From the repository root, after injecting settings through the private session
environment, run:

```powershell
uv run python -m scripts.run_windows_rag_vllm_e2e --preflight-only
```

`PREFLIGHT_OK` means only that the exact non-secret examples, local-only model
identity, verify-full PostgreSQL configuration, localhost HTTPS endpoint,
token file, CA file, strict TLS setting, and bounds are structurally valid.
Preflight opens no database or HTTP connection and performs no write.

## Future real E2E lifecycle

Only after PostgreSQL, the external tunnel, and AWS vLLM are independently
ready, run without `--preflight-only`. The controller:

1. clears only the fixed synthetic tenant/KB scope;
2. verifies schema readiness and provisions the exact embedding profile;
3. re-verifies readiness/profile identity, ingests one synthetic document, and
   requires its exact expected document/chunk identity;
4. executes the existing retrieval, cited prompt, and HTTPS `/v1/completions`
   orchestration with bounded component timeouts and a 300-second total
   deadline;
5. applies the existing `LLMCoachingResultGate` and suggestion factory, then
   requires the exact citation/scope, non-empty suggestion, and `llm` source;
6. deletes only the synthetic vector records/profile on success or failure.

The only stdout success values are `PREFLIGHT_OK` and `E2E_OK`. Failures use a
fixed phase code. No transcript, document, prompt, completion, token, DSN,
certificate value/path, or private path is printed. Missing retrieval and vLLM
connectivity fail deterministically as `E_RETRIEVAL_UNAVAILABLE` and
`E_VLLM_UNAVAILABLE`; they do not fabricate a suggestion. PostgreSQL failures
are separated into `E_INITIAL_CLEANUP`, `E_PROVISIONING`, `E_INGESTION`, and
`E_RETRIEVAL_UNAVAILABLE`, without exposing database exception details. Final
exact-scope cleanup always runs and cannot replace an existing primary phase.

The cleanup does not remove PostgreSQL volumes, embedding/model caches,
certificates, tunnels, containers, networks, or unrelated tenant data. If the
process is forcibly terminated, repeat the same exact-scope cleanup through an
approved database procedure before another run.
