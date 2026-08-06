# Windows document-dashboard RAG/vLLM E2E controller

This opt-in controller proves the safe dashboard presentation-model boundary
without starting Streamlit. It composes the committed migration, readiness,
profile, document manager, MiniLM, retrieval, HTTPS vLLM, result gate,
completion pump and citation projector operations. Success is not a literal
browser-render test.

Ubuntu L40S runs the separately reviewed TTL-bounded vLLM service on
`127.0.0.1:8001`. Windows owns TLS PostgreSQL, the immutable MiniLM snapshot and
this controller. An operator-owned SSH forward exposes the Ubuntu service as
`https://localhost:9443/v1`; neither controller creates or owns that tunnel.
No vLLM port is public, and hostname/CA verification must remain enabled.

Set these placeholders through a private Windows process environment:

```text
CALLMETRIC_DASHBOARD_RAG_E2E_EXPECTED_BRANCH=<REVIEWED_BRANCH>
CALLMETRIC_DASHBOARD_RAG_E2E_EXPECTED_HEAD=<REVIEWED_40_HEX_HEAD>
CALLMETRIC_DASHBOARD_RAG_E2E_EXPECTED_BASELINE=<REVIEWED_40_HEX_BASELINE>
CALLMETRIC_DASHBOARD_RAG_E2E_POSTGRES_TTL_SECONDS=<300..7200>
CALLMETRIC_POSTGRES_TLS_SERVICE_HANDOFF_ROOT=<OWNER_ONLY_ROOT>
CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH=<ABSOLUTE_PROVIDER_JSON>
CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH=<ABSOLUTE_POLICY_JSON>
CALLMETRIC_VLLM_BASE_URL=https://localhost:9443/v1
CALLMETRIC_VLLM_MODEL_ID=<PINNED_SERVED_IDENTITY>
CALLMETRIC_VLLM_API_TOKEN=<PRIVATE_BEARER_TOKEN>
CALLMETRIC_VLLM_CA_CERTIFICATE_PATH=<PRIVATE_CA_FILE>
CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS=<1..60>
CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS=<1..600>
CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS=256
CALLMETRIC_VLLM_TEMPERATURE=<APPROVED_BOUND>
CALLMETRIC_VLLM_VERIFY_TLS=true
```

Provider JSON contains only the fixed synthetic scope, canonical MiniLM model
identity, approved absolute immutable snapshot, dimension 384, normalization
true, CPU and local-only true. Policy JSON contains only the reviewed synthetic
RAG label/action/title/priority/expiry. Neither file contains secrets or an
endpoint. The ignored snapshot must identify revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and match the checked manifest.
Preflight verifies it without constructing the model.

Execution order:

1. Verify both repositories and external resource inventories.
2. Start Ubuntu vLLM; transfer its CA/token confidentially.
3. Establish the Windows loopback SSH forward.
4. Run `uv run python -m scripts.run_windows_dashboard_rag_vllm_e2e --preflight-only`.
5. Require only `PREFLIGHT_OK`; no Docker/database/HTTP/model activity occurs.
6. Run `uv run python -m scripts.run_windows_dashboard_rag_vllm_e2e`.
7. Require only `E2E_OK`.

Full mode starts one randomized loopback TLS PostgreSQL service. That service
proves migrations 0001-0003 and repeat idempotency before its owner-only
handoff. The controller verifies ledger/readiness, provisions the exact cosine
profile, loads MiniLM offline, ingests two bounded in-memory TXT sources,
requires READY and NULL source keys, checks normalized 384-dimensional vectors,
and runs one HTTPS orchestration. The existing gate must ground the admitted
suggestion. Completion pumping and exact-scope projection must yield exactly one
safe dashboard source containing only display filename and `TXT`. Duplicate
ingestion may not expand vectors; deletion preserves the other document/profile.

Failures print only a fixed `E_*` phase. On every path the controller closes
background resources, deletes only reachable synthetic documents and terminates
the PostgreSQL service. The service removes its exact Compose project, volumes,
orphans and private run material, then verifies unrelated resources. It never
prunes or removes images, model caches, unrelated stopped containers, networks
or anonymous volumes. Ubuntu cleanup stops only its vLLM project, verifies GPU
idle, removes run TLS/token material and preserves the model cache.

Never log or screenshot source/transcript/prompt/completion content; scope or
document/chunk/job identities; filenames or private paths; DSNs, passwords,
tokens, certificates, environment mappings, embeddings, endpoint/model settings
or raw exceptions. Main merge remains prohibited until the real E2E and both-host
cleanup succeed.
