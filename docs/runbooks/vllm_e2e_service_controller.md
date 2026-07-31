# PR54 TTL-Bounded vLLM E2E Service

This controller is an opt-in artifact for a later Windows-to-VM synthetic E2E
test. It keeps the PR53 vLLM service available only for an explicit bounded
window. It must not be run during repository verification.

## Fixed boundaries

The controller reuses the PR53 Compose file and its immutable
`vllm/vllm-openai:v0.26.0-ubuntu2404` Linux/amd64 manifest, official
`Qwen/Qwen2.5-7B-Instruct-AWQ` revision
`b25037543e9394b818fdfca67ab2a00ecc7dd641`, and served name
`callmetric-qwen25-7b-awq`. GPU 0, 8192 context tokens, 0.80 GPU-memory
utilization, two sequences, and the 256-token application bound remain fixed.

Docker publishes only `127.0.0.1:8001`. The service is intended to be reached
from Windows through an SSH local forward to that loopback endpoint; this
artifact never opens a firewall or security-group port and never creates a
public, wildcard, or IPv6 listener.

## Future execution contract

Execution requires two existing owner-only directories outside the repository:
the persistent Hugging Face cache selected by
`CALLMETRIC_VLLM_SMOKE_CACHE_DIR`, and a handoff parent selected by
`CALLMETRIC_VLLM_E2E_HANDOFF_ROOT`. It also requires the reviewed clean commit
in `CALLMETRIC_VLLM_E2E_EXPECTED_HEAD`. Provide these values without exposing
them in shared logs or committed configuration.

Invoke the controller only after separate approval, with an explicit TTL from
300 through 7200 seconds:

```text
uv run python scripts/run_vllm_e2e_service.py --ttl-seconds <300-7200>
```

Do not place the bearer token or private paths in command arguments or command
history. Each invocation creates a unique PR54 Compose project, a fresh
ephemeral CA and localhost server certificate, and a random bearer token. The
certificate SAN is exactly `DNS:localhost,IP:127.0.0.1`.

A run-specific owner-only handoff directory contains exactly `ca.crt`,
`token`, and `connection.json`. Transfer the CA and token through an
approved confidential channel without printing their contents. The metadata
contains only localhost connection details, the served-model name, filenames,
and TTL.

## Readiness, lifetime, and cleanup

The controller declares READY only after trusted HTTPS `/health` succeeds and
authenticated `/v1/models` returns exactly
`callmetric-qwen25-7b-awq`. It then remains in the foreground until TTL
expiry, SIGINT, or SIGTERM. Status and failures are fixed and secret-free;
requests, prompts, responses, tokens, certificate material, environment
values, cache contents, and private paths are never logged.

Every exit runs `docker compose down` only for that randomized PR54 project,
then removes only its ephemeral TLS/token/handoff material. It does not remove
the pinned image or persistent model cache and never uses volumes cleanup,
prune, broad stop/remove, or wildcard resource selection. The four protected
containers are checked before and after the lifecycle; any identity or status
change fails safely. Pull, startup, readiness, subprocess, and shutdown work
all have finite timeouts.
