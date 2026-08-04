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

Execution requires two existing directories outside the repository. The
persistent Hugging Face cache selected by
`CALLMETRIC_VLLM_SMOKE_CACHE_DIR` must be an absolute, canonical,
non-symlink directory owned by the invoking user and must not be group/world
writable; the cache itself is not required to have mode `0700`. The handoff
parent selected by `CALLMETRIC_VLLM_E2E_HANDOFF_ROOT` remains owner-only and
must permit the invoking user to create a child directory (use mode `0700`).
The run-specific handoff child is mode `0700`, and its `ca.crt`, `token`,
and `connection.json` files are mode `0600`.

The controller also requires the reviewed clean commit in
`CALLMETRIC_VLLM_E2E_EXPECTED_HEAD`. Provide these values without exposing
them in shared logs or committed configuration.

Invoke the controller only after separate approval, with an explicit TTL from
300 through 7200 seconds:

```text
uv run python -m scripts.run_vllm_e2e_service --ttl-seconds <300-7200>
```

Do not place the bearer token or private paths in command arguments or command
history. Each invocation creates a unique PR54 Compose project, a fresh
ephemeral CA and localhost server certificate, and a random bearer token. It
reuses the strict PR53 CA/server profile with critical constraints and usages,
SKI/AKI linkage, server-only EKU, and SAN exactly
`DNS:localhost,IP:127.0.0.1`.

A run-specific owner-only handoff directory contains exactly `ca.crt`,
`token`, and `connection.json`. Transfer the CA and token through an
approved confidential channel without printing their contents. The metadata
contains only localhost connection details, the served-model name, filenames,
and TTL.

The operator can list only the permitted handoff file paths without reading
their contents:

```bash
find "$CALLMETRIC_VLLM_E2E_HANDOFF_ROOT" -mindepth 2 -maxdepth 2 -type f \
  \( -name 'ca.crt' -o -name 'token' -o -name 'connection.json' \) -print
```

Compose always performs `pull vllm`, including when the pinned image layers
already exist locally. Cached layers remain reusable, but the pull still
performs a registry check. The persistent model cache is reused and preserved.

## Cold and verified-warm disk guards

The unchanged cold-install guard requires exactly 60,726,534,621 bytes
(56.556 GiB): the conservative image/model transient allowance, 15 GiB final
reserve, and 2 GiB margin.

The lower verified-warm guard is selected only when all of these checks pass:

- the local image ID is exactly the pinned Linux/amd64 manifest digest;
- the cache root passes the existing canonical-path, ownership, symlink, and
  group/world-write checks;
- the metadata-only expected snapshot directory exists at the exact approved
  model revision;
- the Compose contract exposes controlled offline inputs and the PR54
  controller forces both `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`. Their Compose defaults remain `0`, so the
  verified PR53 smoke behavior is unchanged.

The controller derives the expected snapshot path directly and does not list,
read, or log cache contents. Verified-warm mode requires exactly
20,401,094,656 bytes (19 GiB): the unchanged 15 GiB reserve, unchanged 2 GiB
margin, and a 2 GiB bounded allowance for container writable-layer, logs, and
runtime temporary overhead. If any proof is missing or wrong, the controller
uses the unchanged cold guard. An incomplete snapshot that has the expected
directory still fails closed during offline startup/readiness; Hugging Face or
Transformers cannot fall back to a model download.

The exact-image Compose pull and registry verification remain enabled and may
contact the image registry. Offline mode applies to model resolution and never
permits a different image digest.

## Readiness, lifetime, and cleanup

The controller declares READY only after trusted HTTPS `/health` succeeds and
authenticated `/v1/models` returns exactly
`callmetric-qwen25-7b-awq`. It then remains in the foreground until TTL
expiry, SIGINT, SIGTERM, or SIGHUP where supported. Status and failures are
fixed and secret-free;
requests, prompts, responses, tokens, certificate material, environment
values, cache contents, and private paths are never logged.

GPU 0 must be idle before startup. After readiness, any positive GPU-memory
use is accepted as bounded service activity; no exact MiB value is required.
After exact-project shutdown, GPU idleness is polled for at most 30 seconds.
Persistent activity fails closed without inspecting or stopping GPU processes.

Failures emit exactly one fixed phase code followed by
`PR54 vLLM service failed`. Codes include `E_REPOSITORY`,
`E_RUNTIME_CONTRACT`, `E_GPU`, `E_GPU_ACTIVITY`, `E_GPU_CLEANUP`,
`E_DISK_CAPACITY`,
`E_CACHE_METADATA`, `E_IMAGE_METADATA`, `E_MODEL_METADATA`, `E_TLS`,
`E_STARTUP`, `E_READINESS`, `E_CLEANUP`, and
`E_PROTECTED_CONTAINERS`. Exception text and sensitive values are never
emitted.

Bounded operation uses these exact timeouts:

- General subprocess: 30 seconds
- Image pull: 1800 seconds
- Startup: 120 seconds
- Trusted HTTPS readiness: 900 seconds
- Each HTTPS request: 30 seconds
- Shutdown: 120 seconds
- Post-cleanup GPU-idle wait: 30 seconds

Normal completion, TTL expiry, SIGINT, SIGTERM, supported SIGHUP, and handled
failures run `docker compose down` only for that randomized PR54 project, then
remove only
its ephemeral TLS/token/handoff material. Cleanup does not remove the pinned
image or persistent model cache and never uses volumes cleanup, prune, broad
stop/remove, or wildcard resource selection. The four protected containers are
checked before and after the lifecycle; any identity or status change fails
safely. Cleanup ordering is exact-project shutdown, bounded GPU-idle polling,
handoff and ephemeral TLS removal, then protected-container validation. A
cleanup failure is reported as `E_CLEANUP` only when no earlier lifecycle phase
already failed; cleanup cannot replace the original failure phase. Controlled
signal handlers are installed before startup and restored only after protected
resource validation, so repeated signals request shutdown without duplicating
project cleanup.

SIGKILL, host failure, or interpreter-runtime failure can prevent Python
`finally` cleanup. After such an interruption, first use `docker compose ls` to identify
the exact `callmetric-vllm-e2e-<pid>-<suffix>` project name. Verify only that
exact project with:

```bash
docker compose --project-name '<exact-pr54-project>' \
  --file compose.vllm-loopback-smoke.yml ps
```

Also verify that port 8001 is free and inspect only the known handoff parent for
the three permitted filenames above. Any cleanup must target the confirmed
exact PR54 project and run-specific handoff child; never use prune, broad
container removal, wildcard selection, or cache deletion.
