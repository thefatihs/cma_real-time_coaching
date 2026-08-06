# PR54 TTL-Bounded Windows PostgreSQL TLS Controller

This artifact starts the existing PR52 PostgreSQL/pgvector TLS Compose service
as a loopback-only, operator-controlled dependency. It does not change the PR52
smoke runner or Compose contract. Docker Desktop must use `desktop-linux`.

Create an empty directory outside the repository that only the current Windows
account can access, then set `CALLMETRIC_POSTGRES_TLS_SERVICE_HANDOFF_ROOT` to
its absolute resolved path. The controller creates a run-specific child with
mode 0700 and owner-only `ca.crt`, `application.dsn`, and `connection.json` files.
The DSN uses `localhost`, the random application credential, the ephemeral CA,
and `sslmode=verify-full`. Never print, copy into Git, or retain those files.

Run read-only preflight first (it checks the checkout, Docker/Compose/server,
`desktop-linux` context, pinned image contract, and snapshots resources):

```powershell
$env:CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_HEAD = (git rev-parse HEAD).Trim()
$env:CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_BRANCH = "<REVIEWED_BRANCH>"
uv run python -m scripts.run_postgres_tls_service --ttl-seconds 600 --preflight-only
```

After approval, start the foreground controller with a TTL from 300 through
7200 seconds:

```powershell
$env:CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_HEAD = (git rev-parse HEAD).Trim()
$env:CALLMETRIC_POSTGRES_TLS_SERVICE_EXPECTED_BRANCH = "<REVIEWED_BRANCH>"
uv run python -m scripts.run_postgres_tls_service --ttl-seconds 7200
```

The supplied expected HEAD must be the exact lowercase 40-character commit.
The controller fails closed unless HEAD matches the current checkout and the
remote-tracking ref for the reviewed expected branch; it never prints either
supplied value. The branch setting defaults to the legacy integration branch
for compatibility, while document verification sets it explicitly.

It generates fresh passwords and one-day TLS material, binds only a random
`127.0.0.1` port, validates the certificate, waits for container health, then
runs the existing integration proof. At a document-capable commit that proof
applies migrations 0001-0003, checks repeat idempotence, the nullable source-key
contract and schema readiness, and performs synthetic tenant-safe
pgvector operations. The opt-in ephemeral application role retains its existing
`CONNECT`, schema `USAGE`, and table `SELECT`/`INSERT`/`UPDATE` privileges. It
also receives `DELETE` only on `callmetric_vector.vector_records` and
`callmetric_vector.embedding_profiles` and `callmetric_vector.documents`, which permits the Windows E2E to remove
its fixed `tenant_alpha` / `kb_smoke` scope child-first. It receives no broad or
future-table delete grant. Job deletion occurs only through the document parent
cascade. `READY` is emitted only after all of those checks and
the owner-only handoff succeed.

## Bounded command timeouts

| Phase | Bound |
| --- | ---: |
| Repository, Docker validation, identity, ACL and ordinary resource queries | 30 seconds per command |
| Compose configuration validation | 30 seconds |
| Certificate generation and each certificate validation command | 60 seconds per command |
| Compose PostgreSQL startup | 120 seconds |
| PostgreSQL health window | 60 seconds, with 30 seconds per Docker query |
| Migration, provisioning and synthetic pgvector proof | 180 seconds |
| Exact-project Compose down and each residue query | 120 seconds per command |

A command timeout reports only its fixed lifecycle phase. Cleanup is still
attempted after startup or migration failure; a cleanup timeout cannot replace
that primary phase, but is `E_CLEANUP` when no earlier phase failed.

TTL expiry, SIGINT, SIGTERM, and SIGHUP where Windows exposes it remove the
handoff, exact randomized Compose project container/network/ephemeral volume,
and TLS directory. The controller never prunes or selects resources broadly;
it compares all pre-existing container, network, and volume IDs afterward.
The signal handlers remain installed through one cleanup lifecycle. Cleanup
first attempts bounded exact-project Compose down. If down fails, times out, or
leaves residue, the fallback enumerates only the exact project label, validates
the expected Compose service/container, default network, and named volume, then
removes those validated references in container/network/volume order. Any
unexpected name, label, or count fails closed. Residue verification completes
before the owner-only handoff and ephemeral TLS directory are deleted.
Forced process termination, `Stop-Process -Force`, host shutdown, power loss,
or a Python/Docker host crash can prevent signal/finally cleanup from running;
use exact-project resource verification before any manual recovery.

Failures expose only one fixed phase code and `PR54 PostgreSQL TLS service
failed`. Do not troubleshoot by printing DSNs, passwords, certificate paths, or
subprocess output. A cleanup failure cannot replace an earlier lifecycle phase.
