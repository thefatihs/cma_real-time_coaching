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
uv run python -m scripts.run_postgres_tls_service --ttl-seconds 600 --preflight-only
```

After approval, start the foreground controller with a TTL from 300 through
7200 seconds:

```powershell
uv run python -m scripts.run_postgres_tls_service --ttl-seconds 600
```

It generates fresh passwords and one-day TLS material, binds only a random
`127.0.0.1` port, validates the certificate, waits for container health, then
runs the existing PR52 integration proof. That proof applies migration 0001,
checks its idempotence and schema readiness, and performs synthetic tenant-safe
pgvector operations. `READY` is emitted only after all of those checks and the
owner-only handoff succeed.

TTL expiry, SIGINT, SIGTERM, and SIGHUP where Windows exposes it remove the
handoff, exact randomized Compose project container/network/ephemeral volume,
and TLS directory. The controller never prunes or selects resources broadly;
it compares all pre-existing container, network, and volume IDs afterward.

Failures expose only one fixed phase code and `PR54 PostgreSQL TLS service
failed`. Do not troubleshoot by printing DSNs, passwords, certificate paths, or
subprocess output. A cleanup failure cannot replace an earlier lifecycle phase.
