from __future__ import annotations

import json
import re
import ssl
import subprocess
from pathlib import Path

import pytest

from scripts import run_vllm_loopback_smoke as subject


def _completed(
    arguments: list[str], stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


def test_immutable_artifacts_and_awq_selection_are_exact() -> None:
    assert subject.IMAGE_INDEX_DIGEST == (
        "sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e"
    )
    assert subject.IMAGE_AMD64_DIGEST == (
        "sha256:1161da8a5edbdff239ab1812784d7fe5d28775c675809a8420e8a0a05d0e56d1"
    )
    assert subject.MODEL == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert subject.MODEL_REVISION == "b25037543e9394b818fdfca67ab2a00ecc7dd641"
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8")
    assert subject.IMAGE in compose
    assert "--quantization" in compose and '- "awq"' in compose


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:ef7bfc14df9233e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e",
        "sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53",
        "sha256:EF7BFC14DF9233E3E5D41E733E3BE0AFA6ABBE5AE5F14EE0758110030F6CD53E",
        "sha256:not-hexadecimal",
        "sha512:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e",
    ],
)
def test_sha256_digest_validation_rejects_invalid_values(digest: str) -> None:
    with pytest.raises(subject.VLLMSmokeError, match="^vLLM loopback smoke failed$"):
        subject._validate_sha256_digest(digest)


def test_registry_digest_is_valid_and_consistent_across_artifacts() -> None:
    subject._validate_sha256_digest(subject.IMAGE_INDEX_DIGEST)
    assert len(subject.IMAGE_INDEX_DIGEST.removeprefix("sha256:")) == 64
    runner = Path(subject.__file__).read_text(encoding="utf-8")
    tests = Path(__file__).read_text(encoding="utf-8")
    runbook = (
        subject.REPOSITORY_ROOT / "docs" / "runbooks" / "vllm_loopback_smoke.md"
    ).read_text(encoding="utf-8")
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8")
    assert runner.count(subject.IMAGE_INDEX_DIGEST) == 1
    assert tests.count(subject.IMAGE_INDEX_DIGEST) == 1
    assert subject.IMAGE_INDEX_DIGEST in runbook
    assert subject.IMAGE_INDEX_DIGEST not in compose
    assert compose.count(subject.IMAGE) == 1


def test_compose_is_loopback_only_and_isolated() -> None:
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8")
    assert '"127.0.0.1:8001:8001"' in compose
    assert "0.0.0.0:8001" not in compose
    assert "[::]" not in compose
    assert "container_name" not in compose
    assert "privileged" not in compose
    assert "/var/run/docker.sock" not in compose
    assert 'restart: "no"' in compose
    assert "compose.postgres" not in compose


def test_compose_has_tls_token_gpu_logging_and_runtime_guards() -> None:
    compose = subject.COMPOSE_FILE.read_text(encoding="utf-8")
    for expected in (
        'device_ids: ["0"]',
        'shm_size: "4gb"',
        'max-size: "10m"',
        'max-file: "2"',
        ":/run/vllm-tls:ro",
        "--ssl-keyfile",
        "--ssl-certfile",
        "--max-model-len",
        '- "8192"',
        "--gpu-memory-utilization",
        '- "0.80"',
        "--max-num-seqs",
        '- "2"',
        "--no-trust-remote-code",
        "--no-enable-log-requests",
    ):
        assert expected in compose
    assert "VLLM_API_KEY:" in compose
    assert "--api-key" not in compose
    assert "HF_TOKEN" not in compose


@pytest.mark.parametrize(
    "name",
    ["", "callmetric-vllm-smoke", "unsafe project", "callmetric-vllm-smoke-1-A"],
)
def test_project_names_are_strictly_validated(name: str) -> None:
    with pytest.raises(subject.VLLMSmokeError):
        subject._compose_arguments(name, "config")


def _porcelain_record(status: str, path: str) -> str:
    return f"{status} {path}\0"


def _exact_artifact_status() -> str:
    return "".join(
        _porcelain_record("??", path)
        for path in sorted(subject.EXPECTED_ARTIFACT_PATHS)
    )


def test_worktree_status_accepts_exact_four_untracked_artifacts() -> None:
    subject._validate_worktree_status(_exact_artifact_status())


@pytest.mark.parametrize(
    "raw_status",
    [
        "",
        _exact_artifact_status().replace(
            _porcelain_record("??", "compose.vllm-loopback-smoke.yml"), ""
        ),
        _exact_artifact_status() + _porcelain_record("??", "unexpected.txt"),
        _porcelain_record(" M", "tracked.py") + _exact_artifact_status(),
        _exact_artifact_status().replace(
            _porcelain_record("??", "compose.vllm-loopback-smoke.yml"),
            _porcelain_record("A ", "compose.vllm-loopback-smoke.yml"),
        ),
        _porcelain_record("R ", "renamed.py")
        + "original.py\0"
        + _exact_artifact_status(),
        _porcelain_record(" D", "tracked.py") + _exact_artifact_status(),
        _porcelain_record("UU", "tracked.py") + _exact_artifact_status(),
        _exact_artifact_status().replace(
            _porcelain_record("??", "compose.vllm-loopback-smoke.yml"),
            _porcelain_record("??", "../compose.vllm-loopback-smoke.yml"),
        ),
        _exact_artifact_status().replace(
            _porcelain_record("??", "scripts/run_vllm_loopback_smoke.py"),
            _porcelain_record("??", "scripts/run_vllm_loopback_smoke.py.backup"),
        ),
        "?? malformed-without-nul",
        "? malformed-status\0",
        "??\0",
        _exact_artifact_status() + "\0",
    ],
)
def test_worktree_status_rejects_non_exact_or_malformed_state(
    raw_status: str,
) -> None:
    with pytest.raises(subject.VLLMSmokeError, match="^vLLM loopback smoke failed$"):
        subject._validate_worktree_status(raw_status)


def test_current_worktree_uses_nul_porcelain_with_all_untracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return _completed(arguments, _exact_artifact_status())

    monkeypatch.setattr(subject, "_run", fake_run)
    subject._validate_current_worktree()

    assert calls == [
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ]
    ]


def test_disk_guard_requires_transient_reserve_and_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert subject.MINIMUM_FREE_BYTES == (subject.TRANSIENT_BYTES + 17 * 1024**3)
    monkeypatch.setattr(
        subject,
        "_free_disk_bytes",
        lambda: subject.MINIMUM_FREE_BYTES - 1,
    )
    with pytest.raises(subject.VLLMSmokeError):
        subject._validate_disk()
    monkeypatch.setattr(subject, "_free_disk_bytes", lambda: subject.MINIMUM_FREE_BYTES)
    subject._validate_disk()


def test_cache_must_be_external_owned_private_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(subject.os, "getuid", lambda: cache.stat().st_uid)
    assert subject._validate_cache_directory(str(cache)) == cache.resolve()
    cache.chmod(0o777)
    with pytest.raises(subject.VLLMSmokeError):
        subject._validate_cache_directory(str(cache))


def test_image_metadata_requires_index_and_linux_amd64_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = {
        "manifests": [
            {
                "digest": subject.IMAGE_AMD64_DIGEST,
                "platform": {"architecture": "amd64", "os": "linux"},
            }
        ]
    }
    outputs = iter(
        [
            f"Name: x\nDigest: {subject.IMAGE_INDEX_DIGEST}\n",
            json.dumps(index),
        ]
    )
    monkeypatch.setattr(subject, "_output", lambda *_args, **_kwargs: next(outputs))
    subject._validate_image_metadata()


def test_model_metadata_rejects_community_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "community-publisher/Qwen2.5-7B-Instruct-AWQ",
                    "author": "community-publisher",
                    "sha": subject.MODEL_REVISION,
                    "tags": ["license:apache-2.0", "awq"],
                    "config": {
                        "architectures": ["Qwen2ForCausalLM"],
                        "quantization_config": {"bits": 4, "quant_method": "awq"},
                    },
                }
            ).encode()

    monkeypatch.setattr(
        subject.urllib.request, "urlopen", lambda *_a, **_k: _Response()
    )
    with pytest.raises(subject.VLLMSmokeError):
        subject._validate_model_metadata()


def test_certificate_commands_use_exact_san_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Path]] = []
    profiles: list[str] = []

    def fake_run(
        arguments: list[str], *, cwd: Path = subject.REPOSITORY_ROOT, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, cwd))
        if "-extfile" in arguments:
            profiles.append(
                (cwd / arguments[arguments.index("-extfile") + 1]).read_text()
            )
        if "x509" in arguments and "-ext" in arguments:
            return _completed(
                arguments,
                "X509v3 Subject Alternative Name:\n"
                "    DNS:localhost, IP Address:127.0.0.1\n",
            )
        for name in subject.EXPECTED_CERTIFICATE_FILES:
            (tmp_path / name).touch(exist_ok=True)
        return _completed(arguments)

    monkeypatch.setattr(subject, "_run", fake_run)
    subject._generate_certificates(tmp_path)
    rendered = tuple(" ".join(arguments) for arguments, _cwd in calls)
    assert any(
        "basicConstraints=critical,CA:TRUE,pathlen:0" in call for call in rendered
    )
    assert profiles == [subject._SERVER_CERTIFICATE_EXTENSIONS]
    assert "subjectAltName=DNS:localhost,IP:127.0.0.1" in profiles[0]
    assert not (tmp_path / "server.ext").exists()
    assert all(str(tmp_path) not in call for call in rendered)
    assert all(cwd == tmp_path for _arguments, cwd in calls)


def test_request_construction_is_synthetic_bounded_and_never_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"text":"synthetic result"}]}'

    def fake_open(request: object, **kwargs: object) -> _Response:
        captured["request"] = request
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(subject.urllib.request, "urlopen", fake_open)
    status, _payload = subject._request(
        "/v1/completions",
        context=subject.ssl.create_default_context(),
        token="synthetic-secret-token",
        body={
            "model": subject.SERVED_MODEL,
            "prompt": subject.SYNTHETIC_PROMPT,
            "max_tokens": 256,
        },
    )
    assert status == 200
    request = captured["request"]
    assert getattr(request, "full_url") == "https://localhost:8001/v1/completions"
    body = json.loads(getattr(request, "data"))
    assert body["max_tokens"] == 256
    assert "Sentetik" in body["prompt"]
    assert captured["timeout"] == subject.HTTP_TIMEOUT_SECONDS


def test_health_accepts_an_empty_success_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    monkeypatch.setattr(
        subject.urllib.request, "urlopen", lambda *_a, **_k: _Response()
    )
    status, payload = subject._request(
        "/health", context=subject.ssl.create_default_context(), token=None
    )
    assert status == 200
    assert payload == {}


def test_api_checks_tokens_model_and_synthetic_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None, dict[str, object] | None]] = []

    def fake_request(
        path: str,
        *,
        context: object,
        token: str | None,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        del context
        calls.append((path, token, body))
        if token in {None, "synthetic-incorrect-token"}:
            return 401, {}
        if path == "/v1/models":
            return 200, {"data": [{"id": subject.SERVED_MODEL}]}
        return 200, {"choices": [{"text": "synthetic non-empty"}]}

    monkeypatch.setattr(subject, "_request", fake_request)
    subject._verify_api(subject.ssl.create_default_context(), "synthetic-token")

    assert [token for _path, token, _body in calls[:2]] == [
        None,
        "synthetic-incorrect-token",
    ]
    completion = calls[-1]
    assert completion[0] == "/v1/completions"
    assert completion[2] is not None
    assert completion[2]["prompt"] == subject.SYNTHETIC_PROMPT
    assert completion[2]["max_tokens"] == 256


def test_runner_source_has_no_prompt_response_or_secret_logging() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "print(completion" not in source
    assert "print(token" not in source
    assert "logging." not in source
    assert "token_urlsafe" in source
    assert "prune" not in source
    assert 'down", "--volumes' not in source


def test_subprocesses_are_bounded_and_fail_with_fixed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("synthetic", 1)

    monkeypatch.setattr(subject.subprocess, "run", timeout)
    with pytest.raises(subject.VLLMSmokeError, match="^vLLM loopback smoke failed$"):
        subject._run(["synthetic-command"])


def test_cleanup_targets_only_randomized_project_without_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subject,
        "_run",
        lambda arguments, **_kwargs: calls.append(arguments) or _completed(arguments),
    )
    subject._cleanup("callmetric-vllm-smoke-123-abcdef123456", {})
    rendered = " ".join(calls[0])
    assert "callmetric-vllm-smoke-123-abcdef123456" in rendered
    assert " down --remove-orphans --timeout 30" in rendered
    assert "--volumes" not in rendered
    assert "prune" not in rendered


def test_main_emits_only_fixed_secret_free_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        subject,
        "run",
        lambda: (_ for _ in ()).throw(RuntimeError("token prompt private/path")),
    )
    assert subject.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "vLLM loopback smoke failed\n"
    assert "token prompt" not in captured.err


def test_protected_container_ids_are_exact_and_checked_after_cleanup() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert set(subject.PROTECTED_CONTAINERS) == {
        "rag-redis",
        "rag-postgres",
        "rag-adminer",
        "callmetric-local-rabbitmq",
    }
    for container_id in subject.PROTECTED_CONTAINERS.values():
        assert len(container_id) == 64
    assert source.count("_protected_container_snapshot()") >= 2


def _openssl(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["openssl", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=subject.SUBPROCESS_TIMEOUT_SECONDS,
    )


def _certificate_extension(directory: Path, certificate: str, name: str) -> str:
    result = _openssl("x509", "-in", certificate, "-noout", "-ext", name, cwd=directory)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _key_identifier(extension: str) -> str:
    matched = re.search(r"(?:[0-9A-F]{2}:){19}[0-9A-F]{2}", extension)
    assert matched is not None
    return matched.group(0)


def _complete_memory_bio_handshake(
    client_context: ssl.SSLContext,
    server_context: ssl.SSLContext,
    *,
    hostname: str,
) -> None:
    client_input = ssl.MemoryBIO()
    client_output = ssl.MemoryBIO()
    server_input = ssl.MemoryBIO()
    server_output = ssl.MemoryBIO()
    client = client_context.wrap_bio(
        client_input,
        client_output,
        server_hostname=hostname,
    )
    server = server_context.wrap_bio(server_input, server_output, server_side=True)
    client_complete = False
    server_complete = False
    for _ in range(20):
        if not client_complete:
            try:
                client.do_handshake()
                client_complete = True
            except ssl.SSLWantReadError:
                pass
        client_bytes = client_output.read()
        if client_bytes:
            server_input.write(client_bytes)
        if not server_complete:
            try:
                server.do_handshake()
                server_complete = True
            except ssl.SSLWantReadError:
                pass
        server_bytes = server_output.read()
        if server_bytes:
            client_input.write(server_bytes)
        if client_complete and server_complete:
            return
    pytest.fail("in-memory TLS handshake did not complete")


def test_generated_certificates_have_strict_extensions(tmp_path: Path) -> None:
    subject._generate_certificates(tmp_path)

    verified = _openssl(
        "verify",
        "-x509_strict",
        "-CAfile",
        "ca.crt",
        "server.crt",
        cwd=tmp_path,
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == "server.crt: OK\n"

    ca_basic = _certificate_extension(tmp_path, "ca.crt", "basicConstraints")
    ca_usage = _certificate_extension(tmp_path, "ca.crt", "keyUsage")
    ca_ski = _certificate_extension(tmp_path, "ca.crt", "subjectKeyIdentifier")
    ca_aki = _certificate_extension(tmp_path, "ca.crt", "authorityKeyIdentifier")
    assert "X509v3 Basic Constraints: critical" in ca_basic
    assert "CA:TRUE, pathlen:0" in ca_basic
    assert "X509v3 Key Usage: critical" in ca_usage
    assert "Certificate Sign, CRL Sign" in ca_usage
    assert _key_identifier(ca_aki) == _key_identifier(ca_ski)

    server_basic = _certificate_extension(tmp_path, "server.crt", "basicConstraints")
    server_usage = _certificate_extension(tmp_path, "server.crt", "keyUsage")
    server_eku = _certificate_extension(tmp_path, "server.crt", "extendedKeyUsage")
    server_ski = _certificate_extension(tmp_path, "server.crt", "subjectKeyIdentifier")
    server_aki = _certificate_extension(
        tmp_path, "server.crt", "authorityKeyIdentifier"
    )
    server_san = _certificate_extension(tmp_path, "server.crt", "subjectAltName")
    assert "X509v3 Basic Constraints: critical" in server_basic
    assert "CA:FALSE" in server_basic
    assert "X509v3 Key Usage: critical" in server_usage
    assert "Digital Signature, Key Encipherment" in server_usage
    assert "TLS Web Server Authentication" in server_eku
    assert _key_identifier(server_ski) != _key_identifier(ca_ski)
    assert _key_identifier(server_aki) == _key_identifier(ca_ski)
    assert server_san.splitlines()[1].strip() == ("DNS:localhost, IP Address:127.0.0.1")


def test_wrong_ca_and_hostname_still_fail(tmp_path: Path) -> None:
    subject._generate_certificates(tmp_path)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    subject._generate_certificates(wrong)

    wrong_ca = _openssl(
        "verify",
        "-x509_strict",
        "-CAfile",
        str(wrong / "ca.crt"),
        "server.crt",
        cwd=tmp_path,
    )
    wrong_hostname = _openssl(
        "verify",
        "-x509_strict",
        "-verify_hostname",
        "wrong.invalid",
        "-CAfile",
        "ca.crt",
        "server.crt",
        cwd=tmp_path,
    )
    assert wrong_ca.returncode != 0
    assert wrong_hostname.returncode != 0


def test_python_ssl_context_strict_verification_when_supported(
    tmp_path: Path,
) -> None:
    subject._generate_certificates(tmp_path)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    subject._generate_certificates(wrong)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(tmp_path / "server.crt", tmp_path / "server.key")

    trusted = ssl.create_default_context(cafile=tmp_path / "ca.crt")
    strict = getattr(ssl, "VERIFY_X509_STRICT", None)
    if strict is not None:
        trusted.verify_flags |= strict
        assert trusted.verify_flags & strict
    _complete_memory_bio_handshake(trusted, server_context, hostname="localhost")

    wrong_hostname = ssl.create_default_context(cafile=tmp_path / "ca.crt")
    if strict is not None:
        wrong_hostname.verify_flags |= strict
    with pytest.raises(ssl.SSLCertVerificationError):
        _complete_memory_bio_handshake(
            wrong_hostname, server_context, hostname="wrong.invalid"
        )

    wrong_ca = ssl.create_default_context(cafile=wrong / "ca.crt")
    if strict is not None:
        wrong_ca.verify_flags |= strict
    with pytest.raises(ssl.SSLCertVerificationError):
        _complete_memory_bio_handshake(wrong_ca, server_context, hostname="localhost")
