from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier, Lock

import pytest

from app.composition.postgres_rag_background import (
    BoundedPostgreSQLRAGManager,
)
from app.composition.postgres_rag_orchestration import (
    PostgreSQLRAGOrchestrationComposition,
)
from app.composition.postgres_rag_runtime import (
    ProfileVerifiedPostgreSQLRAGRunner,
)
from app.integration import RAGCoachingIntegrationDependencies
from app.integration.policy import RAGCoachingIntegrationPolicy
from live_dashboard.demo_data import tenant_demos
from live_dashboard.rag_runtime import (
    DashboardRAGRuntimeController,
    DashboardRAGRuntimeStatus,
)
from live_dashboard.runtime_wiring import DashboardExecutionResourceRegistry

_ACTIVATION_KEYS = {
    "CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH": "provider.json",
    "CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH": "policy.json",
    "CALLMETRIC_DASHBOARD_RAG_MAX_WORKERS": "1",
    "CALLMETRIC_DASHBOARD_RAG_CAPACITY": "2",
}
_PROVIDER = {
    "tenant_id": "tenant_alpha",
    "knowledge_base_id": "kb_alpha",
    "model_id": "synthetic-model",
    "model_name_or_path": "synthetic-model",
    "vector_dimension": 3,
    "normalize_embeddings": True,
    "device": "cpu",
    "local_files_only": True,
}
_POLICY = {
    "rag_llm_enabled_labels": ["risk"],
    "title": "Synthetic guidance",
    "action": "RAG_ACTION",
    "priority": "MEDIUM",
    "label_id": None,
    "expires_after_seconds": None,
}


class _FakeManager:
    def __init__(self) -> None:
        self.close_calls: list[bool] = []

    def close(self, *, wait: bool) -> None:
        self.close_calls.append(wait)


def _tenant_config():
    config = tenant_demos()["tenant_alpha"].config.model_copy(deep=True)
    config.rag.enabled = True
    config.rag.knowledge_base_id = "kb_alpha"
    config.coaching.enable_llm = True
    return config


def _integration(manager: _FakeManager) -> RAGCoachingIntegrationDependencies:
    background_manager = object.__new__(BoundedPostgreSQLRAGManager)
    background_manager.close = manager.close  # type: ignore[method-assign]
    return RAGCoachingIntegrationDependencies(
        background_manager=background_manager,
        policy=RAGCoachingIntegrationPolicy.model_validate(_POLICY),
        suggestion_id_factory=lambda: "synthetic-suggestion",
        utc_datetime_factory=lambda: pytest.fail("timestamp factory was invoked"),
    )


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _configured_environment(tmp_path: Path) -> dict[str, str]:
    provider = _write_json(tmp_path / "provider.json", _PROVIDER)
    policy = _write_json(tmp_path / "policy.json", _POLICY)
    return {
        **_ACTIVATION_KEYS,
        "CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH": str(provider),
        "CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH": str(policy),
    }


def _set_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "CALLMETRIC_POSTGRES_DSN": "postgresql://synthetic:synthetic@localhost/db",
        "CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS": "5",
        "CALLMETRIC_POSTGRES_SSL_MODE": "require",
        "CALLMETRIC_POSTGRES_APPLICATION_NAME": "callmetric-dashboard",
        "CALLMETRIC_VLLM_BASE_URL": "https://synthetic.invalid/v1",
        "CALLMETRIC_VLLM_MODEL_ID": "synthetic-model",
        "CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS": "5",
        "CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS": "30",
        "CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS": "128",
        "CALLMETRIC_VLLM_TEMPERATURE": "0",
        "CALLMETRIC_VLLM_VERIFY_TLS": "true",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_runtime_status_is_fixed_and_immutable() -> None:
    assert [status.value for status in DashboardRAGRuntimeStatus] == [
        "DISABLED",
        "UNAVAILABLE",
        "READY",
    ]
    with pytest.raises((AttributeError, FrozenInstanceError)):
        DashboardRAGRuntimeStatus.READY.value = "secret"  # type: ignore[misc]


def test_all_activation_variables_absent_is_disabled() -> None:
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment={},
    )

    status, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.DISABLED
    assert resource.integration is None


def test_disabled_tenant_flags_do_not_require_configuration() -> None:
    config = _tenant_config()
    config.rag.enabled = False
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment={"CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH": "ignored"},
    )

    status, resource = controller.activate(
        tenant_config=config,
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.DISABLED
    assert resource.integration is None


@pytest.mark.parametrize("missing_key", tuple(_ACTIVATION_KEYS))
def test_partial_activation_configuration_is_unavailable(missing_key: str) -> None:
    environment = dict(_ACTIVATION_KEYS)
    environment.pop(missing_key)
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE
    assert resource.integration is None


@pytest.mark.parametrize(
    ("workers", "capacity"),
    [
        ("0", "1"),
        ("true", "1"),
        ("9", "9"),
        ("2", "1"),
        ("1", "33"),
        (" 1", "2"),
    ],
)
def test_worker_limits_fail_closed(
    tmp_path: Path,
    workers: str,
    capacity: str,
) -> None:
    environment = _configured_environment(tmp_path)
    environment["CALLMETRIC_DASHBOARD_RAG_MAX_WORKERS"] = workers
    environment["CALLMETRIC_DASHBOARD_RAG_CAPACITY"] = capacity
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE
    assert resource.integration is None


@pytest.mark.parametrize(
    "content",
    [
        b"\xef\xbb\xbf{}",
        b'{"tenant_id":"tenant_alpha"}\0',
        b"\xff",
        b"",
    ],
)
def test_invalid_provider_file_encoding_fails_closed(
    tmp_path: Path,
    content: bytes,
) -> None:
    environment = _configured_environment(tmp_path)
    provider = tmp_path / "provider-invalid.json"
    provider.write_bytes(content)
    environment["CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"] = str(provider)
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, _ = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE


@pytest.mark.parametrize(
    "content",
    [
        '{"tenant_id":"tenant_alpha","tenant_id":"other"}',
        '{"secret":"synthetic"}',
        '{"tenant_id":"tenant_alpha"}',
        "[]",
    ],
)
def test_invalid_provider_json_shape_fails_closed(
    tmp_path: Path,
    content: str,
) -> None:
    environment = _configured_environment(tmp_path)
    provider = tmp_path / "provider-invalid.json"
    provider.write_text(content, encoding="utf-8")
    environment["CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"] = str(provider)
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, _ = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE


@pytest.mark.parametrize(
    "content",
    [
        '{"title":"synthetic","title":"duplicate"}',
        '{"api_token":"synthetic"}',
        '{"title":"synthetic"}',
        "[]",
    ],
)
def test_invalid_policy_json_shape_fails_closed(
    tmp_path: Path,
    content: str,
) -> None:
    environment = _configured_environment(tmp_path)
    policy = tmp_path / "policy-invalid.json"
    policy.write_text(content, encoding="utf-8")
    environment["CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH"] = str(policy)
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, _ = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE


def test_directory_and_relative_paths_fail_closed(tmp_path: Path) -> None:
    for raw_path in (str(tmp_path), "provider.json", str(tmp_path / ".." / "x")):
        environment = _configured_environment(tmp_path)
        environment["CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"] = raw_path
        controller = DashboardRAGRuntimeController(
            registry=DashboardExecutionResourceRegistry(capacity=1),
            environment=environment,
        )
        status, _ = controller.activate(
            tenant_config=_tenant_config(),
            call_id=f"synthetic-{len(raw_path)}",
        )
        assert status is DashboardRAGRuntimeStatus.UNAVAILABLE


def test_symlink_configuration_path_fails_closed(tmp_path: Path) -> None:
    environment = _configured_environment(tmp_path)
    target = Path(environment["CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"])
    link = tmp_path / "provider-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    environment["CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"] = str(link)
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, _ = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE


def test_scope_mismatch_fails_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _configured_environment(tmp_path)
    provider = dict(_PROVIDER, tenant_id="other-tenant")
    _write_json(Path(environment[_ACTIVATION_KEYS_KEY(0)]), provider)
    monkeypatch.setattr(
        "app.composition.postgres_rag_orchestration."
        "compose_profile_bound_postgres_rag_orchestration",
        lambda **_kwargs: pytest.fail("composition was invoked"),
    )
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, _ = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE


def test_successful_activation_preserves_exact_construction_order_and_laziness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _configured_environment(tmp_path)
    _set_provider_environment(monkeypatch)
    order: list[str] = []
    composition = object.__new__(PostgreSQLRAGOrchestrationComposition)

    def compose(**kwargs: object) -> PostgreSQLRAGOrchestrationComposition:
        order.append("compose")
        assert callable(kwargs["llm_gateway_factory"])
        assert "embedding_backend_factory" not in kwargs
        return composition

    def prepare(self: ProfileVerifiedPostgreSQLRAGRunner) -> None:
        del self
        order.append("prepare")

    def start(self: BoundedPostgreSQLRAGManager) -> None:
        del self
        order.append("start")

    def close(self: BoundedPostgreSQLRAGManager, *, wait: bool) -> None:
        del self
        assert wait is False
        order.append("close")

    monkeypatch.setattr(
        "app.composition.postgres_rag_orchestration."
        "compose_profile_bound_postgres_rag_orchestration",
        compose,
    )
    monkeypatch.setattr(ProfileVerifiedPostgreSQLRAGRunner, "prepare", prepare)
    monkeypatch.setattr(BoundedPostgreSQLRAGManager, "start", start)
    monkeypatch.setattr(BoundedPostgreSQLRAGManager, "close", close)
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.READY
    assert resource.integration is not None
    assert order == ["compose", "prepare", "start"]
    controller.close_and_remove(resource.opaque_key)
    assert order == ["compose", "prepare", "start", "close"]


def test_manager_start_failure_closes_partial_manager_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _configured_environment(tmp_path)
    _set_provider_environment(monkeypatch)
    composition = object.__new__(PostgreSQLRAGOrchestrationComposition)
    failure = RuntimeError("synthetic-provider-detail")
    close_calls: list[bool] = []
    monkeypatch.setattr(
        "app.composition.postgres_rag_orchestration."
        "compose_profile_bound_postgres_rag_orchestration",
        lambda **_kwargs: composition,
    )
    monkeypatch.setattr(
        ProfileVerifiedPostgreSQLRAGRunner,
        "prepare",
        lambda _self: None,
    )
    monkeypatch.setattr(
        BoundedPostgreSQLRAGManager,
        "start",
        lambda _self: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        BoundedPostgreSQLRAGManager,
        "close",
        lambda _self, *, wait: close_calls.append(wait),
    )
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE
    assert resource.integration is None
    assert close_calls == [False]


def test_same_scope_rerun_reuses_exact_resource_without_reactivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    manager = _FakeManager()

    def activate(**_kwargs: object):
        nonlocal calls
        calls += 1
        return DashboardRAGRuntimeStatus.READY, _integration(manager)

    monkeypatch.setattr(
        "live_dashboard.rag_runtime._activate_optional_integration",
        activate,
    )
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=dict(_ACTIVATION_KEYS),
    )

    first_status, first = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )
    second_status, second = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert first_status is second_status is DashboardRAGRuntimeStatus.READY
    assert second is first
    assert calls == 1


def test_concurrent_same_scope_publishes_one_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    calls_lock = Lock()
    barrier = Barrier(3)
    manager = _FakeManager()

    def activate(**_kwargs: object):
        nonlocal calls
        with calls_lock:
            calls += 1
        return DashboardRAGRuntimeStatus.READY, _integration(manager)

    monkeypatch.setattr(
        "live_dashboard.rag_runtime._activate_optional_integration",
        activate,
    )
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=dict(_ACTIVATION_KEYS),
    )

    def run():
        barrier.wait()
        return controller.activate(
            tenant_config=_tenant_config(),
            call_id="synthetic-call",
        )[1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run) for _ in range(2)]
        barrier.wait()
        first, second = (future.result() for future in futures)

    assert second is first
    assert calls == 1


def test_close_and_remove_closes_manager_once_and_requires_fresh_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(
        "live_dashboard.rag_runtime._activate_optional_integration",
        lambda **_kwargs: (
            DashboardRAGRuntimeStatus.READY,
            _integration(manager),
        ),
    )
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=dict(_ACTIVATION_KEYS),
    )
    _, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    controller.close_and_remove(resource.opaque_key)
    resource.close()

    assert manager.close_calls == [False]


def test_failure_state_does_not_retain_exception_or_sensitive_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive = "synthetic-sensitive-exception"
    environment = _configured_environment(tmp_path)
    environment["CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH"] = sensitive
    controller = DashboardRAGRuntimeController(
        registry=DashboardExecutionResourceRegistry(capacity=1),
        environment=environment,
    )

    status, resource = controller.activate(
        tenant_config=_tenant_config(),
        call_id="synthetic-call",
    )

    assert status is DashboardRAGRuntimeStatus.UNAVAILABLE
    assert resource.integration is None
    captured = capsys.readouterr()
    assert sensitive not in captured.out
    assert sensitive not in captured.err
    assert sensitive not in caplog.text


def _ACTIVATION_KEYS_KEY(index: int) -> str:
    return tuple(_ACTIVATION_KEYS)[index]
