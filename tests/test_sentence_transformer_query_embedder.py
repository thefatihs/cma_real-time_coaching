from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, MISSING, fields
from math import inf, nan
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Literal, cast

import numpy as np
import pytest

from app.embeddings import (
    QueryEmbedder,
    SentenceTransformerBackend,
    SentenceTransformerQueryEmbedder,
    SentenceTransformerQueryEmbedderConfig,
)
from app.embeddings.sentence_transformers import BackendFactory


class FakeBackend:
    def __init__(
        self,
        output: object = ((0.25, 0.75),),
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> object:
        self.calls.append((list(texts), normalize_embeddings))
        if self.error is not None:
            raise self.error
        return self.output


class FactorySpy:
    def __init__(
        self,
        backend: SentenceTransformerBackend,
        *,
        error: Exception | None = None,
    ) -> None:
        self.backend = backend
        self.error = error
        self.calls: list[SentenceTransformerQueryEmbedderConfig] = []

    def __call__(
        self,
        config: SentenceTransformerQueryEmbedderConfig,
    ) -> SentenceTransformerBackend:
        self.calls.append(config)
        if self.error is not None:
            raise self.error
        return self.backend


class FalseyFactory(FactorySpy):
    def __bool__(self) -> bool:
        return False


def config(
    *,
    expected_tenant_id: str = "tenant_alpha",
    expected_knowledge_base_id: str = "kb_support",
    model_name_or_path: str = "local_models/synthetic-embedding",
    device: Literal["cpu", "cuda"] = "cpu",
    normalize_embeddings: bool = True,
    local_files_only: bool = True,
) -> SentenceTransformerQueryEmbedderConfig:
    return SentenceTransformerQueryEmbedderConfig(
        expected_tenant_id=expected_tenant_id,
        expected_knowledge_base_id=expected_knowledge_base_id,
        model_name_or_path=model_name_or_path,
        device=device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
    )


def embed(
    provider: QueryEmbedder,
    *,
    tenant_id: str = "tenant_alpha",
    knowledge_base_id: str = "kb_support",
    text: str = "Synthetic query.",
) -> tuple[float, ...]:
    return provider.embed_query(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        text=text,
    )


def provider(
    *,
    selected_config: SentenceTransformerQueryEmbedderConfig | None = None,
    output: object = ((0.25, 0.75),),
) -> tuple[SentenceTransformerQueryEmbedder, FakeBackend, FactorySpy]:
    backend = FakeBackend(output)
    factory = FactorySpy(backend)
    subject: QueryEmbedder = SentenceTransformerQueryEmbedder(
        selected_config or config(),
        backend_factory=factory,
    )
    assert isinstance(subject, SentenceTransformerQueryEmbedder)
    return subject, backend, factory


def test_configuration_is_frozen_slotted_and_all_fields_are_required() -> None:
    subject = config()

    assert subject.__slots__ == (
        "expected_tenant_id",
        "expected_knowledge_base_id",
        "model_name_or_path",
        "device",
        "normalize_embeddings",
        "local_files_only",
    )
    assert all(item.default is MISSING for item in fields(type(subject)))
    assert all(item.default_factory is MISSING for item in fields(type(subject)))
    with pytest.raises(FrozenInstanceError):
        subject.device = "cuda"  # type: ignore[misc]


def test_configuration_normalizes_scope_and_model_path() -> None:
    subject = config(
        expected_tenant_id=" tenant_alpha ",
        expected_knowledge_base_id=" kb_support ",
        model_name_or_path=" local_models/synthetic-embedding ",
    )

    assert subject.expected_tenant_id == "tenant_alpha"
    assert subject.expected_knowledge_base_id == "kb_support"
    assert subject.model_name_or_path == "local_models/synthetic-embedding"


@pytest.mark.parametrize(
    ("expected_tenant_id", "expected_knowledge_base_id", "model_name_or_path"),
    [
        (" ", "kb_support", "local_models/synthetic"),
        ("tenant_alpha", " ", "local_models/synthetic"),
        ("tenant_alpha", "kb_support", " "),
    ],
)
def test_blank_configuration_text_is_rejected(
    expected_tenant_id: str,
    expected_knowledge_base_id: str,
    model_name_or_path: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        config(
            expected_tenant_id=expected_tenant_id,
            expected_knowledge_base_id=expected_knowledge_base_id,
            model_name_or_path=model_name_or_path,
        )


@pytest.mark.parametrize("device", ["auto", "cuda:0", "mps", "CPU"])
def test_unsupported_device_is_rejected(device: str) -> None:
    with pytest.raises(ValueError, match="device"):
        config(device=cast(Literal["cpu", "cuda"], device))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("normalize_embeddings", 1),
        ("local_files_only", "true"),
    ],
)
def test_configuration_flags_must_be_explicit_booleans(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        SentenceTransformerQueryEmbedderConfig(
            expected_tenant_id="tenant_alpha",
            expected_knowledge_base_id="kb_support",
            model_name_or_path="local_models/synthetic",
            device="cpu",
            normalize_embeddings=cast(bool, value)
            if field_name == "normalize_embeddings"
            else True,
            local_files_only=cast(bool, value)
            if field_name == "local_files_only"
            else True,
        )


def test_constructor_does_not_construct_backend() -> None:
    subject, backend, factory = provider()

    assert subject is not None
    assert factory.calls == []
    assert backend.calls == []


def test_falsey_callable_factory_is_preserved() -> None:
    backend = FakeBackend()
    factory = FalseyFactory(backend)
    subject = SentenceTransformerQueryEmbedder(config(), backend_factory=factory)

    assert factory.calls == []
    assert embed(subject) == (0.25, 0.75)
    assert factory.calls == [config()]
    assert backend.calls == [(["Synthetic query."], True)]


def test_non_callable_factory_is_rejected_during_construction() -> None:
    with pytest.raises(ValueError, match="backend_factory must be callable"):
        SentenceTransformerQueryEmbedder(
            config(),
            backend_factory=cast(BackendFactory, object()),
        )


@pytest.mark.parametrize(
    ("tenant_id", "knowledge_base_id", "text", "message"),
    [
        ("tenant_beta", "kb_support", "Synthetic query.", "tenant_id"),
        ("tenant_alpha", "kb_other", "Synthetic query.", "knowledge_base_id"),
        ("tenant_alpha", "kb_support", " ", "text"),
    ],
)
def test_invalid_scope_or_query_fails_before_factory(
    tenant_id: str,
    knowledge_base_id: str,
    text: str,
    message: str,
) -> None:
    subject, _, factory = provider()

    with pytest.raises(ValueError, match=message):
        embed(
            subject,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            text=text,
        )

    assert factory.calls == []


def test_first_call_loads_once_and_repeated_calls_reuse_backend() -> None:
    subject, backend, factory = provider()

    first = embed(subject)
    second = embed(subject)

    assert first == second == (0.25, 0.75)
    assert factory.calls == [config()]
    assert backend.calls == [
        (["Synthetic query."], True),
        (["Synthetic query."], True),
    ]


def test_query_and_normalization_flag_are_propagated_exactly() -> None:
    subject, backend, _ = provider(selected_config=config(normalize_embeddings=False))

    result = embed(subject, text="  Synthetic normalized query.  ")

    assert result == (0.25, 0.75)
    assert backend.calls == [(["Synthetic normalized query."], False)]


def test_concurrent_first_calls_construct_backend_once() -> None:
    subject, backend, factory = provider()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: embed(subject), range(16)))

    assert results == ((0.25, 0.75),) * 16
    assert factory.calls == [config()]
    assert len(backend.calls) == 16


@pytest.mark.parametrize(
    "output",
    [
        [[0.25, 0.75]],
        ((0.25, 0.75),),
        np.array([[0.25, 0.75]], dtype=np.float32),
    ],
)
def test_valid_one_row_output_converts_to_tuple(output: object) -> None:
    subject, _, _ = provider(output=output)

    assert embed(subject) == (0.25, 0.75)


@pytest.mark.parametrize(
    "output",
    [
        0.25,
        [0.25, 0.75],
        [],
        [[0.25], [0.75]],
        [[]],
        [[[0.25, 0.75]]],
        [[0.25, [0.75]]],
    ],
)
def test_invalid_output_shape_is_rejected(output: object) -> None:
    subject, _, _ = provider(output=output)

    with pytest.raises(ValueError):
        embed(subject)


@pytest.mark.parametrize(
    "value",
    [True, "0.25", object(), 1 + 2j, nan, inf, -inf],
)
def test_invalid_vector_values_are_rejected(value: object) -> None:
    subject, _, _ = provider(output=[[value]])

    with pytest.raises(ValueError):
        embed(subject)


def test_backend_output_is_not_mutated() -> None:
    output = [[0.25, 0.75]]
    subject, _, _ = provider(output=output)

    assert embed(subject) == (0.25, 0.75)
    assert output == [[0.25, 0.75]]


def test_factory_and_encode_exceptions_propagate_unchanged() -> None:
    factory_error = RuntimeError("synthetic factory failure")
    factory = FactorySpy(FakeBackend(), error=factory_error)
    subject = SentenceTransformerQueryEmbedder(config(), backend_factory=factory)
    with pytest.raises(RuntimeError, match="factory failure"):
        embed(subject)

    encode_error = RuntimeError("synthetic encode failure")
    backend = FakeBackend(error=encode_error)
    subject = SentenceTransformerQueryEmbedder(
        config(),
        backend_factory=FactorySpy(backend),
    )
    with pytest.raises(RuntimeError, match="encode failure"):
        embed(subject)


@pytest.mark.parametrize(
    ("device", "local_files_only"),
    [("cpu", True), ("cuda", False)],
)
def test_default_factory_propagates_explicit_safe_model_arguments(
    monkeypatch: pytest.MonkeyPatch,
    device: Literal["cpu", "cuda"],
    local_files_only: bool,
) -> None:
    constructor_calls: list[dict[str, object]] = []

    class FakeModel:
        def encode(
            self,
            texts: list[str],
            **options: object,
        ) -> list[list[float]]:
            assert texts == ["Synthetic query."]
            assert options == {
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            }
            return [[0.25, 0.75]]

    def sentence_transformer(**options: object) -> FakeModel:
        constructor_calls.append(options)
        return FakeModel()

    fake_module = ModuleType("sentence_transformers")
    setattr(fake_module, "SentenceTransformer", sentence_transformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    subject = SentenceTransformerQueryEmbedder(
        config(device=device, local_files_only=local_files_only)
    )

    assert embed(subject) == (0.25, 0.75)
    assert constructor_calls == [
        {
            "model_name_or_path": "local_models/synthetic-embedding",
            "device": device,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
    ]


def test_provider_module_import_does_not_load_model_stack() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "before=set(sys.modules)\n"
        "import app.embeddings.sentence_transformers\n"
        "after=set(sys.modules)\n"
        "blocked={'sentence_transformers','transformers','torch'}\n"
        "loaded=sorted(blocked & (after-before))\n"
        "raise SystemExit(f'loaded prohibited modules: {loaded}' if loaded else 0)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
