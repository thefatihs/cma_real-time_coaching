"""Lazy local SentenceTransformers query embedding provider."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from threading import Lock
from typing import Literal, Protocol, cast


@dataclass(frozen=True, slots=True)
class SentenceTransformerQueryEmbedderConfig:
    expected_tenant_id: str
    expected_knowledge_base_id: str
    model_name_or_path: str
    device: Literal["cpu", "cuda"]
    normalize_embeddings: bool
    local_files_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_tenant_id",
            _required_text(self.expected_tenant_id, "expected_tenant_id"),
        )
        object.__setattr__(
            self,
            "expected_knowledge_base_id",
            _required_text(
                self.expected_knowledge_base_id,
                "expected_knowledge_base_id",
            ),
        )
        object.__setattr__(
            self,
            "model_name_or_path",
            _required_text(self.model_name_or_path, "model_name_or_path"),
        )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be exactly 'cpu' or 'cuda'")
        if type(self.normalize_embeddings) is not bool:
            raise ValueError("normalize_embeddings must be a boolean")
        if type(self.local_files_only) is not bool:
            raise ValueError("local_files_only must be a boolean")


class SentenceTransformerBackend(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> object: ...


BackendFactory = Callable[
    [SentenceTransformerQueryEmbedderConfig],
    SentenceTransformerBackend,
]


class SentenceTransformerQueryEmbedder:
    def __init__(
        self,
        config: SentenceTransformerQueryEmbedderConfig,
        *,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self._config = config
        if backend_factory is not None and not callable(backend_factory):
            raise ValueError("backend_factory must be callable")
        self._backend_factory = (
            _default_backend_factory if backend_factory is None else backend_factory
        )
        self._backend: SentenceTransformerBackend | None = None
        self._lock = Lock()

    def embed_query(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        text: str,
    ) -> tuple[float, ...]:
        tenant = _required_text(tenant_id, "tenant_id")
        knowledge_base = _required_text(knowledge_base_id, "knowledge_base_id")
        normalized_text = _required_text(text, "text")
        if tenant != self._config.expected_tenant_id:
            raise ValueError("tenant_id does not match query embedder scope")
        if knowledge_base != self._config.expected_knowledge_base_id:
            raise ValueError("knowledge_base_id does not match query embedder scope")

        with self._lock:
            if self._backend is None:
                self._backend = self._backend_factory(self._config)
            output = self._backend.encode(
                [normalized_text],
                normalize_embeddings=self._config.normalize_embeddings,
            )
        return _single_vector(output)


class _SentenceTransformerBackendAdapter:
    def __init__(self, model: object) -> None:
        self._model = model

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> object:
        encode = getattr(self._model, "encode", None)
        if not callable(encode):
            raise RuntimeError("SentenceTransformer model does not expose encode")
        callable_encode = cast(Callable[..., object], encode)
        return callable_encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def _default_backend_factory(
    config: SentenceTransformerQueryEmbedderConfig,
) -> SentenceTransformerBackend:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name_or_path=config.model_name_or_path,
        device=config.device,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
    )
    return _SentenceTransformerBackendAdapter(model)


def _single_vector(output: object) -> tuple[float, ...]:
    to_list = getattr(output, "tolist", None)
    if callable(to_list):
        output = to_list()
    if (
        isinstance(output, (str, bytes))
        or not isinstance(output, Sequence)
        or len(output) != 1
    ):
        raise ValueError("embedding backend must return exactly one row")
    row = output[0]
    row_to_list = getattr(row, "tolist", None)
    if callable(row_to_list):
        row = row_to_list()
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
        raise ValueError("embedding backend row must be a vector")
    if not row:
        raise ValueError("embedding backend vector cannot be empty")

    values: list[float] = []
    for item in row:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("embedding backend vector must contain real numbers")
        value = float(item)
        if not isfinite(value):
            raise ValueError("embedding backend vector must contain finite values")
        values.append(value)
    return tuple(values)


def _required_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned
