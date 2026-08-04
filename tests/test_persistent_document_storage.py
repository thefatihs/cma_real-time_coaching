from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import pytest

import app.ingestion.persistent_storage as storage_module
from app.ingestion.persistent_storage import (
    MAX_RECONCILE_BATCH_SIZE,
    OrphanReconciliationResult,
    PersistentDocumentStorage,
    PersistentStorageError,
    PersistentStorageFailure,
    PosixStoragePrivacyValidator,
    RegistryStorageKeySnapshot,
    StorageDeleteOutcome,
    WindowsDirectorySecurity,
    WindowsStoragePrivacyValidator,
    cleanup_duplicate_source,
    delete_document_then_source,
    store_then_create_document,
)
from app.ingestion.registry_models import (
    DocumentDeletionResult,
    DocumentIngestionPhase,
    DocumentRegistryCreateRequest,
    DocumentRegistryCreateResult,
)
from app.ingestion.upload_preparation import MAX_UPLOAD_BYTES


class AcceptPrivacy:
    def validate(self, root: Path) -> None:
        assert root.is_absolute()


class FakeWindowsInspector:
    def __init__(self, result: WindowsDirectorySecurity) -> None:
        self.result = result

    def inspect(self, root: Path) -> WindowsDirectorySecurity:
        assert root.is_absolute()
        return self.result


def storage(
    tmp_path: Path,
    *,
    key_factory=lambda: "a" * 32,
    clock=lambda: 10_000.0,
) -> PersistentDocumentStorage:
    repository = tmp_path / "repository"
    profile = tmp_path / "profile"
    root = tmp_path / "private-storage"
    repository.mkdir(exist_ok=True)
    profile.mkdir(exist_ok=True)
    root.mkdir(exist_ok=True)
    return PersistentDocumentStorage(
        root=root,
        repository_root=repository,
        user_profile=profile,
        privacy_validator=AcceptPrivacy(),
        key_factory=key_factory,
        clock=clock,
    )


@pytest.mark.parametrize(
    "security",
    (
        WindowsDirectorySecurity(False, True, False),
        WindowsDirectorySecurity(True, False, False),
        WindowsDirectorySecurity(True, True, True),
    ),
)
def test_windows_acl_validator_rejects_non_private_results(
    tmp_path: Path,
    security: WindowsDirectorySecurity,
) -> None:
    validator = WindowsStoragePrivacyValidator(FakeWindowsInspector(security))
    with pytest.raises(PersistentStorageError) as caught:
        validator.validate(tmp_path)
    assert caught.value.failure is PersistentStorageFailure.STORAGE_PERMISSION


def test_windows_acl_validator_accepts_current_owner_only(tmp_path: Path) -> None:
    validator = WindowsStoragePrivacyValidator(
        FakeWindowsInspector(WindowsDirectorySecurity(True, True, False))
    )
    validator.validate(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership and mode test")
def test_posix_root_requires_owner_only_mode(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    PosixStoragePrivacyValidator().validate(tmp_path)
    tmp_path.chmod(0o750)
    with pytest.raises(PersistentStorageError):
        PosixStoragePrivacyValidator().validate(tmp_path)


def test_root_must_be_existing_absolute_and_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    inside = repository / "storage"
    inside.mkdir()
    with pytest.raises(PersistentStorageError) as caught:
        PersistentDocumentStorage(
            root=inside,
            repository_root=repository,
            user_profile=tmp_path / "profile",
            privacy_validator=AcceptPrivacy(),
        )
    assert caught.value.failure is PersistentStorageFailure.STORAGE_CONFIGURATION
    assert str(inside) not in str(caught.value)


def test_symlink_storage_component_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PersistentStorageError):
        PersistentDocumentStorage(
            root=link,
            repository_root=repository,
            user_profile=tmp_path / "profile",
            privacy_validator=AcceptPrivacy(),
        )


def test_mocked_windows_reparse_component_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    root = tmp_path / "private-storage"
    repository.mkdir()
    root.mkdir()
    original_lstat = Path.lstat

    def lstat(path: Path):
        status = original_lstat(path)
        if path == root:
            return SimpleNamespace(
                st_mode=status.st_mode,
                st_file_attributes=0x400,
            )
        return status

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(PersistentStorageError) as caught:
        PersistentDocumentStorage(
            root=root,
            repository_root=repository,
            user_profile=tmp_path,
            privacy_validator=AcceptPrivacy(),
        )
    assert caught.value.failure is PersistentStorageFailure.STORAGE_CONFIGURATION


def test_atomic_write_bounded_read_and_idempotent_delete(tmp_path: Path) -> None:
    subject = storage(tmp_path)
    content = b"synthetic document"
    key = subject.write(content)
    assert key == "obj_" + "a" * 32
    assert subject.read(key) == content
    assert subject.delete(key) is StorageDeleteOutcome.DELETED
    assert subject.delete(key) is StorageDeleteOutcome.NOT_FOUND


@pytest.mark.parametrize(
    "key",
    ("", "/absolute", "../escape", "obj_" + "a" * 31, "obj_" + "g" * 32, "x\x00y"),
)
def test_read_and_delete_reject_untrusted_keys_without_echo(
    tmp_path: Path, key: str
) -> None:
    subject = storage(tmp_path)
    with pytest.raises(PersistentStorageError) as read_error:
        subject.read(key)
    with pytest.raises(PersistentStorageError) as delete_error:
        subject.delete(key)
    if key:
        assert key not in str(read_error.value)
        assert key not in str(delete_error.value)


def test_capacity_and_existing_object_never_overwrite_or_leave_temporary(
    tmp_path: Path,
) -> None:
    subject = storage(tmp_path)
    root = tmp_path / "private-storage"
    existing = root / ("obj_" + "a" * 32)
    existing.write_bytes(b"existing")
    with pytest.raises(PersistentStorageError) as collision:
        subject.write(b"new")
    assert collision.value.failure is PersistentStorageFailure.STORAGE_WRITE
    assert existing.read_bytes() == b"existing"
    assert not tuple(root.glob(".tmp_*"))
    with pytest.raises(PersistentStorageError) as capacity:
        subject.write(b"x" * (MAX_UPLOAD_BYTES + 1))
    assert capacity.value.failure is PersistentStorageFailure.STORAGE_CAPACITY


def test_publish_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = storage(tmp_path)
    monkeypatch.setattr(
        storage_module,
        "_publish_exclusive",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
    )
    with pytest.raises(PersistentStorageError) as caught:
        subject.write(b"synthetic")
    assert caught.value.failure is PersistentStorageFailure.STORAGE_WRITE
    assert not tuple((tmp_path / "private-storage").glob(".tmp_*"))


def test_bounded_read_rejects_oversized_changed_object(tmp_path: Path) -> None:
    subject = storage(tmp_path)
    path = tmp_path / "private-storage" / ("obj_" + "a" * 32)
    path.write_bytes(b"x" * (MAX_UPLOAD_BYTES + 1))
    with pytest.raises(PersistentStorageError) as caught:
        subject.read(path.name)
    assert caught.value.failure is PersistentStorageFailure.STORAGE_READ


def test_reconcile_is_fail_closed_and_preserves_unrelated_files(tmp_path: Path) -> None:
    keys = iter(("a" * 32, "b" * 32))
    subject = storage(tmp_path, key_factory=lambda: next(keys))
    owned = subject.write(b"owned")
    orphan = subject.write(b"orphan")
    root = tmp_path / "private-storage"
    unrelated = root / "operator-note.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    os.utime(root / owned, (1, 1))
    os.utime(root / orphan, (1, 1))

    incomplete = subject.reconcile_orphans(
        RegistryStorageKeySnapshot(frozenset(), complete=False),
        grace_seconds=300,
    )
    assert incomplete == OrphanReconciliationResult(0, 0, 0, 0)
    assert (root / orphan).exists()

    result = subject.reconcile_orphans(
        RegistryStorageKeySnapshot(frozenset({owned}), complete=True),
        grace_seconds=300,
    )
    assert result == OrphanReconciliationResult(2, 1, 1, 1)
    assert (root / owned).exists()
    assert unrelated.read_text(encoding="utf-8") == "unrelated"


def test_reconcile_honors_grace_and_deterministic_batch_bound(tmp_path: Path) -> None:
    values = iter(f"{index:032x}" for index in range(MAX_RECONCILE_BATCH_SIZE + 2))
    subject = storage(tmp_path, key_factory=lambda: next(values), clock=lambda: 1_000.0)
    root = tmp_path / "private-storage"
    for _ in range(MAX_RECONCILE_BATCH_SIZE + 2):
        key = subject.write(b"orphan")
        os.utime(root / key, (1, 1))
    result = subject.reconcile_orphans(
        RegistryStorageKeySnapshot(frozenset(), complete=True),
        grace_seconds=300,
    )
    assert result.eligible_orphans == MAX_RECONCILE_BATCH_SIZE + 2
    assert result.deleted_objects == MAX_RECONCILE_BATCH_SIZE
    assert len(tuple(root.glob("obj_*"))) == 2


@pytest.mark.parametrize(
    ("grace_seconds", "batch_size"),
    ((299, 1), (604_801, 1), (300, 0), (300, 101), (True, 1)),
)
def test_reconcile_rejects_invalid_grace_and_batch_bounds(
    tmp_path: Path, grace_seconds: object, batch_size: object
) -> None:
    with pytest.raises(PersistentStorageError) as caught:
        storage(tmp_path).reconcile_orphans(
            RegistryStorageKeySnapshot(frozenset(), complete=True),
            grace_seconds=grace_seconds,  # type: ignore[arg-type]
            batch_size=batch_size,  # type: ignore[arg-type]
        )
    assert caught.value.failure is PersistentStorageFailure.RECONCILE


def test_duplicate_cleanup_removes_only_new_object(tmp_path: Path) -> None:
    subject = storage(tmp_path)
    new_key = subject.write(b"duplicate")
    result = DocumentRegistryCreateResult.model_construct(
        created=False,
        entry=SimpleNamespace(
            document=SimpleNamespace(storage_object_key="obj_" + "b" * 32)
        ),
    )
    cleanup_duplicate_source(
        subject,
        newly_created_key=new_key,
        registry_result=result,
    )
    assert subject.delete(new_key) is StorageDeleteOutcome.NOT_FOUND


def test_storage_precedes_registry_and_duplicate_cleanup(tmp_path: Path) -> None:
    subject = storage(tmp_path)
    order: list[str] = []
    duplicate = DocumentRegistryCreateResult.model_construct(
        created=False,
        entry=SimpleNamespace(
            document=SimpleNamespace(storage_object_key="obj_" + "b" * 32)
        ),
    )

    class Repository:
        def create_or_get(
            self, request: DocumentRegistryCreateRequest
        ) -> DocumentRegistryCreateResult:
            order.append("registry")
            assert (tmp_path / "private-storage" / request.storage_object_key).is_file()
            return duplicate

    def request_factory(key: str) -> DocumentRegistryCreateRequest:
        order.append("storage")
        return DocumentRegistryCreateRequest(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_id="doc-a",
            job_id="job-a",
            original_filename="guide.txt",
            media_type="text/plain",
            byte_size=9,
            sha256_hex="a" * 64,
            storage_object_key=key,
            total_chunks=0,
            initial_phase=DocumentIngestionPhase.VALIDATION,
        )

    result = store_then_create_document(
        subject,
        Repository(),  # type: ignore[arg-type]
        content=b"synthetic",
        request_factory=request_factory,
    )
    assert result is duplicate
    assert order == ["storage", "registry"]
    assert not tuple((tmp_path / "private-storage").glob("obj_*"))


def test_database_delete_precedes_source_delete(tmp_path: Path) -> None:
    order: list[str] = []

    class Repository:
        def delete_document(self, **scope: str) -> DocumentDeletionResult:
            order.append("database")
            return DocumentDeletionResult(storage_object_key="obj_" + "a" * 32)

    class RecordingStorage(PersistentDocumentStorage):
        def delete(self, key: str) -> StorageDeleteOutcome:
            order.append("storage")
            return super().delete(key)

    base = storage(tmp_path)
    base.write(b"source")
    subject = RecordingStorage(
        root=tmp_path / "private-storage",
        repository_root=tmp_path / "repository",
        user_profile=tmp_path / "profile",
        privacy_validator=AcceptPrivacy(),
    )
    assert delete_document_then_source(
        Repository(),  # type: ignore[arg-type]
        subject,
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id="doc-a",
    )
    assert order == ["database", "storage"]
