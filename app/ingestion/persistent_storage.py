"""Private persistent storage and bounded orphan reconciliation primitives."""

from __future__ import annotations

import ctypes
import os
import stat
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from app.ingestion.registry import DocumentRegistryRepository
from app.ingestion.registry_models import (
    DocumentRegistryCreateRequest,
    DocumentRegistryCreateResult,
)
from app.ingestion.upload_preparation import MAX_UPLOAD_BYTES

MIN_ORPHAN_GRACE_SECONDS = 300
MAX_ORPHAN_GRACE_SECONDS = 604_800
MAX_RECONCILE_BATCH_SIZE = 100
_MANAGED_PREFIX = "obj_"
_TEMPORARY_PREFIX = ".tmp_"
_KEY_HEX_CHARACTERS = 32
_GENERIC_ALL = 0x10000000
_GENERIC_WRITE = 0x40000000
_WRITE_ACCESS_MASK = (
    0x0002
    | 0x0004
    | 0x0040
    | 0x0100
    | 0x00010000
    | 0x00040000
    | _GENERIC_ALL
    | _GENERIC_WRITE
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class PersistentStorageFailure(str, Enum):
    STORAGE_CONFIGURATION = "STORAGE_CONFIGURATION"
    STORAGE_PERMISSION = "STORAGE_PERMISSION"
    STORAGE_WRITE = "STORAGE_WRITE"
    STORAGE_READ = "STORAGE_READ"
    STORAGE_DELETE = "STORAGE_DELETE"
    STORAGE_CAPACITY = "STORAGE_CAPACITY"
    RECONCILE = "RECONCILE"


_FAILURE_MESSAGES = {
    failure: f"Document storage failed during {failure.value}."
    for failure in PersistentStorageFailure
}


class PersistentStorageError(RuntimeError):
    """Fixed secret-free storage failure."""

    def __init__(self, failure: PersistentStorageFailure) -> None:
        self.failure = failure
        super().__init__(_FAILURE_MESSAGES[failure])


class StorageDeleteOutcome(str, Enum):
    DELETED = "DELETED"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True, slots=True)
class RegistryStorageKeySnapshot:
    keys: frozenset[str]
    complete: bool


@dataclass(frozen=True, slots=True)
class OrphanReconciliationResult:
    scanned_objects: int
    eligible_orphans: int
    deleted_objects: int
    retained_objects: int


class StoragePrivacyValidator(Protocol):
    def validate(self, root: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class WindowsDirectorySecurity:
    owner_is_current_user: bool
    current_user_has_control: bool
    unrelated_principal_has_write_access: bool


class WindowsSecurityInspector(Protocol):
    def inspect(self, root: Path) -> WindowsDirectorySecurity: ...


class PosixStoragePrivacyValidator:
    def validate(self, root: Path) -> None:
        try:
            status = root.stat(follow_symlinks=False)
            get_effective_user_id = getattr(os, "geteuid", None)
            if (
                not callable(get_effective_user_id)
                or status.st_uid != get_effective_user_id()
                or stat.S_IMODE(status.st_mode) & 0o077
            ):
                raise PersistentStorageError(
                    PersistentStorageFailure.STORAGE_PERMISSION
                )
        except PersistentStorageError:
            raise
        except Exception:
            raise PersistentStorageError(
                PersistentStorageFailure.STORAGE_PERMISSION
            ) from None


class WindowsStoragePrivacyValidator:
    def __init__(self, inspector: WindowsSecurityInspector | None = None) -> None:
        self._inspector = inspector or CtypesWindowsSecurityInspector()

    def validate(self, root: Path) -> None:
        try:
            security = self._inspector.inspect(root)
            if (
                not security.owner_is_current_user
                or not security.current_user_has_control
                or security.unrelated_principal_has_write_access
            ):
                raise PersistentStorageError(
                    PersistentStorageFailure.STORAGE_PERMISSION
                )
        except PersistentStorageError:
            raise
        except Exception:
            raise PersistentStorageError(
                PersistentStorageFailure.STORAGE_PERMISSION
            ) from None


class CtypesWindowsSecurityInspector:
    """Inspect owner and allow ACEs using locale-independent Windows APIs."""

    def inspect(self, root: Path) -> WindowsDirectorySecurity:
        if os.name != "nt":
            raise OSError("Windows security APIs are unavailable")
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        result = advapi32.GetNamedSecurityInfoW(
            ctypes.c_wchar_p(str(root)),
            1,
            0x00000001 | 0x00000004,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if result != 0 or not owner.value or not dacl.value or not descriptor.value:
            if descriptor.value:
                kernel32.LocalFree(descriptor)
            raise OSError("security descriptor inspection failed")
        try:
            current_sid = _current_windows_user_sid(advapi32, kernel32, wintypes)
            owner_sid = _sid_bytes(advapi32, owner)
            writable_sids, current_control = _allowed_windows_writers(
                advapi32,
                dacl,
                current_sid,
            )
            return WindowsDirectorySecurity(
                owner_is_current_user=owner_sid == current_sid,
                current_user_has_control=current_control,
                unrelated_principal_has_write_access=any(
                    sid != current_sid for sid in writable_sids
                ),
            )
        finally:
            kernel32.LocalFree(descriptor)


class PersistentDocumentStorage:
    """Store bounded source bytes under opaque direct-child object keys."""

    def __init__(
        self,
        *,
        root: Path,
        repository_root: Path = _REPOSITORY_ROOT,
        user_profile: Path | None = None,
        privacy_validator: StoragePrivacyValidator | None = None,
        key_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        try:
            self._root = _validated_storage_root(
                root,
                repository_root=repository_root,
                user_profile=Path.home() if user_profile is None else user_profile,
            )
        except PersistentStorageError:
            raise
        except Exception:
            raise PersistentStorageError(
                PersistentStorageFailure.STORAGE_CONFIGURATION
            ) from None
        validator = privacy_validator or (
            WindowsStoragePrivacyValidator()
            if os.name == "nt"
            else PosixStoragePrivacyValidator()
        )
        validator.validate(self._root)
        self._privacy_validator = validator
        self._key_factory = key_factory or (lambda: uuid4().hex)
        self._clock = clock or __import__("time").time

    def write(self, content: bytes) -> str:
        if type(content) is not bytes:
            raise PersistentStorageError(PersistentStorageFailure.STORAGE_WRITE)
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise PersistentStorageError(PersistentStorageFailure.STORAGE_CAPACITY)
        for _attempt in range(8):
            key = _new_key(self._key_factory)
            final_path = self._path_for_key(key, PersistentStorageFailure.STORAGE_WRITE)
            temporary_path = self._root / f"{_TEMPORARY_PREFIX}{uuid4().hex}"
            descriptor: int | None = None
            published = False
            try:
                descriptor = os.open(
                    temporary_path,
                    _write_flags(),
                    0o600,
                )
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    descriptor = None
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                _set_owner_only_mode(temporary_path)
                _publish_exclusive(temporary_path, final_path)
                published = True
                _validate_regular_object(
                    final_path, PersistentStorageFailure.STORAGE_WRITE
                )
                self._privacy_validator.validate(final_path)
                return key
            except FileExistsError:
                _close_descriptor(descriptor)
                _unlink_temporary(temporary_path)
                if published:
                    _unlink_temporary(final_path)
                continue
            except Exception:
                _close_descriptor(descriptor)
                _unlink_temporary(temporary_path)
                if published:
                    _unlink_temporary(final_path)
                raise PersistentStorageError(
                    PersistentStorageFailure.STORAGE_WRITE
                ) from None
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_WRITE)

    def read(self, key: str) -> bytes:
        path = self._path_for_key(key, PersistentStorageFailure.STORAGE_READ)
        descriptor: int | None = None
        try:
            before = path.stat(follow_symlinks=False)
            _validate_object_status(before, PersistentStorageFailure.STORAGE_READ)
            self._privacy_validator.validate(path)
            descriptor = _open_read_descriptor(path)
            opened = os.fstat(descriptor)
            _validate_object_status(opened, PersistentStorageFailure.STORAGE_READ)
            if _status_identity(before) != _status_identity(opened):
                raise PersistentStorageError(PersistentStorageFailure.STORAGE_READ)
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = None
                content = stream.read(MAX_UPLOAD_BYTES + 1)
                after = os.fstat(stream.fileno())
            if (
                len(content) > MAX_UPLOAD_BYTES
                or len(content) != opened.st_size
                or _status_identity(opened) != _status_identity(after)
            ):
                raise PersistentStorageError(PersistentStorageFailure.STORAGE_READ)
            return content
        except PersistentStorageError:
            _close_descriptor(descriptor)
            raise
        except Exception:
            _close_descriptor(descriptor)
            raise PersistentStorageError(
                PersistentStorageFailure.STORAGE_READ
            ) from None

    def delete(self, key: str) -> StorageDeleteOutcome:
        path = self._path_for_key(key, PersistentStorageFailure.STORAGE_DELETE)
        try:
            try:
                status = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                return StorageDeleteOutcome.NOT_FOUND
            _validate_object_status(status, PersistentStorageFailure.STORAGE_DELETE)
            self._privacy_validator.validate(path)
            path.unlink()
            return StorageDeleteOutcome.DELETED
        except PersistentStorageError:
            raise
        except FileNotFoundError:
            return StorageDeleteOutcome.NOT_FOUND
        except Exception:
            raise PersistentStorageError(
                PersistentStorageFailure.STORAGE_DELETE
            ) from None

    def reconcile_orphans(
        self,
        snapshot: RegistryStorageKeySnapshot,
        *,
        grace_seconds: int,
        batch_size: int = MAX_RECONCILE_BATCH_SIZE,
    ) -> OrphanReconciliationResult:
        if (
            not isinstance(snapshot, RegistryStorageKeySnapshot)
            or type(snapshot.complete) is not bool
            or type(grace_seconds) is not int
            or not MIN_ORPHAN_GRACE_SECONDS <= grace_seconds <= MAX_ORPHAN_GRACE_SECONDS
            or type(batch_size) is not int
            or not 1 <= batch_size <= MAX_RECONCILE_BATCH_SIZE
        ):
            raise PersistentStorageError(PersistentStorageFailure.RECONCILE)
        if not snapshot.complete:
            return OrphanReconciliationResult(0, 0, 0, 0)
        try:
            owned = frozenset(_validated_key(key) for key in snapshot.keys)
            now = float(self._clock())
            if not isfinite(now):
                raise ValueError
            managed = _managed_direct_children(self._root)
            for item in managed:
                self._privacy_validator.validate(self._root / item.name)
            eligible = tuple(
                item
                for item in managed
                if item.name not in owned and now - item.modified_at >= grace_seconds
            )
            selected = eligible[:batch_size]
            deleted = 0
            for item in selected:
                if item.temporary:
                    _delete_managed_temporary(self._root, item.name)
                else:
                    outcome = self.delete(item.name)
                    if outcome is StorageDeleteOutcome.NOT_FOUND:
                        continue
                deleted += 1
            return OrphanReconciliationResult(
                scanned_objects=len(managed),
                eligible_orphans=len(eligible),
                deleted_objects=deleted,
                retained_objects=len(managed) - deleted,
            )
        except PersistentStorageError as error:
            if error.failure is PersistentStorageFailure.RECONCILE:
                raise
            raise PersistentStorageError(PersistentStorageFailure.RECONCILE) from None
        except Exception:
            raise PersistentStorageError(PersistentStorageFailure.RECONCILE) from None

    def _path_for_key(
        self,
        key: object,
        failure: PersistentStorageFailure,
    ) -> Path:
        try:
            validated = _validated_key(key)
            path = self._root / validated
            if path.parent != self._root:
                raise ValueError
            return path
        except Exception:
            raise PersistentStorageError(failure) from None


def cleanup_duplicate_source(
    storage: PersistentDocumentStorage,
    *,
    newly_created_key: str,
    registry_result: DocumentRegistryCreateResult,
) -> None:
    """Remove only this attempt's object when the registry resolved a duplicate."""
    if not isinstance(storage, PersistentDocumentStorage) or not isinstance(
        registry_result, DocumentRegistryCreateResult
    ):
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_DELETE)
    if registry_result.created:
        return
    if newly_created_key == registry_result.entry.document.storage_object_key:
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_DELETE)
    storage.delete(newly_created_key)


def store_then_create_document(
    storage: PersistentDocumentStorage,
    repository: DocumentRegistryRepository,
    *,
    content: bytes,
    request_factory: Callable[[str], DocumentRegistryCreateRequest],
) -> DocumentRegistryCreateResult:
    """Persist bytes first, then create the registry identity and clean duplicates."""
    if not callable(request_factory):
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_WRITE)
    key = storage.write(content)
    try:
        request = request_factory(key)
        if not isinstance(request, DocumentRegistryCreateRequest):
            raise ValueError
        result = repository.create_or_get(request)
    except Exception:
        try:
            storage.delete(key)
        except Exception:
            pass
        raise
    cleanup_duplicate_source(
        storage,
        newly_created_key=key,
        registry_result=result,
    )
    return result


def delete_document_then_source(
    repository: DocumentRegistryRepository,
    storage: PersistentDocumentStorage,
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
) -> bool:
    """Commit exact registry deletion before attempting source deletion."""
    result = repository.delete_document(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )
    if result is None:
        return False
    storage.delete(result.storage_object_key)
    return True


@dataclass(frozen=True, slots=True)
class _ManagedObject:
    name: str
    modified_at: float
    temporary: bool


def _validated_storage_root(
    root: object,
    *,
    repository_root: Path,
    user_profile: Path,
) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_CONFIGURATION)
    _reject_link_components(root)
    if not root.is_dir():
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_CONFIGURATION)
    resolved = root.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    profile = user_profile.resolve(strict=True)
    forbidden = {
        Path(root.anchor).resolve(strict=True),
        profile,
        Path(tempfile.gettempdir()).resolve(strict=True),
    }
    if os.name == "nt":
        for environment_key in ("PUBLIC", "PROGRAMDATA"):
            raw = os.environ.get(environment_key)
            if raw:
                forbidden.add(Path(raw).resolve(strict=True))
        forbidden.add(Path(root.anchor, "Users").resolve(strict=False))
    else:
        forbidden.update(Path(value) for value in ("/tmp", "/var/tmp", "/home"))
    if (
        resolved in forbidden
        or resolved == repository
        or repository in resolved.parents
    ):
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_CONFIGURATION)
    return resolved


def _reject_link_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        status = current.lstat()
        attributes = getattr(status, "st_file_attributes", 0)
        if stat.S_ISLNK(status.st_mode) or attributes & 0x400:
            raise PersistentStorageError(PersistentStorageFailure.STORAGE_CONFIGURATION)


def _validated_key(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != len(_MANAGED_PREFIX) + _KEY_HEX_CHARACTERS
        or not value.startswith(_MANAGED_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[4:])
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError("invalid object key")
    return value


def _new_key(factory: Callable[[], str]) -> str:
    try:
        raw = factory()
        return _validated_key(f"{_MANAGED_PREFIX}{raw}")
    except Exception:
        raise PersistentStorageError(PersistentStorageFailure.STORAGE_WRITE) from None


def _write_flags() -> int:
    return os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)


def _read_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _open_read_descriptor(path: Path) -> int:
    if os.name != "nt":
        return os.open(path, _read_flags())
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,
        0x00000001,
        None,
        3,
        0x00000080 | 0x00200000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise OSError("object open failed")
    try:
        return msvcrt.open_osfhandle(int(handle), os.O_RDONLY)
    except Exception:
        kernel32.CloseHandle(handle)
        raise


def _set_owner_only_mode(path: Path) -> None:
    if os.name == "nt":
        os.chmod(path, 0o600)
    else:
        os.chmod(path, 0o600, follow_symlinks=False)


def _publish_exclusive(temporary_path: Path, final_path: Path) -> None:
    if os.name == "nt":
        os.rename(temporary_path, final_path)
        return
    os.link(temporary_path, final_path, follow_symlinks=False)
    temporary_path.unlink()


def _validate_regular_object(path: Path, failure: PersistentStorageFailure) -> None:
    _validate_object_status(path.stat(follow_symlinks=False), failure)


def _validate_object_status(
    status: os.stat_result, failure: PersistentStorageFailure
) -> None:
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_size < 0
        or status.st_size > MAX_UPLOAD_BYTES
        or not _has_single_link(status)
        or getattr(status, "st_file_attributes", 0) & 0x400
    ):
        raise PersistentStorageError(failure)
    if os.name != "nt" and stat.S_IMODE(status.st_mode) & 0o077:
        raise PersistentStorageError(failure)


def _status_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _has_single_link(status: os.stat_result) -> bool:
    return status.st_nlink in ({0, 1} if os.name == "nt" else {1})


def _close_descriptor(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except Exception:
            pass


def _unlink_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _managed_direct_children(root: Path) -> tuple[_ManagedObject, ...]:
    managed: list[_ManagedObject] = []
    with os.scandir(root) as entries:
        for entry in entries:
            final = _is_managed_name(entry.name)
            temporary = _is_temporary_name(entry.name)
            if not final and not temporary:
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            status = entry.stat(follow_symlinks=False)
            _validate_object_status(status, PersistentStorageFailure.RECONCILE)
            managed.append(
                _ManagedObject(entry.name, status.st_mtime, temporary=temporary)
            )
    return tuple(sorted(managed, key=lambda item: item.name))


def _is_managed_name(name: str) -> bool:
    try:
        _validated_key(name)
    except ValueError:
        return False
    return True


def _is_temporary_name(name: str) -> bool:
    return (
        len(name) == len(_TEMPORARY_PREFIX) + _KEY_HEX_CHARACTERS
        and name.startswith(_TEMPORARY_PREFIX)
        and all(character in "0123456789abcdef" for character in name[5:])
    )


def _delete_managed_temporary(root: Path, name: str) -> None:
    if not _is_temporary_name(name):
        raise PersistentStorageError(PersistentStorageFailure.RECONCILE)
    path = root / name
    status = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode) or not _has_single_link(status):
        raise PersistentStorageError(PersistentStorageFailure.RECONCILE)
    path.unlink()


def _sid_bytes(advapi32: Any, pointer: ctypes.c_void_p) -> bytes:
    length = advapi32.GetLengthSid(pointer)
    if not length:
        raise OSError("SID inspection failed")
    return ctypes.string_at(pointer, length)


def _current_windows_user_sid(advapi32: Any, kernel32: Any, wintypes: Any) -> bytes:
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise OSError("process token inspection failed")
    try:
        required = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        if not required.value:
            raise OSError("process token inspection failed")
        buffer = ctypes.create_string_buffer(required.value)
        if not advapi32.GetTokenInformation(
            token,
            1,
            buffer,
            required,
            ctypes.byref(required),
        ):
            raise OSError("process token inspection failed")
        sid_pointer = ctypes.c_void_p.from_buffer(buffer).value
        if not sid_pointer:
            raise OSError("process token inspection failed")
        return _sid_bytes(advapi32, ctypes.c_void_p(sid_pointer))
    finally:
        kernel32.CloseHandle(token)


def _allowed_windows_writers(
    advapi32: Any,
    dacl: ctypes.c_void_p,
    current_sid: bytes,
) -> tuple[tuple[bytes, ...], bool]:
    class _ACL(ctypes.Structure):
        _fields_ = [
            ("revision", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte),
            ("size", ctypes.c_ushort),
            ("ace_count", ctypes.c_ushort),
            ("reserved_two", ctypes.c_ushort),
        ]

    acl = ctypes.cast(dacl, ctypes.POINTER(_ACL)).contents
    writable: list[bytes] = []
    current_control = False
    for index in range(acl.ace_count):
        ace = ctypes.c_void_p()
        if not advapi32.GetAce(dacl, index, ctypes.byref(ace)) or not ace.value:
            raise OSError("ACL inspection failed")
        ace_type = ctypes.c_ubyte.from_address(ace.value).value
        if ace_type != 0:
            if ace_type in {5, 9}:
                raise OSError("object ACE inspection is unsupported")
            continue
        mask = ctypes.c_uint32.from_address(ace.value + 4).value
        if not mask & _WRITE_ACCESS_MASK:
            continue
        sid_pointer = ctypes.c_void_p(ace.value + 8)
        sid = _sid_bytes(advapi32, sid_pointer)
        writable.append(sid)
        if sid == current_sid and (
            mask & _GENERIC_ALL or (mask & 0x00040000 and mask & 0x0002)
        ):
            current_control = True
    return tuple(writable), current_control
