"""Safe temporary-file handling for user-provided dashboard audio."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


SUPPORTED_UPLOAD_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


@dataclass(frozen=True, slots=True)
class SafeUploadMetadata:
    filename: str
    format_name: str
    size_bytes: int


def safe_upload_metadata(filename: str, size_bytes: int) -> SafeUploadMetadata:
    """Return display-safe metadata without retaining file content or a path."""
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.casefold()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError("Desteklenmeyen ses dosyası biçimi")
    if size_bytes <= 0:
        raise ValueError("Yüklenen ses dosyası boş")
    return SafeUploadMetadata(safe_name, suffix.removeprefix(".").upper(), size_bytes)


@contextmanager
def temporary_uploaded_audio(filename: str, content: bytes) -> Iterator[Path]:
    """Yield an OS temporary path and always remove it after pipeline use."""
    metadata = safe_upload_metadata(filename, len(content))
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix="callmetric-upload-",
            suffix=f".{metadata.format_name.casefold()}",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
