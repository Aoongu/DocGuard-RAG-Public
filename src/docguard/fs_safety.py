"""Shared fail-closed filesystem primitives for atomic directory publication."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

from docguard.ingestion.download import DownloadError, _download_lock

_BINARY_FLAG = getattr(os, "O_BINARY", 0)
_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
DirectoryIdentity = tuple[int, int]


class FileSafetyError(RuntimeError):
    """Raised when a filesystem object violates the publication policy."""


class FileConflictError(FileSafetyError):
    """Raised when a destination already exists or differs."""


def validate_directory(
    path: Path, expected_identity: DirectoryIdentity | None = None
) -> DirectoryIdentity:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FileSafetyError(f"unable to inspect directory: {path}") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
    ):
        raise FileSafetyError(f"unsafe directory: {path}")
    identity = metadata.st_dev, metadata.st_ino
    if expected_identity is not None and identity != expected_identity:
        raise FileSafetyError(f"directory identity changed: {path}")
    return identity


def ensure_directory(path: Path) -> DirectoryIdentity:
    if os.path.lexists(path):
        return validate_directory(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(path, 0o700)
    except FileExistsError:
        return validate_directory(path)
    except OSError as error:
        raise FileSafetyError(f"unable to create output directory: {path}") from error
    return validate_directory(path)


def validate_regular_file(path: Path, *, require_single_link: bool = True) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FileSafetyError(f"unable to inspect regular file: {path}") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or (require_single_link and metadata.st_nlink != 1)
    ):
        raise FileSafetyError(f"unsafe regular file: {path}")


def write_stage_file(stage: Path, name: str, data: bytes) -> None:
    """Create a verified single-link file without following existing links."""
    validate_directory(stage)
    if PurePosixPath(name).name != name or PureWindowsPath(name).name != name:
        raise FileSafetyError(f"invalid staging filename: {name}")
    target = stage / name
    descriptor: int | None = None
    identity: DirectoryIdentity | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG | _NOFOLLOW_FLAG
        descriptor = os.open(target, flags, 0o600)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise FileSafetyError(f"unsafe staging file: {target}")
        identity = opened.st_dev, opened.st_ino
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise FileSafetyError(f"short write to staging file: {target}")
            offset += written
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (*identity, len(data)):
            raise FileSafetyError(f"staging file identity changed: {target}")
    except OSError as error:
        raise FileSafetyError(f"unable to write staging file: {target}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    validate_regular_file(target)
    final = target.lstat()
    if identity != (final.st_dev, final.st_ino):
        raise FileSafetyError(f"staging file identity changed: {target}")
    if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise FileSafetyError(f"staging file verification failed: {target}")


def create_stage(parent: Path, prefix: str) -> tuple[Path, DirectoryIdentity]:
    parent_identity = validate_directory(parent)
    try:
        stage = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    except OSError as error:
        raise FileSafetyError("unable to create staging directory") from error
    validate_directory(parent, parent_identity)
    return stage, validate_directory(stage)


def verify_stage(
    stage: Path, identity: DirectoryIdentity, expected: Mapping[str, bytes]
) -> None:
    validate_directory(stage, identity)
    try:
        children = list(stage.iterdir())
    except OSError as error:
        raise FileSafetyError(f"unable to inspect staging directory: {stage}") from error
    if {path.name for path in children} != set(expected):
        raise FileSafetyError(f"staging file set mismatch: {stage}")
    for child in children:
        validate_regular_file(child)
        if child.read_bytes() != expected[child.name]:
            raise FileSafetyError(f"staging file verification failed: {child}")
    validate_directory(stage, identity)


@contextmanager
def exclusive_lock(parent: Path, name: str) -> Iterator[None]:
    parent_identity = validate_directory(parent)
    lock_directory = parent / ".docguard-locks"
    ensure_directory(lock_directory)
    validate_directory(parent, parent_identity)
    lock_path = lock_directory / f"{name}.lock"
    try:
        with _download_lock(lock_path):
            validate_directory(parent, parent_identity)
            yield
            validate_directory(parent, parent_identity)
    except DownloadError as error:
        raise FileSafetyError(f"unable to acquire publication lock: {lock_path}") from error


def _raise_rename_error(error: OSError, source: Path, destination: Path) -> None:
    if os.path.lexists(destination) or error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileConflictError(f"destination already exists: {destination}") from error
    raise FileSafetyError(
        f"atomic no-replace rename failed: {source} -> {destination}"
    ) from error


def atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing an existing destination."""
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except OSError as error:
            _raise_rename_error(error, source, destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise FileSafetyError("atomic no-replace directory rename is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result != 0:
            code = ctypes.get_errno()
            _raise_rename_error(OSError(code, os.strerror(code)), source, destination)
        return
    if sys.platform == "darwin" or "bsd" in sys.platform:
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = getattr(libc, "renamex_np", None)
        if rename_exclusive is None:
            raise FileSafetyError("atomic no-replace directory rename is unavailable")
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(os.fsencode(source), os.fsencode(destination), 0x4)
        if result != 0:
            code = ctypes.get_errno()
            _raise_rename_error(OSError(code, os.strerror(code)), source, destination)
        return
    raise FileSafetyError("atomic no-replace directory rename is unavailable")
