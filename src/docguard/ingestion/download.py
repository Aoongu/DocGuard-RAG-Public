"""Verified, resumable downloads for pinned documentation archives."""

import hashlib
import math
import os
import re
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from docguard.ingestion.models import ArchiveSource

if os.name == "nt":
    import msvcrt
else:
    import fcntl

USER_AGENT = "DocGuard-RAG/0.1 (+https://github.com/Aoongu/DocGuard-RAG)"
_BUFFER_SIZE = 1024 * 1024
_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_BINARY_FLAG = getattr(os, "O_BINARY", 0)
_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if (
            isinstance(self.backoff_seconds, bool)
            or not isinstance(self.backoff_seconds, (int, float))
            or not math.isfinite(self.backoff_seconds)
            or self.backoff_seconds < 0
        ):
            raise ValueError("backoff_seconds must be a non-negative finite number")


class DownloadError(RuntimeError):
    """Raised when a fixed archive cannot be safely downloaded and published."""


class _RetryableDownloadError(Exception):
    pass


def download_archive(
    source: ArchiveSource,
    destination: Path,
    *,
    retry_policy: RetryPolicy = RetryPolicy(),  # noqa: B008
    timeout_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Download, verify, and atomically publish one fixed archive."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DownloadError("unable to create destination directory") from error

    part_path = destination.with_name(f"{destination.name}.part")
    lock_path = destination.with_name(f"{destination.name}.download.lock")
    with _download_lock(lock_path):
        _download_locked(
            source,
            destination,
            part_path,
            retry_policy,
            float(timeout_seconds),
            sleep,
        )


def _download_locked(
    source: ArchiveSource,
    destination: Path,
    part_path: Path,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
    sleep: Callable[[float], None],
) -> None:
    if os.path.lexists(part_path):
        try:
            with _secure_open_part(part_path):
                pass
        except DownloadError:
            _discard_part(part_path)
            raise

    if destination.exists():
        try:
            matches = _matches_source(destination, source)
        except OSError as error:
            _discard_part(part_path)
            raise DownloadError("unable to inspect existing destination") from error
        _discard_part(part_path)
        if matches:
            return
        raise DownloadError(
            f"destination already exists with conflicting content: {destination}"
        )

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            _download_attempt(source, part_path, timeout_seconds)
        except HTTPError as error:
            error.close()
            if error.code == 429 or 500 <= error.code <= 599:
                if attempt < retry_policy.max_attempts:
                    _backoff(retry_policy, attempt, sleep)
                    continue
                raise DownloadError(
                    f"download failed after {attempt} attempts: HTTP {error.code}"
                ) from error
            _discard_part(part_path)
            raise DownloadError(f"download failed with HTTP {error.code}") from error
        except _RetryableDownloadError as error:
            if attempt < retry_policy.max_attempts:
                _backoff(retry_policy, attempt, sleep)
                continue
            raise DownloadError(
                f"download failed after {attempt} attempts: {error}"
            ) from error.__cause__
        except DownloadError:
            _discard_part(part_path)
            raise
        break

    _verify_part(part_path, source)
    _publish_part(part_path, destination, source)


def _download_attempt(
    source: ArchiveSource, part_path: Path, timeout_seconds: float
) -> None:
    with _secure_open_part(part_path) as part:
        try:
            resume_at = os.fstat(part.fileno()).st_size
        except OSError as error:
            raise DownloadError("unable to inspect open part file") from error
        if resume_at > source.size_bytes:
            raise DownloadError("part file exceeds expected size")
        if resume_at == source.size_bytes:
            return

        headers = {"User-Agent": USER_AGENT}
        if resume_at > 0:
            headers["Range"] = f"bytes={resume_at}-"
        request = Request(source.url, headers=headers)
        try:
            response = urlopen(request, timeout=timeout_seconds)
        except HTTPError:
            raise
        except (URLError, OSError) as error:
            raise _RetryableDownloadError(str(error)) from error

        try:
            with response:
                _consume_response(response, part, source, resume_at)
        except DownloadError:
            raise
        except (URLError, OSError) as error:
            raise _RetryableDownloadError(str(error)) from error


def _consume_response(
    response: object,
    part: BinaryIO,
    source: ArchiveSource,
    resume_at: int,
) -> None:
    status = response.status  # type: ignore[attr-defined]
    expected_bytes = source.size_bytes
    if resume_at > 0:
        if status == 206:
            _validate_content_range(
                response.headers.get("Content-Range"),  # type: ignore[attr-defined]
                resume_at,
                source.size_bytes,
            )
            expected_bytes -= resume_at
            _position_part(part, append=True)
        elif status == 200:
            _position_part(part, append=False)
        else:
            raise DownloadError(f"unexpected HTTP {status} for Range request")
    elif status == 200:
        _position_part(part, append=False)
    else:
        raise DownloadError(f"unexpected HTTP {status}")

    _validate_content_length(
        response.headers.get("Content-Length"),  # type: ignore[attr-defined]
        expected_bytes,
    )
    received = 0
    while True:
        remaining = expected_bytes - received
        try:
            chunk = response.read(min(_BUFFER_SIZE, remaining + 1))  # type: ignore[attr-defined]
        except (URLError, OSError) as error:
            raise _RetryableDownloadError(str(error)) from error
        if not chunk:
            break
        if len(chunk) > remaining:
            raise DownloadError("response exceeds expected size")
        try:
            part.write(chunk)
        except OSError as error:
            raise DownloadError("unable to write part file") from error
        received += len(chunk)

    try:
        part.flush()
        _validate_part_stat(os.fstat(part.fileno()))
    except OSError as error:
        raise DownloadError("unable to finalize part file") from error
    if received < expected_bytes:
        raise _RetryableDownloadError(
            f"short response: expected {expected_bytes}, received {received}"
        )


def _validate_content_length(value: str | None, expected: int) -> None:
    if value is None:
        return
    if re.fullmatch(r"\d+", value) is None or int(value) != expected:
        raise DownloadError(
            f"response Content-Length size must equal expected byte count {expected}"
        )


def _position_part(part: BinaryIO, *, append: bool) -> None:
    try:
        if append:
            part.seek(0, os.SEEK_END)
        else:
            part.seek(0)
            part.truncate(0)
    except OSError as error:
        raise DownloadError("unable to position part file") from error


def _validate_content_range(value: str | None, resume_at: int, total: int) -> None:
    match = _CONTENT_RANGE_RE.fullmatch(value or "")
    expected = (resume_at, total - 1, total)
    if match is None or tuple(map(int, match.groups())) != expected:
        raise DownloadError("response Content-Range does not match fixed archive")


@contextmanager
def _download_lock(lock_path: Path) -> Iterator[None]:
    with _secure_open_lock(lock_path) as lock:
        try:
            _acquire_os_lock(lock)
        except OSError as error:
            raise DownloadError(f"active download lock: {lock_path}") from error
        try:
            yield
        finally:
            try:
                _release_os_lock(lock)
            except OSError as error:
                raise DownloadError(f"unable to release download lock: {lock_path}") from error


@contextmanager
def _secure_open_lock(lock_path: Path) -> Iterator[BinaryIO]:
    descriptor: int | None = None
    for _ in range(3):
        try:
            before = os.lstat(lock_path)
        except FileNotFoundError:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _BINARY_FLAG
            try:
                descriptor = os.open(lock_path, flags, 0o600)
            except FileExistsError:
                continue
            except OSError as error:
                raise DownloadError("unable to safely create download lock") from error
            break
        except OSError as error:
            raise DownloadError("unable to inspect download lock") from error

        _validate_lock_stat(before)
        flags = os.O_RDWR | _BINARY_FLAG | _NOFOLLOW_FLAG
        try:
            descriptor = os.open(lock_path, flags)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise DownloadError("unable to safely open download lock") from error
        try:
            after = os.fstat(descriptor)
            _validate_lock_stat(after)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise DownloadError("download lock identity changed while opening")
        except Exception:
            os.close(descriptor)
            raise
        break
    if descriptor is None:
        raise DownloadError("download lock changed repeatedly while opening")

    try:
        lock = os.fdopen(descriptor, "r+b", closefd=True)
    except OSError as error:
        os.close(descriptor)
        raise DownloadError("unable to open download lock descriptor") from error
    try:
        try:
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"\0")
                lock.flush()
            _validate_lock_stat(os.fstat(lock.fileno()))
            lock.seek(0)
        except OSError as error:
            raise DownloadError("unable to prepare download lock") from error
        yield lock
    finally:
        try:
            lock.close()
        except OSError as error:
            raise DownloadError("unable to close download lock") from error


def _validate_lock_stat(metadata: os.stat_result) -> None:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_flag:
        raise DownloadError("download lock must not be a symlink or reparse point")
    if not stat.S_ISREG(metadata.st_mode):
        raise DownloadError("download lock must be a regular file")
    if metadata.st_nlink != 1:
        raise DownloadError("download lock must have exactly one hard link")


def _acquire_os_lock(lock: BinaryIO) -> None:
    lock.seek(0)
    if os.name == "nt":
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(lock: BinaryIO) -> None:
    lock.seek(0)
    if os.name == "nt":
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def _secure_open_part(part_path: Path) -> Iterator[BinaryIO]:
    descriptor: int | None = None
    for _ in range(3):
        try:
            before = os.lstat(part_path)
        except FileNotFoundError:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _BINARY_FLAG
            try:
                descriptor = os.open(part_path, flags, 0o600)
            except FileExistsError:
                continue
            except OSError as error:
                raise DownloadError("unable to safely create part file") from error
            break
        except OSError as error:
            raise DownloadError("unable to inspect part file") from error

        _validate_part_stat(before)
        flags = os.O_RDWR | _BINARY_FLAG | _NOFOLLOW_FLAG
        try:
            descriptor = os.open(part_path, flags)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise DownloadError("unable to safely open part file") from error
        try:
            after = os.fstat(descriptor)
            _validate_part_stat(after)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise DownloadError("part file identity changed while opening")
        except Exception:
            os.close(descriptor)
            raise
        break
    if descriptor is None:
        raise DownloadError("part file changed repeatedly while opening")

    try:
        file = os.fdopen(descriptor, "r+b", closefd=True)
    except OSError as error:
        os.close(descriptor)
        raise DownloadError("unable to open part file descriptor") from error
    try:
        yield file
    finally:
        try:
            file.close()
        except OSError as error:
            raise DownloadError("unable to close part file") from error


def _validate_part_stat(metadata: os.stat_result) -> None:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_flag:
        raise DownloadError("part path must not be a symlink or reparse point")
    if not stat.S_ISREG(metadata.st_mode):
        error = OSError("part path is not a regular file")
        raise DownloadError("part path is not a regular file") from error
    if metadata.st_nlink != 1:
        raise DownloadError("part file must have exactly one hard link")


def _matches_source(path: Path, source: ArchiveSource) -> bool:
    if path.stat().st_size != source.size_bytes:
        return False
    return _sha256_path(path) == source.sha256


def _verify_part(part_path: Path, source: ArchiveSource) -> None:
    try:
        with _secure_open_part(part_path) as part:
            metadata = os.fstat(part.fileno())
            if metadata.st_size != source.size_bytes:
                raise DownloadError(
                    f"size mismatch: expected {source.size_bytes}, "
                    f"received {metadata.st_size}"
                )
            digest = hashlib.sha256()
            part.seek(0)
            while chunk := part.read(_BUFFER_SIZE):
                digest.update(chunk)
            _validate_part_stat(os.fstat(part.fileno()))
            if digest.hexdigest() != source.sha256:
                raise DownloadError(
                    f"hash mismatch: expected {source.sha256}, "
                    f"received {digest.hexdigest()}"
                )
    except DownloadError:
        _discard_part(part_path)
        raise
    except OSError as error:
        _discard_part(part_path)
        raise DownloadError("unable to verify part file") from error


def _publish_part(part_path: Path, destination: Path, source: ArchiveSource) -> None:
    try:
        os.link(part_path, destination)
    except FileExistsError as error:
        try:
            matches = _matches_source(destination, source)
        except OSError as inspect_error:
            _discard_part(part_path)
            raise DownloadError("unable to inspect concurrent destination") from inspect_error
        _discard_part(part_path)
        if matches:
            return
        raise DownloadError("concurrent destination has conflicting content") from error
    except OSError as error:
        _discard_part(part_path)
        raise DownloadError("unable to atomically publish archive") from error

    try:
        os.unlink(part_path)
    except OSError as error:
        raise DownloadError("archive published but part cleanup failed") from error


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _backoff(
    retry_policy: RetryPolicy,
    attempt: int,
    sleep: Callable[[float], None],
) -> None:
    sleep(retry_policy.backoff_seconds * (2 ** (attempt - 1)))


def _discard_part(part_path: Path) -> None:
    try:
        os.unlink(part_path)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise DownloadError(f"unable to remove part file: {part_path}") from error
