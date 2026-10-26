import hashlib
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Self
from urllib.error import URLError

import pytest

import docguard.ingestion.download as download_module
from docguard.ingestion.download import DownloadError, RetryPolicy, download_archive
from docguard.ingestion.models import ArchiveSource

USER_AGENT = "DocGuard-RAG/0.1 (+https://github.com/Aoongu/DocGuard-RAG)"


class _ScenarioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        body: bytes,
        *,
        statuses: tuple[int, ...] = (200,),
        range_mode: str = "ignore",
        delay_seconds: float = 0,
        truncate_first_at: int | None = None,
        max_body_bytes: int | None = None,
        omit_content_length: bool = False,
        content_length_delta: int = 0,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _ScenarioHandler)
        self.body = body
        self.statuses = statuses
        self.range_mode = range_mode
        self.delay_seconds = delay_seconds
        self.truncate_first_at = truncate_first_at
        self.max_body_bytes = max_body_bytes
        self.omit_content_length = omit_content_length
        self.content_length_delta = content_length_delta
        self.requests: list[dict[str, str]] = []


class _ScenarioHandler(BaseHTTPRequestHandler):
    server: _ScenarioServer

    def do_GET(self) -> None:
        self.server.requests.append(dict(self.headers.items()))
        if self.server.delay_seconds:
            time.sleep(self.server.delay_seconds)

        index = min(len(self.server.requests) - 1, len(self.server.statuses) - 1)
        status = self.server.statuses[index]
        body = self.server.body
        headers: dict[str, str] = {}
        requested_range = self.headers.get("Range")

        if status == 200 and requested_range and self.server.range_mode != "ignore":
            start = int(requested_range.removeprefix("bytes=").removesuffix("-"))
            if self.server.range_mode == "support":
                status = 206
                body = body[start:]
                headers["Content-Range"] = (
                    f"bytes {start}-{len(self.server.body) - 1}/{len(self.server.body)}"
                )
            elif self.server.range_mode.startswith("bad"):
                status = 206
                body = body[start:]
                content_range = (
                    f"bytes {start}-{len(self.server.body) - 1}/"
                    f"{len(self.server.body)}"
                )
                if self.server.range_mode in {"bad", "bad_start"}:
                    content_range = content_range.replace(
                        f"bytes {start}-", f"bytes {start + 1}-"
                    )
                elif self.server.range_mode == "bad_end":
                    content_range = content_range.replace(
                        f"-{len(self.server.body) - 1}/",
                        f"-{len(self.server.body) - 2}/",
                    )
                elif self.server.range_mode == "bad_total":
                    content_range = content_range.replace(
                        f"/{len(self.server.body)}", f"/{len(self.server.body) + 1}"
                    )
                elif self.server.range_mode == "bad_malformed":
                    content_range = f"bytes {start}-*/*"
                headers["Content-Range"] = content_range

        self.send_response(status)
        if not self.server.omit_content_length:
            self.send_header(
                "Content-Length", str(len(body) + self.server.content_length_delta)
            )
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        if len(self.server.requests) == 1 and self.server.truncate_first_at is not None:
            body = body[: self.server.truncate_first_at]
        if self.server.max_body_bytes is not None:
            body = body[: self.server.max_body_bytes]
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _serve(
    body: bytes,
    *,
    statuses: tuple[int, ...] = (200,),
    range_mode: str = "ignore",
    delay_seconds: float = 0,
    truncate_first_at: int | None = None,
    max_body_bytes: int | None = None,
    omit_content_length: bool = False,
    content_length_delta: int = 0,
) -> Iterator[_ScenarioServer]:
    server = _ScenarioServer(
        body,
        statuses=statuses,
        range_mode=range_mode,
        delay_seconds=delay_seconds,
        truncate_first_at=truncate_first_at,
        max_body_bytes=max_body_bytes,
        omit_content_length=omit_content_length,
        content_length_delta=content_length_delta,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _source(url: str, body: bytes) -> ArchiveSource:
    return ArchiveSource(
        version="17.11",
        major_version="17",
        url=url,
        archive_path=PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        retrieved_at="2026-10-11T16:36:16Z",
    )


def _url(server: _ScenarioServer) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}/archive.tar.gz"


def _part(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def _lock(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.download.lock")


def test_retry_policy_is_frozen_slotted_and_has_expected_defaults() -> None:
    policy = RetryPolicy()

    assert (policy.max_attempts, policy.backoff_seconds) == (3, 0.25)
    assert not hasattr(policy, "__dict__")
    with pytest.raises(FrozenInstanceError):
        policy.max_attempts = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": True}, "max_attempts"),
        ({"backoff_seconds": -0.1}, "backoff_seconds"),
    ],
)
def test_retry_policy_rejects_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"max_attempts": 3, "backoff_seconds": 0.25}
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        RetryPolicy(**values)  # type: ignore[arg-type]


def test_download_writes_verified_archive_atomically(tmp_path: Path) -> None:
    body = b"archive-bytes"
    with _serve(body) as server:
        destination = tmp_path / "nested" / "archive.tar.gz"
        download_archive(_source(_url(server), body), destination)

    assert destination.read_bytes() == body
    assert not _part(destination).exists()
    assert server.requests[0]["User-Agent"] == USER_AGENT


def test_existing_verified_destination_is_idempotent_without_request(
    tmp_path: Path,
) -> None:
    body = b"already-downloaded"
    with _serve(body) as server:
        destination = tmp_path / "archive.tar.gz"
        destination.write_bytes(body)
        _part(destination).write_bytes(b"stale")
        download_archive(_source(_url(server), body), destination)

    assert server.requests == []
    assert not _part(destination).exists()


def test_existing_conflicting_destination_is_not_overwritten(tmp_path: Path) -> None:
    body = b"expected"
    with _serve(body) as server:
        destination = tmp_path / "archive.tar.gz"
        destination.write_bytes(b"conflict")
        _part(destination).write_bytes(b"stale")
        with pytest.raises(DownloadError, match="destination"):
            download_archive(_source(_url(server), body), destination)

    assert destination.read_bytes() == b"conflict"
    assert not _part(destination).exists()
    assert server.requests == []


def test_404_is_not_retried(tmp_path: Path) -> None:
    body = b"not-found"
    with _serve(body, statuses=(404,)) as server, pytest.raises(
        DownloadError, match="HTTP 404"
    ):
        download_archive(
            _source(_url(server), body),
            tmp_path / "missing.tar.gz",
            retry_policy=RetryPolicy(3, 0),
        )

    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_retryable_status_exhausts_bounded_attempts(
    tmp_path: Path, status: int
) -> None:
    body = b"retry"
    with _serve(body, statuses=(status,)) as server, pytest.raises(
        DownloadError, match="after 3 attempts"
    ):
        download_archive(
            _source(_url(server), body),
            tmp_path / "retry.tar.gz",
            retry_policy=RetryPolicy(3, 0),
        )

    assert len(server.requests) == 3


def test_timeout_is_retried_and_bounded(tmp_path: Path) -> None:
    body = b"too-slow"
    with _serve(body, delay_seconds=0.1) as server, pytest.raises(
        DownloadError, match="after 2 attempts"
    ):
        download_archive(
            _source(_url(server), body),
            tmp_path / "timeout.tar.gz",
            retry_policy=RetryPolicy(2, 0),
            timeout_seconds=0.01,
        )

    assert len(server.requests) == 2


def test_connection_error_is_retried_and_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def fail_connection(request: object, timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise URLError("connection refused")

    monkeypatch.setattr("docguard.ingestion.download.urlopen", fail_connection)
    source = _source("http://127.0.0.1:1/archive.tar.gz", b"offline")

    with pytest.raises(DownloadError, match="after 2 attempts"):
        download_archive(
            source,
            tmp_path / "offline.tar.gz",
            retry_policy=RetryPolicy(2, 0),
        )

    assert calls == 2


def test_range_resume_appends_only_on_matching_206(tmp_path: Path) -> None:
    body = b"archive-bytes"
    with _serve(body, range_mode="support") as server:
        destination = tmp_path / "resume.tar.gz"
        _part(destination).write_bytes(b"archive-")
        download_archive(_source(_url(server), body), destination)

    assert destination.read_bytes() == body
    assert server.requests[0]["Range"] == "bytes=8-"


def test_range_ignored_with_200_safely_rewrites_part(tmp_path: Path) -> None:
    body = b"archive-bytes"
    with _serve(body, range_mode="ignore") as server:
        destination = tmp_path / "rewrite.tar.gz"
        _part(destination).write_bytes(b"wrong-prefix")
        download_archive(_source(_url(server), body), destination)

    assert destination.read_bytes() == body
    assert server.requests[0]["Range"] == "bytes=12-"


def test_mismatched_content_range_fails_without_publishing(tmp_path: Path) -> None:
    body = b"archive-bytes"
    with _serve(body, range_mode="bad") as server:
        destination = tmp_path / "bad-range.tar.gz"
        _part(destination).write_bytes(b"archive-")
        with pytest.raises(DownloadError, match="Content-Range"):
            download_archive(_source(_url(server), body), destination)

    assert not destination.exists()
    assert not _part(destination).exists()
    assert len(server.requests) == 1


def test_range_416_fails_without_retry_or_publishing(tmp_path: Path) -> None:
    body = b"archive-bytes"
    with _serve(body, statuses=(416,)) as server:
        destination = tmp_path / "range-416.tar.gz"
        _part(destination).write_bytes(b"archive-")
        with pytest.raises(DownloadError, match="HTTP 416"):
            download_archive(
                _source(_url(server), body),
                destination,
                retry_policy=RetryPolicy(3, 0),
            )

    assert not destination.exists()
    assert not _part(destination).exists()
    assert len(server.requests) == 1


@pytest.mark.parametrize("bad_field", ["size", "hash"])
def test_integrity_failure_removes_part_without_publishing(
    tmp_path: Path, bad_field: str
) -> None:
    body = b"downloaded"
    expected = b"corrupted!" if bad_field == "hash" else body + b"!"
    with _serve(body) as server:
        destination = tmp_path / f"bad-{bad_field}.tar.gz"
        with pytest.raises(DownloadError, match=bad_field):
            download_archive(_source(_url(server), expected), destination)

    assert not destination.exists()
    assert not _part(destination).exists()


def test_integrity_failure_does_not_touch_existing_destination(tmp_path: Path) -> None:
    body = b"downloaded"
    destination = tmp_path / "archive.tar.gz"
    destination.write_bytes(b"pre-existing")

    with _serve(body) as server, pytest.raises(DownloadError, match="destination"):
        download_archive(_source(_url(server), b"expected"), destination)

    assert destination.read_bytes() == b"pre-existing"
    assert not _part(destination).exists()


def test_retries_use_exponential_backoff_sequence(tmp_path: Path) -> None:
    body = b"eventual-success"
    sleeps: list[float] = []
    with _serve(body, statuses=(503, 429, 200)) as server:
        destination = tmp_path / "archive.tar.gz"
        download_archive(
            _source(_url(server), body),
            destination,
            retry_policy=RetryPolicy(3, 0.5),
            sleep=sleeps.append,
        )

    assert destination.read_bytes() == body
    assert sleeps == [0.5, 1.0]


def test_each_response_is_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = b"response-body"

    class Response:
        status = 200

        def __init__(self) -> None:
            self.closed = False
            self.offset = 0
            self.headers: dict[str, str] = {}

        def read(self, size: int = -1) -> bytes:
            if self.offset:
                return b""
            self.offset = len(body)
            return body

        def close(self) -> None:
            self.closed = True

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    response = Response()

    def fake_urlopen(request: object, timeout: float) -> Response:
        return response

    monkeypatch.setattr("docguard.ingestion.download.urlopen", fake_urlopen)
    destination = tmp_path / "archive.tar.gz"
    download_archive(
        _source("http://127.0.0.1:1/archive.tar.gz", body), destination
    )

    assert response.closed
    assert destination.read_bytes() == body


def test_urlerror_is_retryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: object, timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise URLError("offline")

    monkeypatch.setattr("docguard.ingestion.download.urlopen", fake_urlopen)
    destination = tmp_path / "archive.tar.gz"
    _part(destination).write_bytes(b"off")
    with pytest.raises(DownloadError, match="after 2 attempts"):
        download_archive(
            _source("http://127.0.0.1:1/archive.tar.gz", b"offline"),
            destination,
            retry_policy=RetryPolicy(2, 0.25),
            sleep=sleeps.append,
        )

    assert calls == 2
    assert sleeps == [0.25]
    assert _part(destination).read_bytes() == b"off"


@pytest.mark.parametrize("timeout_seconds", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_finite_and_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout_seconds: float
) -> None:
    def unexpected_request(request: object, timeout: float) -> object:
        raise AssertionError("invalid timeout reached the network")

    monkeypatch.setattr("docguard.ingestion.download.urlopen", unexpected_request)
    with pytest.raises(ValueError, match="timeout_seconds"):
        download_archive(
            _source("http://127.0.0.1:1/archive.tar.gz", b"archive"),
            tmp_path / "archive.tar.gz",
            timeout_seconds=timeout_seconds,
        )


def test_short_first_response_is_resumed_by_matching_206(tmp_path: Path) -> None:
    body = b"complete-archive"
    with _serve(body, range_mode="support", truncate_first_at=8) as server:
        destination = tmp_path / "archive.tar.gz"
        download_archive(
            _source(_url(server), body),
            destination,
            retry_policy=RetryPolicy(2, 0),
        )

    assert destination.read_bytes() == body
    assert len(server.requests) == 2
    assert "Range" not in server.requests[0]
    assert server.requests[1]["Range"] == "bytes=8-"
    assert not _part(destination).exists()
    assert _lock(destination).is_file()


def test_short_responses_exhaust_attempts_but_keep_safe_part(tmp_path: Path) -> None:
    body = b"complete-archive"
    with _serve(body, range_mode="support", max_body_bytes=2) as server:
        destination = tmp_path / "archive.tar.gz"
        with pytest.raises(DownloadError, match="after 2 attempts"):
            download_archive(
                _source(_url(server), body),
                destination,
                retry_policy=RetryPolicy(2, 0),
            )

    assert len(server.requests) == 2
    assert _part(destination).read_bytes() == body[:4]
    assert not destination.exists()
    assert _lock(destination).is_file()


@pytest.mark.parametrize("resume_at", [0, 4])
def test_content_length_must_match_expected_response_bytes(
    tmp_path: Path, resume_at: int
) -> None:
    body = b"archive-body"
    with _serve(
        body, range_mode="support", content_length_delta=1
    ) as server:
        destination = tmp_path / "archive.tar.gz"
        if resume_at:
            _part(destination).write_bytes(body[:resume_at])
        with pytest.raises(DownloadError, match="Content-Length"):
            download_archive(_source(_url(server), body), destination)

    assert not destination.exists()
    assert not _part(destination).exists()


@pytest.mark.parametrize(
    "range_mode", ["bad_start", "bad_end", "bad_total", "bad_malformed"]
)
def test_content_range_must_strictly_cover_expected_remainder(
    tmp_path: Path, range_mode: str
) -> None:
    body = b"archive-body"
    with _serve(body, range_mode=range_mode) as server:
        destination = tmp_path / "archive.tar.gz"
        _part(destination).write_bytes(body[:4])
        with pytest.raises(DownloadError, match="Content-Range"):
            download_archive(_source(_url(server), body), destination)

    assert not destination.exists()
    assert not _part(destination).exists()


def test_stream_without_content_length_cannot_exceed_expected_size(
    tmp_path: Path,
) -> None:
    expected = b"archive-body"
    with _serve(expected + b"unbounded", omit_content_length=True) as server:
        destination = tmp_path / "archive.tar.gz"
        with pytest.raises(DownloadError, match="exceeds expected size"):
            download_archive(_source(_url(server), expected), destination)

    assert not destination.exists()
    assert not _part(destination).exists()


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_unsafe_existing_part_is_rejected_without_changing_victim(
    tmp_path: Path, link_kind: str
) -> None:
    body = b"safe-archive"
    destination = tmp_path / "archive.tar.gz"
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"victim-content")
    try:
        if link_kind == "hardlink":
            os.link(victim, _part(destination))
        else:
            os.symlink(victim, _part(destination))
    except OSError as error:
        pytest.skip(f"{link_kind} unavailable: {error}")

    with _serve(body) as server, pytest.raises(DownloadError, match="part"):
        download_archive(_source(_url(server), body), destination)

    assert victim.read_bytes() == b"victim-content"
    assert server.requests == []
    assert not destination.exists()
    assert _lock(destination).is_file()


@pytest.mark.parametrize("concurrent_content", [b"archive-body", b"conflict"])
def test_publish_is_atomic_create_if_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    concurrent_content: bytes,
) -> None:
    body = b"archive-body"
    destination = tmp_path / "archive.tar.gz"
    real_link = os.link
    calls = 0

    def inject_concurrent_destination(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal calls
        calls += 1
        destination.write_bytes(concurrent_content)
        raise FileExistsError(str(destination))

    monkeypatch.setattr(os, "link", inject_concurrent_destination)
    with _serve(body) as server:
        if concurrent_content == body:
            download_archive(_source(_url(server), body), destination)
        else:
            with pytest.raises(DownloadError, match="conflicting"):
                download_archive(_source(_url(server), body), destination)

    monkeypatch.setattr(os, "link", real_link)
    assert calls == 1
    assert destination.read_bytes() == concurrent_content
    assert not _part(destination).exists()
    assert _lock(destination).is_file()


def test_publish_oserror_is_wrapped_with_cause_and_cleans_part_and_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b"archive-body"
    destination = tmp_path / "archive.tar.gz"

    def deny_link(*args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "link", deny_link)
    with _serve(body) as server, pytest.raises(DownloadError) as raised:
        download_archive(_source(_url(server), body), destination)

    assert isinstance(raised.value.__cause__, PermissionError)
    assert not destination.exists()
    assert not _part(destination).exists()
    assert _lock(destination).is_file()


def test_lock_exists_during_request_and_is_reacquirable_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "archive.tar.gz"
    observed_lock = False

    def fail_after_observing_lock(request: object, timeout: float) -> object:
        nonlocal observed_lock
        observed_lock = _lock(destination).exists()
        raise URLError("offline")

    monkeypatch.setattr("docguard.ingestion.download.urlopen", fail_after_observing_lock)
    with pytest.raises(DownloadError, match="after 1 attempts"):
        download_archive(
            _source("http://127.0.0.1:1/archive.tar.gz", b"archive"),
            destination,
            retry_policy=RetryPolicy(1, 0),
        )

    assert observed_lock
    assert _lock(destination).is_file()
    with download_module._download_lock(_lock(destination)):
        pass


def test_stale_lock_file_does_not_block_download_or_part_resume(tmp_path: Path) -> None:
    body = b"archive"
    destination = tmp_path / "archive.tar.gz"
    _lock(destination).write_bytes(b"stale")
    _part(destination).write_bytes(body[:3])

    with _serve(body, range_mode="support") as server:
        download_archive(_source(_url(server), body), destination)

    assert destination.read_bytes() == body
    assert server.requests[0]["Range"] == "bytes=3-"
    assert _lock(destination).is_file()


def test_active_os_lock_refuses_second_cooperative_download(tmp_path: Path) -> None:
    body = b"archive"
    destination = tmp_path / "archive.tar.gz"

    with (
        download_module._download_lock(_lock(destination)),
        _serve(body) as server,
        pytest.raises(DownloadError, match="active download lock"),
    ):
        download_archive(_source(_url(server), body), destination)

    assert server.requests == []
    assert _lock(destination).is_file()


def test_released_os_lock_can_be_acquired_again(tmp_path: Path) -> None:
    lock_path = _lock(tmp_path / "archive.tar.gz")

    with download_module._download_lock(lock_path):
        pass
    with download_module._download_lock(lock_path):
        pass

    assert lock_path.is_file()


def test_part_file_oserror_is_wrapped_and_lock_is_released(
    tmp_path: Path,
) -> None:
    body = b"archive"
    destination = tmp_path / "archive.tar.gz"
    _part(destination).mkdir()

    with _serve(body) as server, pytest.raises(DownloadError) as raised:
        download_archive(_source(_url(server), body), destination)

    assert isinstance(raised.value.__cause__, OSError)
    assert _lock(destination).is_file()


def test_read_connection_reset_resumes_from_safely_written_part(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b"archive-body"
    ranges: list[str | None] = []

    class Response:
        def __init__(
            self,
            status: int,
            headers: dict[str, str],
            chunks: list[bytes | BaseException],
        ) -> None:
            self.status = status
            self.headers = headers
            self.chunks = iter(chunks)

        def read(self, size: int = -1) -> bytes:
            value = next(self.chunks, b"")
            if isinstance(value, BaseException):
                raise value
            return value

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    responses = iter(
        [
            Response(200, {}, [body[:4], ConnectionResetError("reset")]),
            Response(
                206,
                {
                    "Content-Length": str(len(body) - 4),
                    "Content-Range": f"bytes 4-{len(body) - 1}/{len(body)}",
                },
                [body[4:]],
            ),
        ]
    )

    def fake_urlopen(request: object, timeout: float) -> Response:
        ranges.append(request.get_header("Range"))  # type: ignore[attr-defined]
        return next(responses)

    monkeypatch.setattr("docguard.ingestion.download.urlopen", fake_urlopen)
    destination = tmp_path / "archive.tar.gz"
    download_archive(
        _source("http://127.0.0.1:1/archive.tar.gz", body),
        destination,
        retry_policy=RetryPolicy(2, 0),
    )

    assert destination.read_bytes() == body
    assert ranges == [None, "bytes=4-"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_windows_reparse_part_is_rejected_without_changing_victim(
    tmp_path: Path,
) -> None:
    body = b"archive"
    destination = tmp_path / "archive.tar.gz"
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    marker = victim / "marker.bin"
    marker.write_bytes(b"victim-content")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(_part(destination)), str(victim)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"junction unavailable: {result.stderr or result.stdout}")

    try:
        with _serve(body) as server, pytest.raises(DownloadError, match="part"):
            download_archive(_source(_url(server), body), destination)
        assert marker.read_bytes() == b"victim-content"
        assert server.requests == []
        assert _lock(destination).is_file()
    finally:
        if os.path.lexists(_part(destination)):
            os.rmdir(_part(destination))


def test_hardlinked_lock_is_rejected_without_changing_victim(tmp_path: Path) -> None:
    body = b"archive"
    destination = tmp_path / "archive.tar.gz"
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"victim-content")
    os.link(victim, _lock(destination))

    with _serve(body) as server, pytest.raises(DownloadError, match="lock"):
        download_archive(_source(_url(server), body), destination)

    assert victim.read_bytes() == b"victim-content"
    assert server.requests == []
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_windows_reparse_lock_is_rejected_without_changing_victim(
    tmp_path: Path,
) -> None:
    body = b"archive"
    destination = tmp_path / "archive.tar.gz"
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    marker = victim / "marker.bin"
    marker.write_bytes(b"victim-content")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(_lock(destination)), str(victim)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"junction unavailable: {result.stderr or result.stdout}")

    try:
        with _serve(body) as server, pytest.raises(DownloadError, match="lock"):
            download_archive(_source(_url(server), body), destination)
        assert marker.read_bytes() == b"victim-content"
        assert server.requests == []
        assert not destination.exists()
    finally:
        if os.path.lexists(_lock(destination)):
            os.rmdir(_lock(destination))


def test_gitignore_only_ignores_archive_download_locks() -> None:
    project_root = Path(__file__).resolve().parents[2]
    patterns = (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert patterns.count("data/archives/*.download.lock") == 1
    assert "*.lock" not in patterns
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "uv.lock"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
