"""Persistent asynchronous evaluation jobs for the S9 API."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from docguard.fs_safety import ensure_directory

_METHODS = frozenset({"b0", "b1", "m1", "m2", "b2", "m4"})
_SPLITS = frozenset({"development", "validation"})


class EvaluationJobNotFound(LookupError):
    """Raised when a run ID is absent from the S9 job store."""


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    run_id: str
    method: str
    split: str
    status: str
    created_at: str
    updated_at: str
    metrics: dict[str, object] | None = None
    failure_type: str | None = None
    failure_message: str | None = None


EvaluationExecutor = Callable[[str, str, str], Awaitable[dict[str, object]]]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class _JobStore:
    def __init__(self, path: Path) -> None:
        ensure_directory(path.parent)
        self._path = path
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_jobs ("
                "run_id TEXT PRIMARY KEY, method TEXT NOT NULL, split TEXT NOT NULL, "
                "status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "metrics_json TEXT, failure_type TEXT, failure_message TEXT)"
            )
            timestamp = _now()
            connection.execute(
                "UPDATE evaluation_jobs SET status='failed', updated_at=?, "
                "failure_type='InterruptedError', "
                "failure_message='job was interrupted by a service restart' "
                "WHERE status IN ('queued', 'running')",
                (timestamp,),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _job(row: tuple[object, ...]) -> EvaluationJob:
        metrics = json.loads(row[6]) if isinstance(row[6], str) else None
        if metrics is not None and not isinstance(metrics, dict):
            raise RuntimeError("stored evaluation metrics must be an object")
        return EvaluationJob(
            run_id=str(row[0]),
            method=str(row[1]),
            split=str(row[2]),
            status=str(row[3]),
            created_at=str(row[4]),
            updated_at=str(row[5]),
            metrics=metrics,
            failure_type=row[7] if isinstance(row[7], str) else None,
            failure_message=row[8] if isinstance(row[8], str) else None,
        )

    def create(self, run_id: str, method: str, split: str) -> EvaluationJob:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_jobs VALUES (?, ?, ?, 'queued', ?, ?, NULL, NULL, NULL)",
                (run_id, method, split, timestamp, timestamp),
            )
        return self.get(run_id)

    def transition(
        self,
        run_id: str,
        status: str,
        *,
        metrics: dict[str, object] | None = None,
        failure_type: str | None = None,
        failure_message: str | None = None,
    ) -> EvaluationJob:
        metrics_json = (
            json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if metrics is not None
            else None
        )
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE evaluation_jobs SET status=?, updated_at=?, metrics_json=?, "
                "failure_type=?, failure_message=? WHERE run_id=?",
                (status, _now(), metrics_json, failure_type, failure_message, run_id),
            )
            if cursor.rowcount != 1:
                raise EvaluationJobNotFound(f"evaluation job not found: {run_id}")
        return self.get(run_id)

    def get(self, run_id: str) -> EvaluationJob:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id, method, split, status, created_at, updated_at, metrics_json, "
                "failure_type, failure_message FROM evaluation_jobs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise EvaluationJobNotFound(f"evaluation job not found: {run_id}")
        return self._job(row)

    def list(self) -> tuple[EvaluationJob, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, method, split, status, created_at, updated_at, metrics_json, "
                "failure_type, failure_message FROM evaluation_jobs "
                "ORDER BY created_at DESC, run_id DESC"
            ).fetchall()
        return tuple(self._job(row) for row in rows)


class EvaluationJobService:
    def __init__(self, database_path: Path, executor: EvaluationExecutor) -> None:
        self._store = _JobStore(database_path)
        self._executor = executor
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def create_job(self, *, method: str, split: str) -> EvaluationJob:
        if method not in _METHODS:
            raise ValueError("evaluation method is not supported")
        if split == "test":
            raise ValueError("Test split is locked until S12")
        if split not in _SPLITS:
            raise ValueError("evaluation split is not supported")
        run_id = f"s9-{uuid4().hex}"
        job = self._store.create(run_id, method, split)
        self._tasks[run_id] = asyncio.create_task(self._execute(job))
        return job

    async def _execute(self, job: EvaluationJob) -> None:
        self._store.transition(job.run_id, "running")
        try:
            metrics = await self._executor(job.run_id, job.method, job.split)
        except asyncio.CancelledError:
            self._store.transition(
                job.run_id,
                "failed",
                failure_type="InterruptedError",
                failure_message="job was interrupted by service shutdown",
            )
            raise
        except Exception as error:  # noqa: BLE001 - persist arbitrary worker failures
            self._store.transition(
                job.run_id,
                "failed",
                failure_type=type(error).__name__,
                failure_message=str(error)[:1000] or "evaluation failed",
            )
        else:
            if not isinstance(metrics, dict):
                self._store.transition(
                    job.run_id,
                    "failed",
                    failure_type="TypeError",
                    failure_message="evaluation executor returned non-object metrics",
                )
            else:
                self._store.transition(job.run_id, "succeeded", metrics=metrics)

    def list_jobs(self) -> tuple[EvaluationJob, ...]:
        return self._store.list()

    def get_job(self, run_id: str) -> EvaluationJob:
        return self._store.get(run_id)

    async def wait(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await task

    async def close(self) -> None:
        tasks = tuple(task for task in self._tasks.values() if not task.done())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
