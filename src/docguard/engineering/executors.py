"""Allow-list-only background executors for S11 production wiring."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[tuple[str, ...], Path], Awaitable[CommandResult]]


async def run_command(command: tuple[str, ...], cwd: Path) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    stdout, stderr = await process.communicate()
    return CommandResult(
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class EngineeringExecutors:
    """Run only repository-owned, frozen commands and expose sanitized outcomes."""

    _RETRIEVAL_METHODS = frozenset({"b0", "b1", "m1", "m2"})
    _METHODS = frozenset({*_RETRIEVAL_METHODS, "b2", "m4"})
    _SPLITS = frozenset({"development", "validation"})

    def __init__(
        self,
        project_root: Path,
        asset_root: Path,
        *,
        runner: CommandRunner = run_command,
        python: str = sys.executable,
    ) -> None:
        self._project_root = project_root.resolve()
        self._asset_root = asset_root.resolve()
        self._runner = runner
        self._python = python
        self._rebuild_task: asyncio.Task[None] | None = None
        self._rebuild_id: str | None = None
        self._rebuild_tasks: dict[str, asyncio.Task[None]] = {}

    def _run_root(self, run_id: str) -> Path:
        return self._asset_root / "reports/generated/s11-evaluations" / run_id

    async def _execute(self, command: tuple[str, ...], label: str) -> None:
        result = await self._runner(command, self._project_root)
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    def _retrieval_command(self, run_id: str, method: str, split: str) -> tuple[str, ...]:
        dataset = self._project_root / f"data/eval/s4-golden-v1/{split}.jsonl"
        output = self._run_root(run_id)
        return (
            self._python,
            "-m",
            "docguard.evaluation.cli",
            "run",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--run-id",
            run_id,
            "--method",
            method,
            "--eval-config",
            str(self._project_root / "configs/evaluation/s4-eval-v1.json"),
            "--dense-config",
            str(self._project_root / "configs/retrieval/s3-dense-v1.json"),
            "--s5-config",
            str(self._project_root / "configs/retrieval/s5-hybrid-v1.json"),
            "--s6-config",
            str(self._project_root / "configs/retrieval/s6-reranker-v1.json"),
            "--index-root",
            str(self._asset_root / ".qdrant/s3-dense-v1"),
            "--bm25-index-root",
            str(self._asset_root / ".bm25/s5-bm25-v1"),
            "--model-root",
            str(self._asset_root / "data/models/s3-e5-small"),
            "--reranker-model-root",
            str(self._asset_root / "data/models/s6-bge-reranker-base"),
        )

    async def run_evaluation(
        self, run_id: str, method: str, split: str
    ) -> dict[str, object]:
        if split == "test":
            raise ValueError("Test split is locked until S12")
        if split not in self._SPLITS:
            raise ValueError("evaluation split is not supported")
        if method not in self._METHODS:
            raise ValueError("evaluation method is not supported")
        if method not in self._RETRIEVAL_METHODS:
            raise RuntimeError(
                f"{method} online execution requires an explicit S12 run protocol"
            )
        command = self._retrieval_command(run_id, method, split)
        await self._execute(command, f"{method} evaluation")
        summary_path = self._run_root(run_id) / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["retrieval"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("evaluation summary is unavailable or invalid") from error
        if not isinstance(metrics, dict):
            raise TypeError("evaluation metrics must be an object")
        return metrics

    def _rebuild_commands(self) -> tuple[tuple[str, ...], ...]:
        chunks = self._asset_root / "data/processed/s2-chunk-v1/chunks.jsonl"
        return (
            (
                self._python,
                "-m",
                "docguard.processing.pipeline",
                "build",
                "--raw-root",
                str(self._asset_root / "data/raw"),
                "--output-root",
                str(self._asset_root / "data/processed/s2-chunk-v1"),
                "--config",
                str(self._project_root / "configs/chunking/s2-chunk-v1.json"),
                "--tokenizer-manifest",
                str(self._project_root / "data/tokenizers/manifest.json"),
                "--snapshot-manifest",
                str(self._project_root / "data/manifests/snapshot.jsonl"),
            ),
            (
                self._python,
                "-m",
                "docguard.retrieval.pipeline",
                "build",
                "--chunks",
                str(chunks),
                "--output-root",
                str(self._asset_root / ".qdrant/s3-dense-v1"),
            ),
            (
                self._python,
                "-m",
                "docguard.retrieval.hybrid_pipeline",
                "build",
                "--chunks",
                str(chunks),
                "--config",
                str(self._project_root / "configs/retrieval/s5-hybrid-v1.json"),
                "--output-root",
                str(self._asset_root / ".bm25/s5-bm25-v1"),
            ),
        )

    def _status_path(self, job_id: str) -> Path:
        return self._asset_root / "reports/generated/s11-rebuild" / f"{job_id}.json"

    def _write_status(self, job_id: str, status: str, completed: int) -> None:
        path = self._status_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "commands_completed": completed,
            "event": "corpus_rebuild",
            "job_id": job_id,
            "status": status,
            "updated_at": _now(),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)

    async def _run_rebuild(self, job_id: str) -> None:
        completed = 0
        self._write_status(job_id, "running", completed)
        try:
            for command in self._rebuild_commands():
                await self._execute(command, "corpus rebuild")
                completed += 1
                self._write_status(job_id, "running", completed)
        except Exception:
            self._write_status(job_id, "failed", completed)
            raise
        else:
            self._write_status(job_id, "succeeded", completed)

    async def submit_rebuild(self) -> str:
        if self._rebuild_task is not None and not self._rebuild_task.done():
            assert self._rebuild_id is not None
            return self._rebuild_id
        job_id = f"s11-rebuild-{uuid4().hex}"
        task = asyncio.create_task(self._run_rebuild(job_id))
        self._rebuild_id = job_id
        self._rebuild_task = task
        self._rebuild_tasks[job_id] = task
        return job_id

    async def wait_rebuild(self, job_id: str) -> None:
        task = self._rebuild_tasks.get(job_id)
        if task is None:
            raise LookupError(f"rebuild job not found: {job_id}")
        await task

    async def close(self) -> None:
        tasks = tuple(task for task in self._rebuild_tasks.values() if not task.done())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
