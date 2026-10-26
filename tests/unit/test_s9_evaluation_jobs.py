from __future__ import annotations

from pathlib import Path

import pytest

from docguard.service.evaluations import EvaluationJobNotFound, EvaluationJobService


@pytest.mark.asyncio
async def test_job_success_is_persisted_across_service_instances(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []

    async def executor(run_id: str, method: str, split: str) -> dict[str, object]:
        calls.append((run_id, method, split))
        return {"recall_at_10": 0.8}

    path = tmp_path / "jobs.sqlite3"
    service = EvaluationJobService(path, executor)
    created = await service.create_job(method="m4", split="validation")
    await service.wait(created.run_id)

    completed = service.get_job(created.run_id)
    assert completed.status == "succeeded"
    assert completed.metrics == {"recall_at_10": 0.8}
    assert calls == [(created.run_id, "m4", "validation")]
    await service.close()

    reopened = EvaluationJobService(path, executor)
    assert reopened.get_job(created.run_id) == completed
    await reopened.close()


@pytest.mark.asyncio
async def test_job_failure_records_type_and_message(tmp_path: Path) -> None:
    async def executor(run_id: str, method: str, split: str) -> dict[str, object]:
        raise RuntimeError("fixture failed")

    service = EvaluationJobService(tmp_path / "jobs.sqlite3", executor)
    job = await service.create_job(method="m2", split="development")
    await service.wait(job.run_id)

    failed = service.get_job(job.run_id)
    assert failed.status == "failed"
    assert failed.failure_type == "RuntimeError"
    assert failed.failure_message == "fixture failed"
    await service.close()


@pytest.mark.asyncio
async def test_test_split_is_rejected_before_job_creation(tmp_path: Path) -> None:
    async def executor(run_id: str, method: str, split: str) -> dict[str, object]:
        return {}

    service = EvaluationJobService(tmp_path / "jobs.sqlite3", executor)

    with pytest.raises(ValueError, match="Test"):
        await service.create_job(method="m4", split="test")

    assert service.list_jobs() == ()
    await service.close()


@pytest.mark.asyncio
async def test_unknown_job_is_explicit(tmp_path: Path) -> None:
    async def executor(run_id: str, method: str, split: str) -> dict[str, object]:
        return {}

    service = EvaluationJobService(tmp_path / "jobs.sqlite3", executor)

    with pytest.raises(EvaluationJobNotFound, match="not found"):
        service.get_job("missing")
    await service.close()


@pytest.mark.asyncio
async def test_interrupted_jobs_fail_closed_on_restart(tmp_path: Path) -> None:
    async def executor(run_id: str, method: str, split: str) -> dict[str, object]:
        return {}

    path = tmp_path / "jobs.sqlite3"
    service = EvaluationJobService(path, executor)
    service._store.create("run-interrupted", "m4", "validation")
    await service.close()

    reopened = EvaluationJobService(path, executor)
    recovered = reopened.get_job("run-interrupted")
    assert recovered.status == "failed"
    assert recovered.failure_type == "InterruptedError"
    await reopened.close()
