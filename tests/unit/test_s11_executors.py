from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from docguard.engineering.executors import CommandResult, EngineeringExecutors


class FakeRunner:
    def __init__(self, *, returncode: int = 0) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.returncode = returncode

    async def __call__(self, command: tuple[str, ...], cwd: Path) -> CommandResult:
        self.calls.append((command, cwd))
        if "docguard.evaluation.cli" in command:
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "retrieval": {
                            "evaluated_answerable_count": 12,
                            "recall_at_5": 0.75,
                        },
                        "operations": {"success_count": 60, "failed_count": 0},
                    }
                ),
                encoding="utf-8",
            )
        await asyncio.sleep(0)
        return CommandResult(
            self.returncode,
            "ok",
            "DEEPSEEK_API_KEY=secret-value" if self.returncode else "",
        )


@pytest.mark.asyncio
async def test_evaluation_executor_runs_only_frozen_non_test_command(tmp_path: Path) -> None:
    runner = FakeRunner()
    executors = EngineeringExecutors(tmp_path, tmp_path, runner=runner, python="python")
    metrics = await executors.run_evaluation("run-1", "b1", "development")
    command, cwd = runner.calls[0]
    assert cwd == tmp_path
    assert command[:4] == ("python", "-m", "docguard.evaluation.cli", "run")
    assert command[command.index("--dataset") + 1].endswith("development.jsonl")
    assert command[command.index("--method") + 1] == "b1"
    assert metrics == {"evaluated_answerable_count": 12, "recall_at_5": 0.75}

    with pytest.raises(ValueError, match="locked"):
        await executors.run_evaluation("run-test", "b1", "test")
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_evaluation_executor_propagates_sanitized_failure(tmp_path: Path) -> None:
    runner = FakeRunner(returncode=2)
    executors = EngineeringExecutors(tmp_path, tmp_path, runner=runner, python="python")
    with pytest.raises(RuntimeError, match="exit code 2") as caught:
        await executors.run_evaluation("run-1", "m1", "validation")
    assert "secret-value" not in str(caught.value)


@pytest.mark.asyncio
async def test_rebuild_is_single_flight_and_persists_structured_status(tmp_path: Path) -> None:
    runner = FakeRunner()
    executors = EngineeringExecutors(tmp_path, tmp_path, runner=runner, python="python")
    first = await executors.submit_rebuild()
    second = await executors.submit_rebuild()
    assert second == first
    await executors.wait_rebuild(first)
    assert len(runner.calls) == 3
    status = json.loads(
        (tmp_path / "reports/generated/s11-rebuild" / f"{first}.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "succeeded"
    assert status["commands_completed"] == 3
    assert set(status) == {
        "commands_completed",
        "event",
        "job_id",
        "status",
        "updated_at",
    }


def test_executor_rejects_unknown_method_before_spawning(tmp_path: Path) -> None:
    runner = FakeRunner()
    executors = EngineeringExecutors(tmp_path, tmp_path, runner=runner)
    with pytest.raises(ValueError, match="method"):
        asyncio.run(executors.run_evaluation("run-1", "shell", "development"))
    assert runner.calls == []
