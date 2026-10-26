from __future__ import annotations

import json
from pathlib import Path

import pytest


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, int, bool]] = []

    async def search(
        self,
        query: str,
        version: str | None,
        top_k: int,
        dense_enabled: bool = True,
    ) -> list[object]:
        self.calls.append((query, version, top_k, dense_enabled))
        return []


def _write_queries(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


@pytest.mark.asyncio
async def test_smoke_runner_emits_one_stable_json_object_per_query(tmp_path: Path) -> None:
    from docguard.retrieval.service import run_smoke

    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "results.jsonl"
    rows = [
        {
            "schema_version": 1,
            "query_id": "q02",
            "query": "How does VACUUM work?",
            "version": "18.6",
            "top_k": 5,
        },
        {
            "schema_version": 1,
            "query_id": "q01",
            "query": "What is MVCC?",
            "version": None,
            "top_k": 5,
        },
    ]
    _write_queries(queries, rows)
    retriever = FakeRetriever()

    assert await run_smoke(queries, output, retriever) == 2
    raw = output.read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    actual = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert [row["query_id"] for row in actual] == ["q02", "q01"]
    assert all(row["hits"] == [] for row in actual)
    assert retriever.calls == [
        ("How does VACUUM work?", "18.6", 5, True),
        ("What is MVCC?", None, 5, True),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows.append(dict(rows[0])),
        lambda rows: rows[0].update(version="16"),
        lambda rows: rows[0].update(top_k=4),
        lambda rows: rows[0].update(schema_version=2),
        lambda rows: rows[0].update(extra=True),
    ],
)
async def test_smoke_runner_rejects_invalid_dataset(tmp_path: Path, mutator: object) -> None:
    from docguard.retrieval.service import run_smoke

    queries = tmp_path / "queries.jsonl"
    rows = [
        {
            "schema_version": 1,
            "query_id": "q01",
            "query": "What is MVCC?",
            "version": "17.11",
            "top_k": 5,
        }
    ]
    mutator(rows)  # type: ignore[operator]
    _write_queries(queries, rows)
    with pytest.raises(ValueError):
        await run_smoke(queries, tmp_path / "results.jsonl", FakeRetriever())


def test_official_smoke_dataset_is_frozen_to_ten_queries() -> None:
    path = Path("data/eval/s3-smoke-v1.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10
    assert len({row["query_id"] for row in rows}) == 10
    assert {row["version"] for row in rows} == {None, "17.11", "18.6"}
    assert all(row["schema_version"] == 1 and row["top_k"] == 5 for row in rows)
