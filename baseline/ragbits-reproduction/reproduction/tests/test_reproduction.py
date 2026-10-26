import json
from pathlib import Path

import pytest

from reproduction.run_document_search import QUERIES, run_reproduction


@pytest.mark.asyncio
async def test_local_document_search_returns_expected_top_one_and_writes_results(tmp_path: Path) -> None:
    output_path = tmp_path / "document_search.json"

    results = await run_reproduction(output_path=output_path)

    assert [result["source"] for result in results] == [expected for _, expected in QUERIES]
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == results
    for result, (query, _) in zip(results, QUERIES, strict=True):
        assert result["query"] == query
        assert isinstance(result["source"], str) and result["source"]
        assert isinstance(result["score"], float)
        assert isinstance(result["text"], str) and result["text"].strip()
