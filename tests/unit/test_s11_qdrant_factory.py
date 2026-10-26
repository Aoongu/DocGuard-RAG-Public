from __future__ import annotations

from pathlib import Path

import pytest

from docguard.retrieval.qdrant_factory import create_qdrant_client


@pytest.mark.asyncio
async def test_qdrant_factory_supports_local_and_service_modes(tmp_path: Path) -> None:
    local = create_qdrant_client(path=tmp_path / "qdrant")
    service = create_qdrant_client(
        url="http://qdrant:6333", check_compatibility=False
    )
    assert "AsyncQdrantLocal" in type(local._client).__name__
    assert service._client is not None
    await local.close()
    await service.close()


def test_qdrant_factory_requires_exactly_one_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        create_qdrant_client()
    with pytest.raises(ValueError, match="exactly one"):
        create_qdrant_client(path=tmp_path, url="http://qdrant:6333")
    with pytest.raises(ValueError, match="http"):
        create_qdrant_client(url="file:///tmp/qdrant")
    with pytest.raises(TypeError, match="boolean"):
        create_qdrant_client(url="http://qdrant:6333", check_compatibility=1)  # type: ignore[arg-type]
