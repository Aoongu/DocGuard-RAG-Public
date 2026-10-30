import importlib.util
import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "container_prepare", ROOT / "scripts/prepare_container.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_seed_preserves_payload_and_vectors_without_embedding_and_is_repeatable(
    tmp_path,
):
    module = load_module()
    source = tmp_path / "local"
    local = QdrantClient(path=str(source))
    local.create_collection(
        "frozen",
        vectors_config={
            "dense": models.VectorParams(size=3, distance=models.Distance.COSINE)
        },
    )
    local.upsert(
        "frozen",
        [
            models.PointStruct(
                id=1,
                vector={"dense": [1.0, 0.0, 0.0]},
                payload={"chunk_id": "original"},
            )
        ],
    )
    local.close()
    remote = QdrantClient(location=":memory:")
    receipt = tmp_path / "receipt.json"
    result = module.seed_vectors(source, remote, "frozen", receipt)
    assert result["count"] == 1
    assert remote.retrieve("frozen", [1], with_vectors=True)[0].payload == {
        "chunk_id": "original"
    }
    assert remote.retrieve("frozen", [1], with_vectors=True)[0].vector == {
        "dense": [1.0, 0.0, 0.0]
    }
    assert (
        module.seed_vectors(source, remote, "frozen", receipt)["status"]
        == "already_present"
    )
    remote.close()


def test_seed_does_not_overwrite_an_existing_unmanaged_collection(tmp_path):
    module = load_module()
    source = tmp_path / "local"
    local = QdrantClient(path=str(source))
    local.create_collection(
        "frozen",
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )
    local.upsert("frozen", [models.PointStruct(id=1, vector=[1.0], payload={})])
    local.close()
    remote = QdrantClient(location=":memory:")
    remote.create_collection(
        "frozen",
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )
    with pytest.raises(ValueError, match="existing"):
        module.seed_vectors(source, remote, "frozen", tmp_path / "receipt.json")
    remote.close()


def test_knowledge_seed_copies_only_active_snapshot_and_never_overwrites(tmp_path):
    module = load_module()
    source, target = tmp_path / "source", tmp_path / "target"
    snapshot_id = "a" * 64
    folder = source / "pandas" / snapshot_id
    folder.mkdir(parents=True)
    (source / "pandas/active.json").write_text(json.dumps({"snapshot_id": snapshot_id}))
    (folder / "manifest.json").write_text(
        json.dumps(
            {"snapshot_id": snapshot_id, "knowledge_base_id": "pandas", "files": {}}
        )
    )
    (source / "pandas/.staging-private").mkdir()
    assert module.seed_knowledge(source, target) == ["pandas"]
    assert not (target / "pandas/.staging-private").exists()
    assert module.seed_knowledge(source, target) == []
