"""Initialize container data from local assets without embedding or API calls."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex)
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_dense(source, target):
    """Use SQLite's backup API so an open source DB is copied consistently."""
    for item in source.rglob("*"):
        if item.is_symlink():
            raise ValueError("Source index contains a symlink")
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.name != ".lock" and not item.name.endswith(("-wal", "-shm")):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if item.suffix == ".sqlite":
                with sqlite3.connect(
                    item.resolve().as_uri() + "?mode=ro", uri=True
                ) as original, sqlite3.connect(destination) as copied:
                    original.backup(copied)
            else:
                shutil.copyfile(item, destination)


def seed_vectors(source, remote, collection, receipt):
    if not (source / "meta.json").is_file():
        return {"status": "source_not_present", "count": 0}
    with tempfile.TemporaryDirectory(prefix="docguard-vector-seed-") as temporary:
        copied = Path(temporary)
        _copy_dense(source, copied)
        local = QdrantClient(path=str(copied))
        try:
            configuration = local.get_collection(collection).config.params.vectors
            points, offset = [], None
            while True:
                batch, offset = local.scroll(
                    collection,
                    limit=256,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                points.extend(batch)
                if offset is None:
                    break
        finally:
            local.close()
    points.sort(key=lambda point: str(point.id))
    digest = hashlib.sha256()
    for point in points:
        digest.update(
            json.dumps(
                point.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        )
    identity = {
        "source_sha256": digest.hexdigest(),
        "count": len(points),
        "collection": collection,
    }
    previous = (
        json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else None
    )
    exists = remote.collection_exists(collection)
    if exists:
        if previous is None or any(
            previous.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(
                "Refusing to overwrite an existing unmanaged or different collection"
            )
        if previous.get("status") == "ready" and remote.count(
            collection, exact=True
        ).count == len(points):
            return {**identity, "status": "already_present"}
    else:
        _write(receipt, {**identity, "status": "initializing"})
        remote.create_collection(collection, vectors_config=configuration)
    for start in range(0, len(points), 256):
        batch = points[start : start + 256]
        remote.upsert(
            collection,
            [
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in batch
            ],
            wait=True,
        )
    if remote.count(collection, exact=True).count != len(points):
        raise ValueError("Container vector count mismatch")
    _write(receipt, {**identity, "status": "ready"})
    return {**identity, "status": "initialized"}


def seed_knowledge(source, target):
    imported = []
    target.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return imported
    for base in sorted(source.iterdir()):
        if (
            not base.is_dir()
            or base.is_symlink()
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", base.name)
        ):
            continue
        pointer = base / "active.json"
        destination = target / base.name
        if not pointer.is_file() or destination.exists():
            continue
        if pointer.is_symlink():
            raise ValueError("Source knowledge pointer is a symlink")
        pointer_bytes = pointer.read_bytes()
        snapshot = json.loads(pointer_bytes)["snapshot_id"]
        if not re.fullmatch(r"[0-9a-f]{64}", snapshot):
            raise ValueError("Invalid source snapshot ID")
        folder = base / snapshot
        if folder.is_symlink() or any(path.is_symlink() for path in folder.rglob("*")):
            raise ValueError("Source knowledge contains a symlink")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["snapshot_id"] != snapshot
            or manifest["knowledge_base_id"] != base.name
        ):
            raise ValueError("Source knowledge manifest mismatch")
        for name, expected in manifest["files"].items():
            file = folder / name
            if (
                file.resolve().parent != folder.resolve()
                or hashlib.sha256(file.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Source knowledge checksum mismatch")
        stage = target / (".seed-" + uuid4().hex)
        stage.mkdir()
        shutil.copytree(folder, stage / snapshot)
        (stage / "active.json").write_bytes(pointer_bytes)
        # Recheck the copied bytes before making the directory visible.
        for name, expected in manifest["files"].items():
            if (
                hashlib.sha256((stage / snapshot / name).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("Copied knowledge checksum mismatch")
        stage.rename(destination)
        imported.append(base.name)
    return imported


def main():
    project = Path(os.environ.get("DOCGUARD_PROJECT_ROOT", "/app"))
    runtime = Path("/workspace/reports/generated")
    config = json.loads(
        (project / "configs/retrieval/s3-dense-v1.json").read_text(encoding="utf-8")
    )
    remote = QdrantClient(
        url=os.environ.get("DOCGUARD_QDRANT_URL", "http://qdrant:6333"), timeout=60
    )
    try:
        dense = seed_vectors(
            Path("/workspace/.qdrant/s3-dense-v1/qdrant"),
            remote,
            config["collection_name"],
            runtime / "container-seed/postgresql.json",
        )
    finally:
        remote.close()
    knowledge = seed_knowledge(
        Path("/imports/knowledge"), runtime / "s15-discovery/knowledge"
    )
    print(json.dumps({"vectors": dense, "knowledge_bases_imported": knowledge}))


if __name__ == "__main__":
    main()
