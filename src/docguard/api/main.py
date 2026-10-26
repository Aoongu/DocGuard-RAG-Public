"""Import-safe ASGI entry point: ``docguard.api.main:app``."""

from __future__ import annotations

import os
from pathlib import Path

from docguard.api.app import create_app
from docguard.api.config import load_api_config
from docguard.api.dependencies import AppServices
from docguard.api.runtime import (
    LazyQueryService,
    build_live_query_service,
    dependency_probe,
)
from docguard.engineering.config import (
    load_engineering_config,
    verify_repository_compatibility,
)
from docguard.engineering.executors import EngineeringExecutors
from docguard.service.artifacts import ArtifactService
from docguard.service.evaluations import EvaluationJobService

PROJECT_ROOT = Path(os.environ.get("DOCGUARD_PROJECT_ROOT", Path(__file__).parents[3])).resolve()
ASSET_ROOT = Path(os.environ.get("DOCGUARD_ASSET_ROOT", PROJECT_ROOT)).resolve()
CONFIG = load_api_config(PROJECT_ROOT / "configs/api/s9-api-v1.json")
ENGINEERING_CONFIG = load_engineering_config(
    PROJECT_ROOT / "configs/engineering/s11-engineering-v1.json"
)
verify_repository_compatibility(PROJECT_ROOT, ENGINEERING_CONFIG)

QUERY = LazyQueryService(
    lambda: build_live_query_service(PROJECT_ROOT, ASSET_ROOT, CONFIG.query_timeout_seconds)
)
EXECUTORS = EngineeringExecutors(PROJECT_ROOT, ASSET_ROOT)
ARTIFACTS = ArtifactService(
    ASSET_ROOT / "data/processed/s2-chunk-v1/metadata.sqlite3",
    ASSET_ROOT / "data/processed/s2-chunk-v1/stats.json",
    PROJECT_ROOT / "data/manifests/s2-chunk-v1.json",
    rebuild_submitter=EXECUTORS.submit_rebuild,
)
EVALUATIONS = EvaluationJobService(
    ASSET_ROOT / "reports/generated/s9-evaluation-jobs.sqlite3",
    EXECUTORS.run_evaluation,
)
SERVICES = AppServices(
    query=QUERY,
    artifacts=ARTIFACTS,
    evaluations=EVALUATIONS,
    dependency_probe=lambda: dependency_probe(PROJECT_ROOT, ASSET_ROOT),
    shutdown=EXECUTORS.close,
)

app = create_app(CONFIG, SERVICES)


def main() -> None:
    """Run the repository-local API server without enabling debug reload."""
    import uvicorn

    uvicorn.run(
        "docguard.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
