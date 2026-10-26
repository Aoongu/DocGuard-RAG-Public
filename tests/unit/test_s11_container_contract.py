from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_api_image_is_non_root_and_excludes_runtime_assets() -> None:
    dockerfile = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert "USER docguard" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "data/models" not in dockerfile
    assert "DEEPSEEK_API_KEY=" not in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile


def test_frontend_image_is_non_root_and_proxies_api() -> None:
    dockerfile = (ROOT / "frontend/Dockerfile").read_text(encoding="utf-8")
    nginx = (ROOT / "frontend/nginx.conf").read_text(encoding="utf-8")
    assert "nginxinc/nginx-unprivileged" in dockerfile
    assert "npm ci" in dockerfile
    assert "proxy_pass http://api:8000" in nginx
    assert "location /api/" in nginx


def test_compose_has_healthchecks_and_named_qdrant_volume() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "api:" in compose and "frontend:" in compose and "qdrant:" in compose
    assert compose.count("healthcheck:") >= 3
    assert "qdrant_data:/qdrant/storage" in compose
    assert "qdrant_data:" in compose
    assert "runtime_reports:/workspace/reports/generated" in compose
    assert "runtime_reports:" in compose
    assert '"6333:6333"' in compose
    assert "DOCGUARD_QDRANT_URL: http://qdrant:6333" in compose
    assert "DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:-}" in compose


def test_docker_context_excludes_secrets_models_and_generated_indexes() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", "data/models/", "data/raw/", ".qdrant/", ".bm25/"):
        assert required in ignored
