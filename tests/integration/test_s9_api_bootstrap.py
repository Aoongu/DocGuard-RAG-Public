from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_main_app_import_does_not_require_models_or_secrets(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    module = importlib.import_module("docguard.api.main")

    with TestClient(module.app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert "openapi" in module.app.openapi_url


def test_main_app_query_fails_closed_without_runtime_dependencies(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    module = importlib.import_module("docguard.api.main")

    with TestClient(module.app) as client:
        response = client.post("/api/v1/query", json={"question": "How does VACUUM work?"})

    assert response.status_code == 503
    assert response.json()["decision"]["action"] == "error"
    assert response.json()["decision"]["reason_code"] == "UPSTREAM_ERROR"


def test_console_entrypoint_runs_frozen_asgi_target(monkeypatch) -> None:
    module = importlib.import_module("docguard.api.main")
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        "uvicorn.run", lambda target, **kwargs: calls.append((target, kwargs))
    )
    module.main()

    assert calls == [
        (
            "docguard.api.main:app",
            {"host": "127.0.0.1", "port": 8000, "reload": False},
        )
    ]
