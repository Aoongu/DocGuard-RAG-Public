"""Smoke-test the upstream offline chat example without starting a server."""

import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.chat.offline_chat import MyChat  # noqa: E402
from ragbits.chat.api import RagbitsAPI  # noqa: E402

UPSTREAM_EXAMPLE_CLASS = "examples.chat.offline_chat:MyChat"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "offline_chat_smoke.json"


def run_smoke(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Construct the upstream app, call public endpoints, and save evidence."""
    api = RagbitsAPI(MyChat)

    with TestClient(api.app) as client:
        openapi_response = client.get("/openapi.json")
        config_response = client.get("/api/config")

    assert openapi_response.status_code == 200, openapi_response.text
    assert config_response.status_code == 200, config_response.text

    openapi = openapi_response.json()
    config = config_response.json()
    result = {
        "upstream_example_class": UPSTREAM_EXAMPLE_CLASS,
        "route_count": len(api.app.routes),
        "requests": [
            {
                "endpoint": "/openapi.json",
                "status_code": openapi_response.status_code,
                "response_summary": {
                    "content_type": openapi_response.headers.get("content-type"),
                    "title": openapi.get("info", {}).get("title"),
                    "openapi_version": openapi.get("openapi"),
                    "documented_path_count": len(openapi.get("paths", {})),
                },
            },
            {
                "endpoint": "/api/config",
                "status_code": config_response.status_code,
                "response_summary": {
                    "content_type": config_response.headers.get("content-type"),
                    "top_level_keys": sorted(config),
                    "authentication_enabled": config.get("authentication", {}).get("enabled"),
                    "supports_upload": config.get("supports_upload"),
                    "debug_mode": config.get("debug_mode"),
                    "header_title": (config.get("customization") or {}).get("header", {}).get("title"),
                },
            },
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run_smoke()
