import json
from pathlib import Path

from reproduction.smoke_offline_chat import run_smoke


def test_offline_chat_smoke_uses_upstream_example_and_public_endpoints(tmp_path: Path) -> None:
    output_path = tmp_path / "offline_chat_smoke.json"

    result = run_smoke(output_path=output_path)

    assert result["upstream_example_class"] == "examples.chat.offline_chat:MyChat"
    assert result["route_count"] > 0
    assert [(request["endpoint"], request["status_code"]) for request in result["requests"]] == [
        ("/openapi.json", 200),
        ("/api/config", 200),
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
