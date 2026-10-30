import json

import pytest


def proposals():
    return {
        "tasks": [
            {
                "task_id": "real-dev-001",
                "split": "development",
                "product_family": "python",
                "template_family": "python-docs",
                "product": "Python",
                "version": "3",
                "goal": "Find pathlib mkdir documentation",
                "expected_outcome": "discover",
                "acceptable_urls": ["https://docs.python.org/3/library/pathlib.html"],
                "required_terms": ["mkdir"],
                "clarification_requirements": [],
            },
            {
                "task_id": "real-val-001",
                "split": "validation",
                "product_family": "redis",
                "template_family": "redis-docs",
                "product": "Redis",
                "version": None,
                "goal": "Ask which deployed version is required",
                "expected_outcome": "clarify",
                "acceptable_urls": [],
                "required_terms": [],
                "clarification_requirements": ["version"],
            },
        ]
    }


def prepare(tmp_path):
    from docguard.discovery.real_tasks import prepare_suite

    source = tmp_path / "proposals.json"
    source.write_text(json.dumps(proposals()), encoding="utf-8")
    root = tmp_path / "draft"
    prepare_suite(source, root)
    return root


def test_real_suite_has_no_test_and_cannot_freeze_without_prior_human_review(tmp_path):
    from docguard.discovery.real_tasks import freeze_suite, task_request

    root = prepare(tmp_path)
    with pytest.raises(ValueError, match="review"):
        freeze_suite(root, tmp_path / "frozen")
    case = proposals()["tasks"][0]
    assert set(task_request(case)) == {"goal", "product", "version", "official_only"}
    assert "acceptable_urls" not in task_request(case)
    assert not (tmp_path / "frozen").exists()


def test_real_suite_capture_and_freeze_bind_review_to_exact_pages(tmp_path):
    import hashlib

    from docguard.discovery.fetch import Snapshot
    from docguard.discovery.real_tasks import (
        capture_sources,
        freeze_suite,
        verify_frozen,
    )

    root = prepare(tmp_path)
    body = b"<title>Python pathlib</title><main><h1>Path.mkdir</h1><p>mkdir creates a directory.</p></main>"

    class Fetcher:
        def fetch(self, url):
            return Snapshot(
                url,
                url,
                "2026-10-27T00:00:00Z",
                hashlib.sha256(body).hexdigest(),
                body,
                "https://docs.python.org/robots.txt",
                404,
                hashlib.sha256(b"").hexdigest(),
            )

    capture_sources(root, Fetcher())
    packet_path = root / "expectation_review.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["human_validation_status"] = "complete"
    for row in packet["reviews"]:
        row.update(
            review_status="confirmed",
            reviewer="unit_test_reviewer",
            expectation_correct=True,
            ambiguity_check_passed=True,
            source_acceptable=True if row["expected_outcome"] == "discover" else None,
            notes="Unit test only",
        )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    packet["reviews"][0]["source_snapshots"][0]["content_sha256"] = "0" * 64
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError, match="source review binding"):
        freeze_suite(root, tmp_path / "invalid-binding")
    packet["reviews"][0]["source_snapshots"][0]["content_sha256"] = hashlib.sha256(
        body
    ).hexdigest()
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    output = tmp_path / "frozen"
    freeze_suite(root, output)
    assert verify_frozen(output)["test_execution_status"] == "locked"
    with pytest.raises(FileExistsError):
        freeze_suite(root, output)
    with (output / "tasks.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        verify_frozen(output)


def test_real_suite_rejects_test_cases_and_changed_review_binding(tmp_path):
    from docguard.discovery.real_tasks import capture_sources, prepare_suite

    source = tmp_path / "bad.json"
    payload = proposals()
    payload["tasks"][0]["split"] = "test"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Test"):
        prepare_suite(source, tmp_path / "bad")
    root = prepare(tmp_path)
    with (root / "tasks.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        capture_sources(root, object())
