"""Prepare real-document tasks and require human expectations before freezing.

This module captures public documentation only. It never calls a model/search
provider, never runs an evaluation, and does not create or unlock a Test split.
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from docguard.discovery.dns import DohResolver
from docguard.discovery.evaluation_fixtures import digest, write_json
from docguard.discovery.extract import extract_html
from docguard.discovery.fetch import FetchError, SafeFetcher, validate_url
from docguard.discovery.graph import AgentLimits
from docguard.discovery.models import CreateTask


def task_request(case):
    return {
        "goal": case["goal"],
        "product": case["product"],
        "version": case["version"],
        "official_only": True,
    }


def _validate_tasks(tasks):
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Unique nonempty tasks required")
    groups = {}
    for task in tasks:
        if task["split"] not in ("development", "validation"):
            raise ValueError("Test preparation is locked")
        CreateTask(**task_request(task))
        if task["expected_outcome"] not in ("discover", "clarify", "no_result"):
            raise ValueError("Invalid expected outcome")
        if bool(task["acceptable_urls"]) != (task["expected_outcome"] == "discover"):
            raise ValueError("Only discovery tasks have expected sources")
        for url in task["acceptable_urls"]:
            if validate_url(url) != url or "/s15-fixture/" in url:
                raise ValueError("Real canonical source required")
        groups.setdefault(task["split"], {"products": set(), "templates": set()})
        groups[task["split"]]["products"].add(task["product_family"])
        groups[task["split"]]["templates"].add(task["template_family"])
    if set(groups) != {"development", "validation"} or any(
        groups["development"][name] & groups["validation"][name]
        for name in ("products", "templates")
    ):
        raise ValueError("Disjoint product and template groups required")


def prepare_suite(proposals, output):
    tasks = json.loads(Path(proposals).read_text(encoding="utf-8"))["tasks"]
    _validate_tasks(tasks)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "tasks.jsonl").write_text(
        "".join(
            json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in tasks
        ),
        encoding="utf-8",
        newline="\n",
    )
    protocol = {
        "kind": "real_document_tasks_before_evaluation",
        "task_author": "assistant_authored",
        "test_execution_status": "locked",
        "test_dataset_status": "not_created",
        "limits_per_method_task": asdict(AgentLimits()),
        "methods_planned": ["given_url", "single_search", "real_model_agent"],
        "model_cost_ceiling_usd": len(tasks) * AgentLimits().max_model_cost_usd,
        "search_call_ceiling": len(tasks) * (1 + AgentLimits().max_searches),
        "search_cost_usd": None,
        "search_provider_selection": "serpapi_google",
        "paid_execution_implemented": False,
        "automatic_retries": 0,
    }
    write_json(root / "protocol.json", protocol)
    review = {
        "human_validation_status": "pending",
        "tasks_sha256": digest(root / "tasks.jsonl"),
        "protocol_sha256": digest(root / "protocol.json"),
        "source_manifest_sha256": None,
        "reviews": [
            {
                "task_id": t["task_id"],
                "expected_outcome": t["expected_outcome"],
                "task": t,
                "source_snapshots": [],
                "review_status": "pending",
                "reviewer": None,
                "expectation_correct": None,
                "source_acceptable": None,
                "ambiguity_check_passed": None,
                "notes": None,
            }
            for t in tasks
        ],
    }
    write_json(root / "expectation_review.json", review)
    write_json(
        root / "preparation_manifest.json",
        {
            "status": "draft",
            "files": {
                name: digest(root / name) for name in ("tasks.jsonl", "protocol.json")
            },
        },
    )
    return protocol


def _draft(root):
    root = Path(root)
    manifest = json.loads(
        (root / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError("Draft hash mismatch")
    tasks = [
        json.loads(line)
        for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    _validate_tasks(tasks)
    review = json.loads((root / "expectation_review.json").read_text(encoding="utf-8"))
    if review["tasks_sha256"] != digest(root / "tasks.jsonl"):
        raise ValueError("Review task hash mismatch")
    if review.get("protocol_sha256") != digest(root / "protocol.json"):
        raise ValueError("Review protocol hash mismatch")
    return root, tasks, review


def capture_sources(root, fetcher, *, resolution_identity="injected_fetcher"):
    root, tasks, review = _draft(root)
    if review["human_validation_status"] != "pending" or any(
        r["review_status"] != "pending" or r["reviewer"] is not None
        for r in review["reviews"]
    ):
        raise ValueError("Capture must precede human review")
    folder = root / "sources"
    folder.mkdir(exist_ok=False)
    records, files = {}, {}
    for url in sorted({url for t in tasks for url in t["acceptable_urls"]}):
        try:
            snapshot = fetcher.fetch(url)
            if hashlib.sha256(snapshot.body).hexdigest() != snapshot.content_sha256:
                raise ValueError("Snapshot hash mismatch")
            page = extract_html(snapshot.body, snapshot.final_url)
            name = hashlib.sha256(url.encode()).hexdigest()
            html_path, text_path = (
                "sources/" + name + ".html",
                "sources/" + name + ".txt",
            )
            (root / html_path).write_bytes(snapshot.body)
            (root / text_path).write_text(
                "\n\n".join(b.text for b in page.blocks) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            metadata = asdict(snapshot)
            metadata.pop("body")
            records[url] = {
                **metadata,
                "status": "captured",
                "html_path": html_path,
                "text_path": text_path,
                "title": page.title,
                "block_count": len(page.blocks),
            }
            files.update({name: digest(root / name) for name in (html_path, text_path)})
        except (OSError, ValueError, RuntimeError) as error:
            records[url] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "reason_code": error.reason_code
                if isinstance(error, FetchError)
                else "source_preparation_failed",
            }
    result = {
        "status": "complete"
        if all(r["status"] == "captured" for r in records.values())
        else "partial",
        "sources": records,
        "files": files,
        "model_or_search_calls": 0,
        "resolution_identity": resolution_identity,
    }
    write_json(folder / "manifest.json", result)
    review["source_manifest_sha256"] = digest(folder / "manifest.json")
    for row in review["reviews"]:
        case = next(t for t in tasks if t["task_id"] == row["task_id"])
        row["source_snapshots"] = [records[url] for url in case["acceptable_urls"]]
    write_json(root / "expectation_review.json", review)
    return result


def freeze_suite(root, output):
    root, tasks, review = _draft(root)
    if review["human_validation_status"] != "complete":
        raise ValueError("Prior human review is incomplete")
    records = {r["task_id"]: r for r in review["reviews"]}
    if (
        len(records) != len(tasks)
        or len(review["reviews"]) != len(tasks)
        or set(records) != {t["task_id"] for t in tasks}
    ):
        raise ValueError("Human review identities mismatch")
    for case in tasks:
        row = records[case["task_id"]]
        if (
            row["task"] != case
            or row["expected_outcome"] != case["expected_outcome"]
            or row["review_status"] != "confirmed"
            or not isinstance(row["reviewer"], str)
            or not row["reviewer"].strip()
            or row["expectation_correct"] is not True
            or row["ambiguity_check_passed"] is not True
            or not isinstance(row["notes"], str)
            or not row["notes"].strip()
        ):
            raise ValueError("Human review confirmation is incomplete")
        if (
            case["expected_outcome"] == "discover"
            and row["source_acceptable"] is not True
        ) or (
            case["expected_outcome"] != "discover"
            and row["source_acceptable"] is not None
        ):
            raise ValueError("Human source review mismatch")
    sources = json.loads((root / "sources/manifest.json").read_text(encoding="utf-8"))
    if (
        review["source_manifest_sha256"] != digest(root / "sources/manifest.json")
        or sources["status"] != "complete"
    ):
        raise ValueError("Source hash mismatch or incomplete capture")
    expected_urls = {url for task in tasks for url in task["acceptable_urls"]}
    if set(sources["sources"]) != expected_urls:
        raise ValueError("Source identities mismatch")
    for case in tasks:
        expected = [sources["sources"][url] for url in case["acceptable_urls"]]
        if records[case["task_id"]]["source_snapshots"] != expected:
            raise ValueError("Human source review binding mismatch")
    files = {
        name: digest(root / name)
        for name in (
            "tasks.jsonl",
            "protocol.json",
            "expectation_review.json",
            "sources/manifest.json",
        )
    }
    if (root / "capture_origins.json").exists():
        files["capture_origins.json"] = digest(root / "capture_origins.json")
    for name, expected in sources["files"].items():
        if (
            not (root / name).resolve().is_relative_to(root.resolve())
            or digest(root / name) != expected
        ):
            raise ValueError("Source file hash mismatch")
        files[name] = expected
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination / name)
    write_json(
        destination / "manifest.json",
        {
            "status": "frozen",
            "kind": "real_document_task_expectations",
            "human_expectations_status": "complete",
            "task_count": len(tasks),
            "test_execution_status": "locked",
            "test_dataset_status": "not_created",
            "paid_evaluation_status": "not_started",
            "files": files,
        },
    )
    return verify_frozen(destination)


def verify_frozen(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "frozen" or manifest["test_execution_status"] != "locked":
        raise ValueError("Invalid frozen real-task manifest")
    for name, expected in manifest["files"].items():
        if (
            not (root / name).resolve().is_relative_to(root.resolve())
            or digest(root / name) != expected
        ):
            raise ValueError("Frozen real-task hash mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--proposals", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--suite", type=Path, required=True)
    capture.add_argument("--dns-mode", choices=("system", "doh"), default="system")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--suite", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_suite(args.proposals, args.output)
        print(
            json.dumps(
                {
                    "status": "draft",
                    "test_execution_status": result["test_execution_status"],
                }
            )
        )
    elif args.command == "capture":
        fetcher = (
            SafeFetcher(resolver=DohResolver())
            if args.dns_mode == "doh"
            else SafeFetcher()
        )
        result = capture_sources(
            args.suite,
            fetcher,
            resolution_identity="cloudflare_doh_pinned_1.1.1.1"
            if args.dns_mode == "doh"
            else "system_dns",
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "source_count": len(result["sources"]),
                    "model_or_search_calls": 0,
                }
            )
        )
    else:
        print(json.dumps(freeze_suite(args.suite, args.output)))


if __name__ == "__main__":
    main()
