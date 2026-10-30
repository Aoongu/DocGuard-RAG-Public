"""Offline comparison of fixed/adaptive local policies using production tools.

This runner never constructs a real provider or reads API credentials. Test stays
locked; all approval is explicit simulation in isolated evaluation directories.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from docguard.discovery.evaluation_fixtures import (
    digest,
    load_suite,
    prepare_suite,
    write_json,
)
from docguard.discovery.evaluation_metrics import paired_comparisons, summarize
from docguard.discovery.fetch import Snapshot
from docguard.discovery.graph import AgentLimits
from docguard.discovery.importer import DiscoveryService
from docguard.discovery.providers import Decision, SearchHit
from docguard.knowledge.pipeline import KnowledgeBase
from docguard.processing.config import load_chunking_config
from docguard.processing.token_budget import TokenBudget

METHODS = ("given_url", "single_search_replay", "adaptive_replay")


class ReplaySearch:
    def __init__(self, case):
        # Only the tool environment has access to frozen observations.
        self.goal = case["goal"]
        self.initial, self.refined = case["initial_search"], case["refined_search"]
        self.calls = []

    async def search(self, query, limit=5):
        self.calls.append({"query": query, "limit": limit})
        urls = self.initial if query == self.goal else self.refined
        if urls is None:
            raise RuntimeError("Synthetic upstream search failure")
        return [
            SearchHit(
                "Synthetic documentation result",
                url,
                "Untrusted synthetic search snippet",
            )
            for url in urls[:limit]
        ]


class ReplayFetcher:
    def __init__(self, root, pages):
        self.root, self.pages, self.calls = root, pages, []

    def fetch(self, url):
        self.calls.append(url)
        record = self.pages[url]
        body = (self.root / record["path"]).read_bytes()
        return Snapshot(
            url,
            url,
            "2026-10-27T00:00:00Z",
            hashlib.sha256(body).hexdigest(),
            body,
            "https://synthetic.example.org/robots.txt",
            404,
            hashlib.sha256(b"").hexdigest(),
        )


class ReferencePolicy:
    """Observation-driven local policy. Receives only production model context.

    No case/expected/scenario/gold data is supplied to this policy. This is an
    engineering reference algorithm, not an LLM or a replay of LLM decisions.
    """

    def __init__(self, *, adaptive):
        self.adaptive, self.searches, self.tried = adaptive, 0, set()

    async def decide(self, context):
        if context["version"] is None:
            action = {
                "action": "ask_user",
                "question": "Which documentation version should be used?",
            }
        elif context["candidates"]:
            action = {
                "action": "present_candidates",
                "candidate_ids": [c["candidate_id"] for c in context["candidates"]],
            }
        elif context["last_observation"].get("error_type") == "RuntimeError":
            action = {
                "action": "finish",
                "reason": "Upstream tool failed; preserve failure",
            }
        else:
            unread = [c for c in context["pool"] if c["candidate_id"] not in self.tried]
            if unread:
                candidate_id = unread[0]["candidate_id"]
                self.tried.add(candidate_id)
                action = {"action": "read_page", "candidate_id": candidate_id}
            elif not self.searches or (self.adaptive and self.searches < 3):
                query = (
                    context["goal"]
                    if not self.searches
                    else f"{context['product']} {context['version']} {context['goal']} official documentation"
                )
                self.searches += 1
                action = {"action": "search_documents", "query": query}
            else:
                action = {
                    "action": "finish",
                    "reason": "No qualified inspected page within workflow",
                }
        return Decision(action=action, model="local-reference-policy-not-llm")


async def _settled(service, task_id):
    async with asyncio.timeout(30):
        while True:
            row = service.get_task(task_id)
            if row["status"] not in ("discovering", "importing"):
                return row
            await asyncio.sleep(0.01)


async def _run_case(case, method, suite_root, output, budget, config, pages, protocol):
    relative = Path("cases") / case["case_id"] / method
    root = output / relative
    root.mkdir(parents=True, exist_ok=False)
    fetcher, search = ReplayFetcher(suite_root, pages), ReplaySearch(case)
    policy = ReferencePolicy(adaptive=method == "adaptive_replay")
    limits = replace(
        AgentLimits(**protocol["limits"]), max_model_calls=case["max_model_calls"]
    )
    base = KnowledgeBase(root / "knowledge", budget, config)
    service = DiscoveryService(
        root / "tasks",
        lambda: base,
        fetcher=fetcher,
        decision_model=policy,
        search_provider=search,
        agent_limits=limits,
    )
    started = time.perf_counter()
    candidates, answer = [], None
    failure_codes, observed_issues, import_failure = [], [], None
    row = None
    outcome, status = "no_result", "not_applicable"
    discovery_status = status
    imported_count, wrong_imports, retrieval_success = 0, 0, False
    try:
        if method == "given_url":
            if case["direct_urls"]:
                row = await service.create(
                    case["goal"], case["product"], case["version"], case["direct_urls"]
                )
                row = await _settled(service, row["task_id"])
        else:
            row = service.store.create(
                {
                    "goal": case["goal"],
                    "product": case["product"],
                    "version": case["version"],
                    "urls": [],
                    "official_only": True,
                    "mode": "agent",
                }
            )
            await service.agent.run(row["task_id"])
            row = service.get_task(row["task_id"])
        if row is not None:
            status = row["status"]
            discovery_status = status
            # Inspected but unpresented candidates are never counted as discovery.
            candidates = row["candidates"] if status == "awaiting_selection" else []
            outcome = (
                "discover"
                if candidates
                else "clarify"
                if status == "awaiting_input"
                else "no_result"
            )
            if candidates:
                await service.approve(
                    row["task_id"],
                    [c["candidate_id"] for c in candidates],
                    row["revision"],
                    "eval-kb",
                    "sandbox-simulated-approval",
                )
                imported = await _settled(service, row["task_id"])
                status = imported["status"]
                if imported["job"]["status"] == "succeeded":
                    imported_count = len(candidates)
                    wrong_imports = sum(
                        c["url"] not in case["gold_urls"] for c in candidates
                    )
                    answer = await base.answer("eval-kb", case["question"])
                    retrieval_success = wrong_imports == 0 and any(
                        case["retrieval_marker"] in c["text"]
                        and c["canonical_url"] in case["gold_urls"]
                        for c in answer["citations"]
                    )
                else:
                    import_failure = imported["job"]["failure_type"]
                    failure_codes.append("import_failure")
            events = service.events(row["task_id"])
            for event in events:
                reason = event["data"].get("observation", {}).get("reason")
                if reason in (
                    "tool_failed",
                    "version_mismatch",
                    "official_source_not_confirmed",
                ):
                    observed_issues.append(
                        "tool_failure" if reason == "tool_failed" else reason
                    )
            if not candidates:
                failure_codes.extend(observed_issues)
            if status == "exhausted":
                failure_codes.append("budget_exhausted")
            if case["expected_outcome"] == "discover" and not candidates:
                failure_codes.append("discovery_miss")
            if candidates and not retrieval_success:
                failure_codes.append("import_or_retrieval_miss")
            if wrong_imports:
                failure_codes.append("wrong_import")
            write_json(
                root / "trace.json",
                {
                    "task": service.get_task(row["task_id"]),
                    "events": events,
                    "search_requests": search.calls,
                    "fetch_requests": fetcher.calls,
                    "answer": answer,
                    "approval_mode": "simulated_all_presented_for_sandbox_evaluation",
                },
            )
        else:
            write_json(
                root / "trace.json",
                {
                    "status": "not_applicable",
                    "reason": "No explicit URL supplied; excluded from URL-pipeline denominator",
                },
            )
        agent = row.get("agent", {}) if row else {}

        def metadata(candidate):
            return pages.get(candidate["url"], {})

        return {
            "case_id": case["case_id"],
            "split": case["split"],
            "scenario": case["scenario"],
            "method": method,
            "expected_outcome": case["expected_outcome"],
            "status": status,
            "discovery_status": discovery_status,
            "outcome": outcome,
            "candidate_count": len(candidates),
            "qualified_count": sum(c["url"] in case["gold_urls"] for c in candidates),
            "source_correct_count": sum(
                metadata(c).get("official", False)
                and metadata(c).get("product") == case["product"]
                for c in candidates
            ),
            "version_correct_count": sum(
                metadata(c).get("actual_version") == case["version"] for c in candidates
            ),
            "imported_count": imported_count,
            "wrong_import_count": wrong_imports,
            "import_retrieval_success": retrieval_success,
            "import_failure": import_failure,
            "failure_reason_codes": sorted(set(failure_codes)),
            "observed_issue_codes": sorted(set(observed_issues)),
            "decision_steps": agent.get("model_calls", 0),
            "searches": len(search.calls),
            "reads": agent.get("reads", len(fetcher.calls)),
            "fetched_pages": len(fetcher.calls),
            "limits": limits.__dict__,
            "llm_api_calls": 0,
            "input_tokens": None,
            "output_tokens": None,
            "model_cost_usd": None,
            "model_usage_status": "not_applicable_no_llm",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "trace_path": (relative / "trace.json").as_posix(),
        }
    finally:
        await service.close()


async def run_suite(suite_root, output, splits, project_root):
    suite_root, output, project_root = (
        Path(suite_root),
        Path(output),
        Path(project_root),
    )
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Unique nonempty splits required")
    cases = [case for split in splits for case in load_suite(suite_root, split)]
    output.mkdir(parents=True, exist_ok=False)
    code_paths = sorted(
        set((project_root / "src/docguard/discovery").glob("*.py"))
        | set((project_root / "src/docguard/knowledge").glob("*.py"))
    )
    code_paths += [
        project_root / p
        for p in (
            "src/docguard/processing/block_packer.py",
            "src/docguard/processing/token_budget.py",
            "configs/chunking/s2-chunk-v1.json",
            "data/tokenizers/manifest.json",
            "pyproject.toml",
            "uv.lock",
        )
    ]
    manifest = {
        "run_id": "s15-replay-" + uuid4().hex,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "splits": splits,
        "methods": list(METHODS),
        "evidence_level": "scripted_engineering_replay_only",
        "dataset_manifest_sha256": digest(suite_root / "manifest.json"),
        "code_sha256": {
            p.relative_to(project_root).as_posix(): digest(p) for p in code_paths
        },
        "human_validation_status": "pending",
        "test_execution_status": "locked",
    }
    write_json(output / "run_manifest.json", manifest)
    budget = TokenBudget.from_manifest(
        project_root / "data/tokenizers/manifest.json", project_root
    )
    config = load_chunking_config(project_root / "configs/chunking/s2-chunk-v1.json")
    pages = json.loads((suite_root / "pages.json").read_text(encoding="utf-8"))
    protocol = json.loads((suite_root / "protocol.json").read_text(encoding="utf-8"))
    rows, reviews = [], []
    try:
        with (output / "results.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for case in cases:
                for method in METHODS:
                    row = await _run_case(
                        case,
                        method,
                        suite_root,
                        output,
                        budget,
                        config,
                        pages,
                        protocol,
                    )
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    if method != "given_url":
                        reviews.append(
                            {
                                "case_id": case["case_id"],
                                "method": method,
                                "goal": case["goal"],
                                "expected_outcome": case["expected_outcome"],
                                "gold_urls": case["gold_urls"],
                                "trace_path": row["trace_path"],
                                "result": row,
                                "review_status": "pending",
                                "reviewer": None,
                                "expectation_correct": None,
                                "citation_support_confirmed": None,
                                "failure_reason_confirmed": None,
                                "notes": None,
                            }
                        )
        packet = {
            "human_validation_status": "pending",
            "expectation_author": "assistant_authored",
            "results_sha256": digest(output / "results.jsonl"),
            "reviews": reviews,
        }
        report = {
            **manifest,
            "status": "completed",
            "row_count": len(rows),
            "metrics": summarize(rows),
            "paired_comparisons": paired_comparisons(rows),
            "human_review_packet": packet,
        }
        write_json(output / "human_review_packet.json", packet)
        write_json(
            output / "summary.json",
            {k: v for k, v in report.items() if k != "human_review_packet"},
        )
        manifest.update(
            status="completed",
            completed_at=datetime.now(UTC).isoformat(),
            row_count=len(rows),
            artifact_sha256={
                p.name: digest(p)
                for p in (
                    output / "results.jsonl",
                    output / "summary.json",
                    output / "human_review_packet.json",
                )
            },
            trace_sha256={
                r["trace_path"]: digest(output / r["trace_path"]) for r in rows
            },
        )
        write_json(output / "run_manifest.json", manifest)
        return report
    except BaseException:
        manifest.update(status="interrupted_or_failed", recorded_rows=len(rows))
        write_json(output / "run_manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--split",
        choices=["development", "validation", "test"],
        nargs="+",
        default=["development", "validation"],
    )
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_suite(args.output)
        print(
            json.dumps(
                {
                    "status": result["content_status"],
                    "human_expectations_status": "pending",
                    "test_execution_status": "locked",
                }
            )
        )
    else:
        result = asyncio.run(
            run_suite(args.suite, args.output, args.split, args.project_root)
        )
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "rows": result["row_count"],
                    "evidence_level": result["evidence_level"],
                }
            )
        )


if __name__ == "__main__":
    main()
