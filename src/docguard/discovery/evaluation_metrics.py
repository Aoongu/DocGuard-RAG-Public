"""Fixed-denominator summaries for engineering replay, not model quality."""

import math
from collections import Counter, defaultdict

from docguard.evaluation.compare import _percentile, paired_bootstrap


def proportion(count, total):
    if not total:
        return {
            "numerator": count,
            "denominator": 0,
            "rate": None,
            "wilson_95_ci": None,
        }
    p, z = count / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "numerator": count,
        "denominator": total,
        "rate": p,
        "wilson_95_ci": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def discovery_success(row):
    return (
        row["candidate_count"] > 0 and row["qualified_count"] == row["candidate_count"]
    )


def summarize(rows):
    identities = [(r["split"], r["method"], r["case_id"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate evaluation identity")
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['split']}/{row['method']}"].append(row)
    report = {}
    for name, all_rows in groups.items():
        group = [r for r in all_rows if r["status"] != "not_applicable"]
        positive = [r for r in group if r["expected_outcome"] == "discover"]
        presented = sum(r["candidate_count"] for r in group)
        imported = sum(r["imported_count"] for r in group)
        expected_clarify = [r for r in group if r["expected_outcome"] == "clarify"]
        observed_clarify = [r for r in group if r["outcome"] == "clarify"]
        correct_clarify = sum(r["outcome"] == "clarify" for r in expected_clarify)
        failure_counts = Counter(
            code for r in group for code in set(r["failure_reason_codes"])
        )
        report[name] = {
            "fixed_sample_count": len(all_rows),
            "not_applicable_count": len(all_rows) - len(group),
            "task_count": len(group),
            "discovery_success": proportion(
                sum(discovery_success(r) for r in positive), len(positive)
            ),
            "import_retrieval_success": proportion(
                sum(r["import_retrieval_success"] for r in positive), len(positive)
            ),
            "source_accuracy": proportion(
                sum(r["source_correct_count"] for r in group), presented
            ),
            "version_accuracy": proportion(
                sum(r["version_correct_count"] for r in group), presented
            ),
            "wrong_import_rate": proportion(
                sum(r["wrong_import_count"] for r in group), imported
            ),
            "wrong_import_task_rate": proportion(
                sum(r["wrong_import_count"] > 0 for r in group), len(group)
            ),
            "clarification_appropriateness": proportion(
                sum(
                    (r["expected_outcome"] == "clarify") == (r["outcome"] == "clarify")
                    for r in group
                ),
                len(group),
            ),
            "clarification_recall": proportion(correct_clarify, len(expected_clarify)),
            "clarification_precision": proportion(
                correct_clarify, len(observed_clarify)
            ),
            "failed_or_exhausted_task_rate": proportion(
                sum(r["status"] in ("failed", "exhausted") for r in group), len(group)
            ),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "failure_reason_counting": "multi_label_tasks_may_repeat",
            "observed_issue_counts": dict(
                sorted(
                    Counter(
                        code
                        for r in group
                        for code in set(r.get("observed_issue_codes", []))
                    ).items()
                )
            ),
            "tool_totals": {
                field: sum(r[field] for r in group)
                for field in (
                    "decision_steps",
                    "searches",
                    "reads",
                    "fetched_pages",
                    "llm_api_calls",
                )
            },
            "latency_ms": {
                "p50": _percentile([r["latency_ms"] for r in group], 0.5)
                if group
                else None,
                "p95": _percentile([r["latency_ms"] for r in group], 0.95)
                if group
                else None,
                "scope": "local_replay_only",
            },
            "unknown_usage_rows": sum(
                r.get("model_usage_status") != "not_applicable_no_llm"
                and (
                    r["model_cost_usd"] is None
                    or r["input_tokens"] is None
                    or r["output_tokens"] is None
                )
                for r in group
            ),
            "model_usage_not_applicable_rows": sum(
                r.get("model_usage_status") == "not_applicable_no_llm" for r in group
            ),
            "model_cost_usd": None,
            "offline_external_api_spend_usd": 0.0,
            "generation_quality_evaluation": "not_performed",
            "judge_status": "not_performed",
        }
    return report


def paired_comparisons(rows):
    result = {}
    for split in sorted({r["split"] for r in rows}):
        methods = {}
        for method in ("single_search_replay", "adaptive_replay"):
            methods[method] = {
                r["case_id"]: float(discovery_success(r))
                for r in rows
                if r["split"] == split
                and r["method"] == method
                and r["expected_outcome"] == "discover"
            }
        result[split] = paired_bootstrap(
            methods["single_search_replay"],
            methods["adaptive_replay"],
            seed=20261027,
            samples=10000,
            confidence_level=0.95,
        )
        result[split]["interpretation"] = (
            "constructed_scripted_fixture_difference_not_llm_superiority"
        )
    return result
