"""Render S12 publication figures from frozen report artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
plt.rcParams["svg.hashsalt"] = "docguard-s12-final-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _footer(fig: Any, sources: dict[str, str]) -> None:
    text = "Sources SHA-256:\n" + "\n".join(
        f"{Path(path).name}={digest}" for path, digest in sorted(sources.items())
    )
    fig.text(0.01, 0.006, text, fontsize=5.5, color="#555555")


def _save(fig: Any, output: Path, sources: dict[str, str]) -> dict[str, Any]:
    _footer(fig, sources)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(
        png,
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "DocGuard-RAG"},
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "DocGuard-RAG", "Date": None},
    )
    # Matplotlib uses platform newlines and emits trailing path whitespace.
    # Normalize both so the committed artifact and hash are cross-platform.
    svg.write_bytes(
        b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n"
    )
    plt.close(fig)
    return {
        "png": {"path": png.name, "sha256": _sha256(png), "dpi": 450},
        "svg": {"path": svg.name, "sha256": _sha256(svg)},
        "sources": sources,
    }


def render(repo_root: Path) -> dict[str, Any]:
    report_root = repo_root / "reports" / "s12_final"
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "summary.json"
    summary = _load(summary_path)
    source = {str(summary_path.relative_to(repo_root)): _sha256(summary_path)}
    methods = summary["methods"]
    frozen = summary["frozen_run_summaries"]
    manifest: dict[str, Any] = {"schema_version": 1, "figures": {}}

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for index, method in enumerate(("b0", "b1", "m1", "m2")):
        quality = methods[method]["retrieval"]["ndcg_at_10"]
        latency = methods[method]["latency_ms"]["end_to_end"]["p50"]
        ax.scatter(latency, quality, s=75, color=COLORS[index], label=method.upper())
        x_offset = -5 if method == "m2" else 5
        alignment = "right" if method == "m2" else "left"
        ax.annotate(
            method.upper(),
            (latency, quality),
            xytext=(x_offset, 5),
            textcoords="offset points",
            ha=alignment,
        )
    ax.margins(x=0.06, y=0.08)
    ax.set(xlabel="End-to-end latency p50 (ms)", ylabel="nDCG@10", title="Retrieval quality–latency trade-off")
    ax.grid(alpha=0.25)
    manifest["figures"]["retrieval_quality_latency"] = _save(
        fig, figure_root / "retrieval_quality_latency", source
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for index, method in enumerate(("m3", "m4")):
        curve = frozen[method]["refusal"]["risk_coverage"]
        ax.plot(
            [point["coverage"] for point in curve],
            [point["selective_risk"] for point in curve],
            color=COLORS[index],
            linewidth=2,
            label=method.upper(),
        )
    ax.set(xlabel="Coverage", ylabel="Selective risk", title="Risk–coverage on frozen Test")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    manifest["figures"]["risk_coverage"] = _save(fig, figure_root / "risk_coverage", source)

    online = ("b2", "m3", "m4", "m4_no_version_filter", "m4_no_citation_validation")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    failures = [methods[method]["failed_count"] for method in online]
    bars = ax.bar(range(len(online)), failures, color=COLORS, width=0.68)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xticks(range(len(online)), [name.upper().replace("_", "\n") for name in online], fontsize=8)
    ax.set(ylabel="Failed questions (fixed n=60)", title="Generation failure distribution")
    ax.grid(axis="y", alpha=0.25)
    manifest["figures"]["failure_distribution"] = _save(
        fig, figure_root / "failure_distribution", source
    )

    refusal_methods = ("m3", "m4", "m4_no_version_filter", "m4_no_citation_validation")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = [name.upper().replace("_", "\n") for name in refusal_methods]
    x = list(range(len(labels)))
    width = 0.24
    for offset, metric, label, color in (
        (-width, "refusal_precision", "Precision", COLORS[0]),
        (0.0, "refusal_recall", "Recall", COLORS[1]),
        (width, "refusal_f1", "F1", COLORS[2]),
    ):
        ax.bar(
            [value + offset for value in x],
            [methods[method]["refusal"][metric] for method in refusal_methods],
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set(ylabel="Score", title="Refusal performance and registered ablations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    manifest["figures"]["method_comparison"] = _save(
        fig, figure_root / "method_comparison", source
    )

    manifest_path = figure_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = render(args.repo_root.resolve())
    print(f"S12_FIGURES_PASS figures={len(manifest['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
