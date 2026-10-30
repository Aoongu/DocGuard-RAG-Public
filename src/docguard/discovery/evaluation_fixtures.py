"""Assistant-authored synthetic fixtures; no official content or human labels."""

import hashlib
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from docguard.discovery.graph import AgentLimits

SCENARIOS = (
    "exact",
    "empty_then_rewrite",
    "wrong_version",
    "ambiguous",
    "duplicate",
    "shell",
    "unverified_source",
    "upstream_failure",
    "budget_exhausted",
    "similar_product",
)
FAMILIES = {
    "development": ("Python", "docs.python.org", "3", "2", "main"),
    "validation": ("Redis", "redis.io", "7", "6", "article"),
    "test": ("PostgreSQL", "www.postgresql.org", "18", "17", "role-main"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _body(product, version, template, marker, *, shell=False):
    title = f"{product} {version} configuration (synthetic fixture)"
    content = (
        "<nav>Loading documentation</nav>"
        if shell
        else f'<h1 id="configuration">{escape(title)}</h1><p>Synthetic engineering example: {marker} is available in this configuration section.</p><pre>setting = {marker}</pre>'
    )
    container = (
        f'<div role="main">{content}</div>'
        if template == "role-main"
        else f"<{template}>{content}</{template}>"
    )
    return f"<!doctype html><html><head><title>{title}</title></head><body>{container}</body></html>"


def prepare_suite(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    files, split_groups = {}, {}
    protocol = {
        "schema_version": 1,
        "suite_version": "s15-engineering-v1",
        "methods": ["given_url", "single_search_replay", "adaptive_replay"],
        "limits": asdict(AgentLimits()),
        "bootstrap_seed": 20261027,
        "bootstrap_samples": 10000,
        "evidence_level": "scripted_engineering_replay_only",
        "human_expectations_status": "pending",
        "test_execution_status": "locked",
    }
    write_json(root / "protocol.json", protocol)
    files["protocol.json"] = digest(root / "protocol.json")
    pages = {}
    for split, (product, host, version, old, template) in FAMILIES.items():
        cases = []
        split_groups[split] = {
            "product": product,
            "template_family": template,
            "count": len(SCENARIOS),
        }
        for i, scenario in enumerate(SCENARIOS, 1):
            case_id = f"s15-{split}-{i:03d}"
            marker = f"fixture{split}{i:03d}"
            good = f"https://{host}/{version}/s15-fixture/{scenario}.html"
            wrong = good.replace(f"/{version}/", f"/{old}/")
            shell = good.replace(".html", "-shell.html")
            community = f"https://community.example.org/{product.lower()}/{version}/{scenario}.html"
            other = f"https://other-product.example.org/{version}/{scenario}.html"
            for url, actual_version, container, is_shell, official in [
                (good, version, template, False, True),
                (wrong, old, template, False, True),
                (shell, version, template, True, True),
                (community, version, template, False, False),
                (other, version, template, False, False),
            ]:
                name = "pages/" + hashlib.sha256(url.encode()).hexdigest() + ".html"
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text(
                    _body(
                        product if official else "OtherProduct",
                        actual_version,
                        container,
                        marker,
                        shell=is_shell,
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                files[name] = digest(path)
                pages[url] = {
                    "path": name,
                    "actual_version": actual_version,
                    "official": official,
                    "product": product if official else "OtherProduct",
                    "sha256": files[name],
                }
            outcome = "discover"
            initial, refined = [good], [good]
            if scenario == "empty_then_rewrite":
                initial = []
            elif scenario == "wrong_version":
                initial = [wrong]
            elif scenario == "ambiguous":
                outcome = "clarify"
            elif scenario == "duplicate":
                initial = [good, good, good]
            elif scenario == "shell":
                initial = [shell]
            elif scenario == "unverified_source":
                outcome, initial, refined = "abstain", [community], [community]
            elif scenario == "upstream_failure":
                outcome, initial, refined = "no_result", None, None
            elif scenario == "budget_exhausted":
                outcome, initial, refined = "no_result", [], []
            elif scenario == "similar_product":
                initial = [other]
            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "scenario": scenario,
                    "product": product,
                    "version": None if scenario == "ambiguous" else version,
                    "template_family": template,
                    "goal": f"Find official {product} configuration documentation",
                    "initial_search": initial,
                    "refined_search": refined,
                    "expected_outcome": outcome,
                    "gold_urls": [good] if outcome == "discover" else [],
                    "direct_urls": [good] if outcome == "discover" else [],
                    "question": f"Where is {marker}?",
                    "retrieval_marker": marker,
                    "max_model_calls": 2 if scenario == "budget_exhausted" else 8,
                    "expectation_author": "assistant_authored",
                    "human_expectation_confirmed": None,
                }
            )
        path = root / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases
            ),
            encoding="utf-8",
            newline="\n",
        )
        files[path.name] = digest(path)
    write_json(root / "pages.json", pages)
    files["pages.json"] = digest(root / "pages.json")
    manifest = {
        "schema_version": 1,
        "kind": "synthetic_engineering_fixtures",
        "content_status": "fixture_frozen",
        "human_expectations_status": "pending",
        "test_execution_status": "locked",
        "split_groups": split_groups,
        "files": files,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def load_suite(root, split):
    root = Path(root)
    if split == "test":
        raise ValueError(
            "Synthetic Test is locked pending a separate reviewed protocol"
        )
    if split not in ("development", "validation"):
        raise ValueError("Unsupported split")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Frozen fixture hash mismatch")
    groups = list(manifest["split_groups"].values())
    if (
        len({r["product"] for r in groups}) != 3
        or len({r["template_family"] for r in groups}) != 3
    ):
        raise ValueError("Split product/template overlap")
    cases = [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        len(cases) != 10
        or len({r["case_id"] for r in cases}) != 10
        or any(r["split"] != split for r in cases)
    ):
        raise ValueError("Invalid fixed split identities")
    return cases
