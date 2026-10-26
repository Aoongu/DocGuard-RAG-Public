"""Build the deterministic S4 golden evaluation dataset from frozen S2 chunks.

Merges human-authored questions into the assistant-curated candidate set,
resolves official source references to chunk IDs, validates the two human
review rounds, and only marks the dataset frozen when every gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from docguard.evaluation.dataset import build_source_index, resolve_question_sources
from docguard.evaluation.models import EvaluationQuestion

SPLIT_CATEGORY_COUNTS = {
    "development": {
        "single_fact": 9,
        "how_to_parameter": 8,
        "cross_section": 7,
        "version_difference": 6,
        "out_of_domain": 10,
        "no_evidence_false_premise": 7,
        "unavailable_version": 7,
        "version_conflict_ambiguity": 6,
    },
    "validation": {
        "single_fact": 8,
        "how_to_parameter": 9,
        "cross_section": 6,
        "version_difference": 7,
        "out_of_domain": 10,
        "no_evidence_false_premise": 7,
        "unavailable_version": 6,
        "version_conflict_ambiguity": 7,
    },
    "test": {
        "single_fact": 8,
        "how_to_parameter": 8,
        "cross_section": 7,
        "version_difference": 7,
        "out_of_domain": 10,
        "no_evidence_false_premise": 6,
        "unavailable_version": 7,
        "version_conflict_ambiguity": 7,
    },
}
ANSWERABLE_CATEGORIES = (
    "single_fact",
    "how_to_parameter",
    "cross_section",
    "version_difference",
)
BOUNDARY_CATEGORIES = (
    "out_of_domain",
    "no_evidence_false_premise",
    "unavailable_version",
    "version_conflict_ambiguity",
)
CATEGORY_REASON_CODES = {
    "out_of_domain": "OUT_OF_DOMAIN",
    "no_evidence_false_premise": "NO_EVIDENCE",
    "unavailable_version": "UNSUPPORTED_VERSION",
    "version_conflict_ambiguity": "VERSION_REQUIRED",
}
HUMAN_REQUIRED_TOTAL = 36
HUMAN_PER_SPLIT = 12
HUMAN_ANSWERABLE_PER_SPLIT = 6
HUMAN_CATEGORY_TOTALS = {
    "single_fact": 5,
    "how_to_parameter": 5,
    "cross_section": 4,
    "version_difference": 4,
    "out_of_domain": 6,
    "no_evidence_false_premise": 4,
    "unavailable_version": 4,
    "version_conflict_ambiguity": 4,
}
HUMAN_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "author_index",
        "question",
        "requested_version",
        "answerable",
        "answer_key_points",
        "relevant_source_refs",
        "expected_reason_code",
        "category",
        "split",
    }
)
REVIEW_FIELDS = frozenset(
    {"question_id", "review_round", "decision", "reviewer", "reviewed_at", "method", "notes"}
)
REVIEW_DECISIONS = frozenset({"pass", "modify", "delete"})
OUT_OF_DOMAIN_TOPICS = (
    "Python virtual environments",
    "Kubernetes ingress rules",
    "React component state",
    "Java garbage collection",
    "Linux desktop themes",
    "Docker image signing",
    "Git rebase workflows",
    "TensorFlow model training",
    "AWS billing alerts",
    "Nginx TLS configuration",
    "Redis cluster sharding",
    "MongoDB aggregation",
    "MySQL replication",
    "SQLite mobile storage",
    "Oracle tablespaces",
    "C++ template metaprogramming",
    "Rust ownership",
    "Go goroutine scheduling",
    "CSS grid layout",
    "Android application signing",
    "iOS push notifications",
    "Terraform state backends",
    "Ansible inventories",
    "Prometheus alert rules",
    "Grafana dashboards",
    "Kafka consumer groups",
    "RabbitMQ exchanges",
    "Elasticsearch index mappings",
    "PyTorch CUDA installation",
    "Node.js package publishing",
)
FALSE_PREMISES = (
    "PostgreSQL stores every table exclusively in a blockchain",
    "VACUUM automatically converts SQL into Java bytecode",
    "SELECT always deletes its source rows",
    "PostgreSQL requires a GPU for every query",
    "CREATE TABLE can only create temporary tables",
    "pg_dump modifies all source rows during backup",
    "PostgreSQL has no transaction support",
    "Every index guarantees constant-time lookup",
    "Roles can never own database objects",
    "EXPLAIN executes no planning logic",
    "PostgreSQL accepts only JSON documents",
    "All constraints are checked by the operating system",
    "The server can only listen on Bluetooth",
    "A database cluster contains exactly one database",
    "WAL is a web application language",
    "MVCC permits only one concurrent reader",
    "Every query must use a recursive CTE",
    "PostgreSQL passwords are fixed in source code",
    "ANALYZE removes every table statistic",
    "TRUNCATE is identical to creating an operating-system user",
)
UNAVAILABLE_VERSIONS = (
    "9.0", "9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "10.23", "11.22", "12.17",
    "13.13", "14.10", "15.5", "16.1", "16.4", "18.1", "19.0", "19.1", "20.0", "latest-development",
)
AMBIGUOUS_TOPICS = (
    "the default authentication behavior",
    "the available VACUUM options",
    "the CREATE TABLE syntax",
    "the SELECT syntax",
    "the backup procedure",
    "the planner configuration",
    "role membership behavior",
    "the default server setting",
    "the index creation syntax",
    "the ALTER TABLE behavior",
    "the COPY command options",
    "the transaction isolation description",
    "the generated column rules",
    "the partitioning limitations",
    "the EXPLAIN output format",
    "the JSON function inventory",
    "the replication configuration",
    "the connection option defaults",
    "the privilege rules",
    "the SQL compatibility statement",
)


class FreezeError(ValueError):
    """Raised when the S4 golden dataset contract is violated."""


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise FreezeError("S2 chunks must not be empty")
    return rows


def _group_chunks(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["postgresql_version"], row["canonical_url"], row["anchor"])
        groups[key].append(row)
    return {
        key: sorted(values, key=lambda value: (value["ordinal"], value["chunk_id"]))
        for key, values in sorted(groups.items())
    }


def _label(group: list[dict[str, Any]]) -> str:
    section = group[0].get("section_path") or []
    if section:
        return str(section[-1]).strip()
    return Path(group[0]["canonical_url"]).stem


def _key_point(group: list[dict[str, Any]]) -> str:
    lines = [line.strip() for line in group[0]["text"].splitlines() if line.strip()]
    candidate = lines[1] if len(lines) > 1 else lines[0]
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate[:500]


def _ref(key: tuple[str, str, str]) -> dict[str, str]:
    return {"version": key[0], "normalized_url": key[1], "anchor": key[2]}


def _available_keys(groups: dict[tuple[str, str, str], list[dict[str, Any]]]):
    return [
        key
        for key, values in groups.items()
        if len(values[0]["text"]) >= 80 and not key[2].startswith("R1-")
    ]


def _take_single(
    keys: list[tuple[str, str, str]],
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    used: set[tuple[str, str, str]],
    *,
    command_only: bool,
) -> tuple[str, str, str]:
    for key in keys:
        filename = key[1].rsplit("/", 1)[-1]
        is_command = filename.startswith(("sql-", "app-"))
        if key not in used and (not command_only or is_command):
            used.add(key)
            return key
    raise FreezeError("not enough source groups for candidate dataset")


def _answerable_payload(
    category: str,
    keys: list[tuple[str, str, str]],
    version_pairs: list[tuple[tuple[str, str, str], tuple[str, str, str]]],
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    used: set[tuple[str, str, str]],
) -> tuple[str, str | None, list[str], list[dict[str, str]]]:
    if category in {"single_fact", "how_to_parameter"}:
        key = _take_single(keys, groups, used, command_only=category == "how_to_parameter")
        label = _label(groups[key])
        if category == "single_fact":
            document = Path(key[1]).stem
            question = (
                f"According to PostgreSQL {key[0]}, what does the documentation state about "
                f"“{label}” in the “{document}” documentation?"
            )
        else:
            command = Path(key[1]).stem.removeprefix("sql-").removeprefix("app-").upper()
            question = f"In PostgreSQL {key[0]}, what is {command} used for in the “{label}” section?"
        return question, key[0], [_key_point(groups[key])], [_ref(key)]
    if category == "cross_section":
        first = _take_single(keys, groups, used, command_only=False)
        second = next(
            key
            for key in keys
            if key not in used and key[0] == first[0] and key[1] != first[1]
        )
        used.add(second)
        question = (
            f"For PostgreSQL {first[0]}, summarize both “{_label(groups[first])}” "
            f"and “{_label(groups[second])}” as documented."
        )
        return (
            question,
            first[0],
            [_key_point(groups[first]), _key_point(groups[second])],
            [_ref(first), _ref(second)],
        )
    while version_pairs:
        first, second = version_pairs.pop(0)
        if first not in used and second not in used:
            used.update((first, second))
            question = (
                f"Compare PostgreSQL 17.11 and 18.6 documentation for “{_label(groups[first])}”; "
                "state what each frozen version says."
            )
            return (
                question,
                None,
                [f"17.11: {_key_point(groups[first])}", f"18.6: {_key_point(groups[second])}"],
                [_ref(first), _ref(second)],
            )
    raise FreezeError("not enough cross-version source pairs")


def _boundary_payload(category: str, index: int) -> tuple[str, str]:
    if category == "out_of_domain":
        topic = OUT_OF_DOMAIN_TOPICS[index % len(OUT_OF_DOMAIN_TOPICS)]
        return f"How should I configure {topic}?", "OUT_OF_DOMAIN"
    if category == "no_evidence_false_premise":
        claim = FALSE_PREMISES[index % len(FALSE_PREMISES)]
        return f"The premise says that {claim}. Which frozen PostgreSQL section proves it?", "NO_EVIDENCE"
    if category == "unavailable_version":
        version = UNAVAILABLE_VERSIONS[index % len(UNAVAILABLE_VERSIONS)]
        return f"According to PostgreSQL {version}, what is the exact documented behavior of SELECT?", "UNSUPPORTED_VERSION"
    topic = AMBIGUOUS_TOPICS[index % len(AMBIGUOUS_TOPICS)]
    return f"What is {topic}? Use the correct version-specific answer, but no version is specified.", "VERSION_REQUIRED"


def _canonical_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(_canonical_line(row) for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _load_human_authored(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    seen_questions: dict[str, str] = {}
    split_keys: defaultdict[tuple[str, str, str], str] = defaultdict(str)
    category_totals: Counter[str] = Counter()
    split_totals: Counter[str] = Counter()
    split_answerable: Counter[str] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        unknown = set(row) - HUMAN_INPUT_FIELDS
        missing = HUMAN_INPUT_FIELDS - set(row)
        if unknown or missing:
            raise FreezeError(
                f"human question at line {line_number} has unknown={sorted(unknown)} missing={sorted(missing)}"
            )
        if row["schema_version"] != 1:
            raise FreezeError(f"human question at line {line_number} must use schema_version 1")
        if type(row["author_index"]) is not int or row["author_index"] in seen_indexes:
            raise FreezeError(f"duplicate or invalid author_index at line {line_number}")
        seen_indexes.add(row["author_index"])
        if row["split"] not in SPLIT_CATEGORY_COUNTS:
            raise FreezeError(f"invalid split at line {line_number}: {row['split']}")
        if row["category"] not in CATEGORY_REASON_CODES and row["category"] not in ANSWERABLE_CATEGORIES:
            raise FreezeError(f"invalid category at line {line_number}: {row['category']}")
        if type(row["answerable"]) is not bool:
            raise FreezeError(f"answerable must be a boolean at line {line_number}")
        if row["answerable"]:
            if not row["answer_key_points"] or not row["relevant_source_refs"]:
                raise FreezeError(f"answerable human question at line {line_number} needs key points and refs")
            versions = {ref["version"] for ref in row["relevant_source_refs"]}
            if row["category"] == "version_difference":
                if row["requested_version"] is not None or versions != {"17.11", "18.6"}:
                    raise FreezeError(f"version_difference human question at line {line_number} needs both versions")
            else:
                if row["requested_version"] not in {"17.11", "18.6"} or versions != {row["requested_version"]}:
                    raise FreezeError(f"human question at line {line_number} needs a single consistent version")
            if row["category"] == "how_to_parameter" and not all(
                ref["normalized_url"].rsplit("/", 1)[-1].startswith(("sql-", "app-"))
                for ref in row["relevant_source_refs"]
            ):
                raise FreezeError(f"how_to human question at line {line_number} must cite command pages")
        else:
            if row["expected_reason_code"] != CATEGORY_REASON_CODES[row["category"]]:
                raise FreezeError(f"reason code mismatch for category at line {line_number}")
            if row["answer_key_points"] or row["relevant_source_refs"]:
                raise FreezeError(f"boundary human question at line {line_number} must not carry evidence")
        normalized = " ".join(row["question"].lower().split())
        if normalized in seen_questions:
            raise FreezeError(
                f"human question at line {line_number} duplicates {seen_questions[normalized]}"
            )
        seen_questions[normalized] = str(row["author_index"])
        for ref in row["relevant_source_refs"]:
            key = (ref["version"], ref["normalized_url"], ref["anchor"])
            if key in split_keys and split_keys[key] != row["split"]:
                raise FreezeError(
                    f"human question at line {line_number} reuses source {key} across splits "
                    f"({split_keys[key]} and {row['split']})"
                )
            split_keys[key] = row["split"]
        category_totals[row["category"]] += 1
        split_totals[row["split"]] += 1
        split_answerable[row["split"]] += 1 if row["answerable"] else 0
        rows.append(row)
    if not rows:
        raise FreezeError("human authored input must not be empty")
    if dict(category_totals) != HUMAN_CATEGORY_TOTALS:
        raise FreezeError(f"human category distribution mismatch: {dict(category_totals)}")
    for split, total in split_totals.items():
        if total != HUMAN_PER_SPLIT:
            raise FreezeError(f"human split {split} has {total} questions, expected {HUMAN_PER_SPLIT}")
        if split_answerable[split] != HUMAN_ANSWERABLE_PER_SPLIT:
            raise FreezeError(f"human split {split} has {split_answerable[split]} answerable questions")
    return rows


def _load_reviews(path: Path, expected_round: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        unknown = set(row) - REVIEW_FIELDS
        missing = REVIEW_FIELDS - set(row)
        if unknown or missing:
            raise FreezeError(
                f"review record at line {line_number} has unknown={sorted(unknown)} missing={sorted(missing)}"
            )
        if row["review_round"] != expected_round:
            raise FreezeError(f"review record at line {line_number} has the wrong round")
        if row["decision"] not in REVIEW_DECISIONS:
            raise FreezeError(f"invalid review decision at line {line_number}: {row['decision']}")
        if expected_round == 2 and not row["method"]:
            raise FreezeError(f"round-2 review record at line {line_number} must declare its method")
        rows.append(row)
    if not rows:
        raise FreezeError(f"review file {path.name} must not be empty")
    return rows


def _validate_review_coverage(
    reviews: list[dict[str, Any]],
    question_ids: set[str],
    label: str,
) -> bool:
    reviewed_ids = [row["question_id"] for row in reviews]
    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise FreezeError(f"{label} contains duplicate question_id records")
    if set(reviewed_ids) != question_ids:
        raise FreezeError(
            f"{label} must cover exactly {len(question_ids)} questions; "
            f"covered={len(set(reviewed_ids))}"
        )
    return all(row["decision"] == "pass" for row in reviews)


def _validate_round2_distinct(
    round1: list[dict[str, Any]],
    round2: list[dict[str, Any]],
) -> None:
    first = {row["question_id"]: row for row in round1}
    for row in round2:
        earlier = first.get(row["question_id"])
        if earlier is None:
            continue
        same_identity = (
            earlier["reviewer"] == row["reviewer"]
            and earlier["reviewed_at"] == row["reviewed_at"]
        )
        identical_content = (
            earlier["decision"] == row["decision"]
            and earlier["notes"] == row["notes"]
            and earlier["method"] == row["method"]
        )
        if same_identity and identical_content:
            raise FreezeError(
                f"round-2 record for {row['question_id']} reuses the round-1 record verbatim"
            )


def _human_final_row(
    human: dict[str, Any],
    slot: dict[str, object],
    source_index: dict[tuple[str, str, str], tuple[str, ...]],
) -> dict[str, object]:
    seed = {
        "schema_version": 1,
        "question_id": slot["question_id"],
        "question": human["question"],
        "requested_version": human["requested_version"],
        "answerable": human["answerable"],
        "answer_key_points": list(human["answer_key_points"]),
        "relevant_source_refs": list(human["relevant_source_refs"]),
        "resolved_chunk_ids": ["chk_" + "0" * 64] if human["answerable"] else [],
        "chunking_version": "s2-chunk-v1",
        "expected_reason_code": human["expected_reason_code"],
        "category": human["category"],
        "split": human["split"],
        "source": "human_written",
        "review_status": "source_verified",
        "paraphrase_group_id": f"{human['split']}-human-{human['author_index']:03d}",
    }
    question = EvaluationQuestion.from_dict(seed)
    if question.answerable:
        seed["resolved_chunk_ids"] = list(resolve_question_sources(question, source_index))
    return EvaluationQuestion.from_dict(seed).to_dict()


def _merge_human_questions(
    split_rows: dict[str, list[dict[str, object]]],
    human_rows: list[dict[str, Any]],
    source_index: dict[tuple[str, str, str], tuple[str, ...]],
) -> None:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for human in human_rows:
        grouped[(human["split"], human["category"])].append(human)
    for (split, category), humans in grouped.items():
        humans.sort(key=lambda item: item["author_index"])
        produced = split_rows[split]
        slots = [row for row in produced if row["category"] == category]
        if len(humans) > len(slots):
            raise FreezeError(
                f"{split}/{category} cannot host {len(humans)} human questions in {len(slots)} slots"
            )
        replaced = slots[len(slots) - len(humans) :]
        for human, slot in zip(humans, replaced, strict=True):
            produced[produced.index(slot)] = _human_final_row(human, slot, source_index)
        split_rows[split] = produced


def freeze(
    chunks_path: Path,
    output_root: Path,
    smoke_path: Path,
    human_path: Path,
    round1_path: Path,
    round2_path: Path,
) -> dict[str, object]:
    rows = _load_chunks(chunks_path)
    groups = _group_chunks(rows)
    keys = _available_keys(groups)
    source_index = build_source_index(chunks_path)
    by_identity = {(key[1].split(f"/docs/{key[0].split('.')[0]}/", 1)[1], key[2], key[0]): key for key in keys}
    version_pairs = []
    for (relative, anchor, version), first in sorted(by_identity.items()):
        if version != "17.11":
            continue
        second = by_identity.get((relative, anchor, "18.6"))
        if second is not None:
            version_pairs.append((first, second))

    human_rows = _load_human_authored(human_path)

    used: set[tuple[str, str, str]] = set()
    split_rows: dict[str, list[dict[str, object]]] = {}
    boundary_indexes: Counter[str] = Counter()
    for split in ("development", "validation", "test"):
        produced: list[dict[str, object]] = []
        categories = [
            category
            for category in (*ANSWERABLE_CATEGORIES, *BOUNDARY_CATEGORIES)
            for _ in range(SPLIT_CATEGORY_COUNTS[split][category])
        ]
        prefix = {"development": "dev", "validation": "val", "test": "test"}[split]
        for number, category in enumerate(categories, 1):
            answerable = category in ANSWERABLE_CATEGORIES
            if answerable:
                question, requested_version, key_points, refs = _answerable_payload(
                    category, keys, version_pairs, groups, used
                )
                seed: dict[str, object] = {
                    "schema_version": 1,
                    "question_id": f"s4-{prefix}-{number:03d}",
                    "question": question,
                    "requested_version": requested_version,
                    "answerable": True,
                    "answer_key_points": key_points,
                    "relevant_source_refs": refs,
                    "resolved_chunk_ids": ["chk_" + "0" * 64],
                    "chunking_version": "s2-chunk-v1",
                    "expected_reason_code": None,
                    "category": category,
                    "split": split,
                    "source": "assistant_curated",
                    "review_status": "source_verified",
                    "paraphrase_group_id": f"{split}-{category}-{number:03d}",
                }
                temporary = EvaluationQuestion.from_dict(seed)  # type: ignore[arg-type]
                seed["resolved_chunk_ids"] = list(resolve_question_sources(temporary, source_index))
            else:
                question, reason = _boundary_payload(category, boundary_indexes[category])
                boundary_indexes[category] += 1
                seed = {
                    "schema_version": 1,
                    "question_id": f"s4-{prefix}-{number:03d}",
                    "question": question,
                    "requested_version": None,
                    "answerable": False,
                    "answer_key_points": [],
                    "relevant_source_refs": [],
                    "resolved_chunk_ids": [],
                    "chunking_version": "s2-chunk-v1",
                    "expected_reason_code": reason,
                    "category": category,
                    "split": split,
                    "source": "assistant_curated",
                    "review_status": "source_verified",
                    "paraphrase_group_id": f"{split}-{category}-{number:03d}",
                }
            produced.append(EvaluationQuestion.from_dict(seed).to_dict())  # type: ignore[arg-type]
        split_rows[split] = produced

    _merge_human_questions(split_rows, human_rows, source_index)

    all_rows = [row for split in ("development", "validation", "test") for row in split_rows[split]]
    all_ids = {row["question_id"] for row in all_rows}  # type: ignore[index]
    test_ids = {row["question_id"] for row in split_rows["test"]}  # type: ignore[index]
    normalized_questions: defaultdict[str, tuple[str, str]] = defaultdict(tuple)
    for row in all_rows:
        normalized = " ".join(str(row["question"]).lower().split())  # type: ignore[union-attr]
        earlier = normalized_questions[normalized]
        if earlier and earlier[1] != str(row["split"]):  # type: ignore[union-attr]
            raise FreezeError(
                f"duplicate question text across splits: {row['question_id']} and {earlier[0]}"
            )
        normalized_questions[normalized] = (str(row["question_id"]), str(row["split"]))  # type: ignore[index]

    human_review_complete = False
    test_double_review_complete = False
    round1_count = 0
    round2_count = 0
    if round1_path.exists():
        round1 = _load_reviews(round1_path, expected_round=1)
        round1_count = len(round1)
        if not _validate_review_coverage(round1, all_ids, "reviews_round1.jsonl"):
            raise FreezeError("reviews_round1.jsonl must record a final pass decision for every question")
        human_review_complete = True
        for row in all_rows:
            row["review_status"] = "human_reviewed"
        if round2_path.exists():
            round2 = _load_reviews(round2_path, expected_round=2)
            round2_count = len(round2)
            if not _validate_review_coverage(round2, test_ids, "reviews_test_round2.jsonl"):
                raise FreezeError("reviews_test_round2.jsonl must record a final pass decision for every Test question")
            _validate_round2_distinct(round1, round2)
            test_double_review_complete = True
            for row in split_rows["test"]:
                row["review_status"] = "double_reviewed"
    elif round2_path.exists():
        raise FreezeError("reviews_test_round2.jsonl cannot substitute for a missing reviews_round1.jsonl")

    human_written_count = sum(1 for row in all_rows if row["source"] == "human_written")
    if human_written_count < HUMAN_REQUIRED_TOTAL:
        raise FreezeError(
            f"dataset contains {human_written_count} human questions, expected {HUMAN_REQUIRED_TOTAL}"
        )

    frozen = human_review_complete and test_double_review_complete

    files = {}
    for split in ("development", "validation", "test"):
        produced = split_rows[split]
        filename = f"{split}.jsonl"
        files[filename] = {"sha256": _write(output_root / filename, produced), "count": len(produced)}
    smoke_rows = split_rows["development"][:15] + split_rows["development"][30:45]
    smoke_sha256 = _write(smoke_path, smoke_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": "s4-golden-v1",
        "status": "frozen" if frozen else "candidate",
        "chunking_version": "s2-chunk-v1",
        "corpus_manifest_sha256": "0559ee4d75d6079fc2859f8e09910570c4f37f631511f0a9f4348ae7023911d2",
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "files": files,
        "review_inputs": {
            human_path.name: {"sha256": _hash_file(human_path), "count": len(human_rows)},
            round1_path.name: {
                "sha256": _hash_file(round1_path) if round1_path.exists() else None,
                "count": round1_count,
            },
            round2_path.name: {
                "sha256": _hash_file(round2_path) if round2_path.exists() else None,
                "count": round2_count,
            },
        },
        "question_count": 180,
        "answerable_count": sum(1 for row in all_rows if row["answerable"]),
        "boundary_count": sum(1 for row in all_rows if not row["answerable"]),
        "human_written_count": human_written_count,
        "required_human_written_count": HUMAN_REQUIRED_TOTAL,
        "human_review_complete": human_review_complete,
        "test_double_review_complete": test_double_review_complete,
        "test_execution_status": "locked_until_s12",
        "smoke": {"path": smoke_path.name, "count": 30, "sha256": smoke_sha256},
        "category_counts": dict(sorted(Counter(row["category"] for row in all_rows).items())),
        "review_status_counts": dict(
            sorted(Counter(row["review_status"] for row in all_rows).items())
        ),
        "note": (
            "Golden set frozen after merging 36 human-authored questions and completing "
            "two review rounds; Test execution remains locked until S12."
            if frozen
            else "Merged candidate set; two-round human review remains required before frozen status."
        ),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/eval/s4-golden-v1"))
    parser.add_argument("--smoke-output", type=Path, default=Path("data/eval/s4-evaluator-smoke-v1.jsonl"))
    parser.add_argument("--human-authored", type=Path, default=None)
    parser.add_argument("--reviews-round1", type=Path, default=None)
    parser.add_argument("--reviews-test-round2", type=Path, default=None)
    args = parser.parse_args()
    human_path = args.human_authored or (args.output_root / "human_authored.jsonl")
    round1_path = args.reviews_round1 or (args.output_root / "reviews_round1.jsonl")
    round2_path = args.reviews_test_round2 or (args.output_root / "reviews_test_round2.jsonl")
    manifest = freeze(args.chunks, args.output_root, args.smoke_output, human_path, round1_path, round2_path)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
