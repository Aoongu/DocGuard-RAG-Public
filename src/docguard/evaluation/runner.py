"""Resumable per-question evaluation runner with a hard Test-split lock."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from docguard.evaluation.config import EvaluationConfig
from docguard.evaluation.final_protocol import S12TestAuthorization
from docguard.evaluation.metrics import (
    aggregate_retrieval_metrics,
    retrieval_metrics,
    summarize_operations,
)
from docguard.evaluation.models import EvaluationQuestion
from docguard.fs_safety import ensure_directory


class TestSplitLockedError(RuntimeError):
    """Raised when code attempts to run the frozen Test split before S12."""

    __test__ = False


class RunContractError(RuntimeError):
    """Raised when a resumed run does not match its immutable metadata."""


class EvaluationRetriever(Protocol):
    async def search(
        self,
        query: str,
        version: str | None,
        top_k: int,
        **options: object,
    ) -> Sequence[Any]: ...


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            raise RunContractError(f"invalid saved result at line {line_number}")
        question_id = row["question_id"]
        if question_id in rows:
            raise RunContractError(f"duplicate saved question_id: {question_id}")
        rows[question_id] = row
    return rows


def _write_result_views(output_dir: Path, rows: Mapping[str, dict[str, object]]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    _atomic_text(output_dir / "results.jsonl", "".join(_json_line(row) for row in ordered))
    failures = [row for row in ordered if row["status"] == "failed"]
    failure_text = "".join(
        _json_line(
            {
                "question_id": row["question_id"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
            }
        )
        for row in failures
    )
    _atomic_text(output_dir / "failures.jsonl", failure_text)


def _validate_run_contract(
    questions: Sequence[EvaluationQuestion],
    config: EvaluationConfig,
    metadata: Mapping[str, object],
    output_dir: Path,
    test_authorization: S12TestAuthorization | None,
) -> str:
    split = metadata.get("split")
    if split == "test" or any(question.split == "test" for question in questions):
        if split != "test" or any(question.split != "test" for question in questions):
            raise RunContractError("all Test questions must match the run split")
        if test_authorization is None:
            raise TestSplitLockedError("Test split requires the frozen S12 authorization")
        try:
            test_authorization.validate(output_dir, metadata, questions)
        except ValueError as error:
            raise RunContractError(f"S12 authorization failed: {error}") from error
    elif test_authorization is not None:
        raise RunContractError("S12 Test authorization cannot be used for a non-Test run")
    elif split not in config.allowed_splits:
        raise RunContractError("run split is not allowed by the frozen config")
    if not questions:
        raise RunContractError("run requires at least one question")
    if any(question.split != split for question in questions):
        raise RunContractError("all question splits must match run metadata")
    if metadata.get("dataset_version") != config.dataset_version:
        raise RunContractError("dataset version does not match config")
    if metadata.get("chunking_version") != config.chunking_version:
        raise RunContractError("chunking version does not match config")
    if metadata.get("corpus_manifest_sha256") != config.corpus_manifest_sha256:
        raise RunContractError("corpus manifest hash does not match config")
    if metadata.get("chunks_sha256") != config.chunks_sha256:
        raise RunContractError("chunks hash does not match config")
    if not isinstance(split, str):
        raise RunContractError("split must be a string")
    return split


def _retrieval_options(metadata: Mapping[str, object]) -> dict[str, object]:
    method = metadata.get("method")
    if method is None:
        return {"dense_enabled": True}
    if method not in {"b0", "b1", "m1", "m2"}:
        raise RunContractError("method must be b0, b1, m1, or m2")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        raise RunContractError("S5 run parameters must be an object")
    expected_switches = {
        "b0": (False, True),
        "b1": (True, False),
        "m1": (True, True),
        "m2": (True, True),
    }[method]
    if (
        parameters.get("dense_enabled"),
        parameters.get("bm25_enabled"),
    ) != expected_switches:
        raise RunContractError("S5 route switches do not match the method")
    for field in ("s5_config_sha256", "dense_index_manifest_sha256", "bm25_index_manifest_sha256"):
        value = metadata.get(field)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RunContractError(f"{field} must be null or a lowercase SHA-256")
    if not isinstance(metadata.get("s5_config_sha256"), str):
        raise RunContractError("S5 config hash is required")
    if method in {"b1", "m1", "m2"} and not isinstance(
        metadata.get("dense_index_manifest_sha256"), str
    ):
        raise RunContractError("Dense index manifest hash is required")
    if method in {"b0", "m1", "m2"} and not isinstance(
        metadata.get("bm25_index_manifest_sha256"), str
    ):
        raise RunContractError("BM25 index manifest hash is required")
    if method == "m2":
        for field in (
            "s6_config_sha256",
            "reranker_model_manifest_sha256",
            "reranker_tokenizer_sha256",
        ):
            value = metadata.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RunContractError(f"{field} must be a lowercase SHA-256")
        if metadata.get("reranker_model_name") != "BAAI/bge-reranker-base":
            raise RunContractError("reranker_model_name does not match S6")
        revision = metadata.get("reranker_model_revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise RunContractError("reranker_model_revision must be a commit SHA")
        if metadata.get("reranker_max_length") != 512:
            raise RunContractError("reranker_max_length must be 512")
        if (
            parameters.get("dense_depth"),
            parameters.get("bm25_depth"),
            parameters.get("rrf_k"),
        ) != (50, 20, 20):
            raise RunContractError("M2 must use the frozen M1 parameters")
        rerank_enabled = parameters.get("rerank_enabled")
        rerank_depth = parameters.get("rerank_depth")
        if type(rerank_enabled) is not bool:
            raise RunContractError("rerank_enabled must be boolean")
        if rerank_depth not in {10, 20, 50}:
            raise RunContractError("rerank_depth is outside the pre-registered grid")
        return {
            "method": method,
            "rerank_enabled": rerank_enabled,
            "rerank_depth": rerank_depth,
        }
    if method == "m1":
        dense_depth = parameters.get("dense_depth")
        bm25_depth = parameters.get("bm25_depth")
        rrf_k = parameters.get("rrf_k")
        if dense_depth not in {20, 50} or bm25_depth not in {20, 50} or rrf_k not in {20, 60}:
            raise RunContractError("M1 parameters are outside the pre-registered grid")
        return {
            "method": method,
            "dense_depth": dense_depth,
            "bm25_depth": bm25_depth,
            "rrf_k": rrf_k,
        }
    return {"method": method}


def _hit_row(hit: Any) -> dict[str, object]:
    chunk_id = hit.chunk_id
    score = float(hit.score)
    rank = int(hit.rank)
    if (
        not isinstance(chunk_id, str)
        or type(hit.rank) is not int
        or rank <= 0
        or not math.isfinite(score)
    ):
        raise RunContractError("retriever returned an invalid hit")
    row: dict[str, object] = {"rank": rank, "chunk_id": chunk_id, "score": score}
    optional = (
        "rrf_score",
        "sources",
        "dense_rank",
        "dense_score",
        "bm25_rank",
        "bm25_score",
        "rerank_score",
        "rerank_truncated",
        "original_rrf_rank",
    )
    for field in optional:
        if not hasattr(hit, field):
            continue
        value = getattr(hit, field)
        if field == "sources" and value is not None:
            if not isinstance(value, (tuple, list)) or not all(
                item in {"dense", "bm25"} for item in value
            ):
                raise RunContractError("retriever returned invalid route sources")
            row[field] = list(value)
        elif field.endswith("_score") and value is not None:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RunContractError("retriever returned a non-finite route score")
            row[field] = numeric
        elif field.endswith("_rank") and value is not None:
            if type(value) is not int or value <= 0:
                raise RunContractError("retriever returned an invalid route rank")
            row[field] = value
        elif field == "rerank_truncated":
            if type(value) is not bool:
                raise RunContractError("retriever returned an invalid truncation flag")
            row[field] = value
        else:
            row[field] = value
    return row


async def run_evaluation(
    questions: Sequence[EvaluationQuestion],
    output_dir: Path,
    retriever: EvaluationRetriever,
    config: EvaluationConfig,
    metadata: Mapping[str, object],
    *,
    clock: Callable[[], float] = time.perf_counter,
    test_authorization: S12TestAuthorization | None = None,
) -> dict[str, object]:
    """Run or resume one immutable evaluation run and return its saved summary."""
    split = _validate_run_contract(
        questions, config, metadata, output_dir, test_authorization
    )
    retrieval_options = _retrieval_options(metadata)
    metadata_dict = dict(metadata)
    ensure_directory(output_dir)
    run_path = output_dir / "run.json"
    if run_path.exists():
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        if saved != metadata_dict:
            raise RunContractError("run metadata differs from the existing run metadata")
    else:
        _atomic_text(run_path, _json_line(metadata_dict))

    results_path = output_dir / "results.jsonl"
    rows = _read_jsonl_by_id(results_path)
    question_ids = {question.question_id for question in questions}
    unexpected = set(rows) - question_ids
    if unexpected:
        raise RunContractError(f"saved results contain unexpected question IDs: {sorted(unexpected)}")

    for question in sorted(questions, key=lambda item: item.question_id):
        previous = rows.get(question.question_id)
        if previous is not None:
            continue
        started = clock()
        try:
            hits = await retriever.search(
                question.question,
                question.requested_version,
                config.retrieval_top_k,
                **retrieval_options,
            )
            retrieved_at = clock()
            stage_latency = {
                "retrieval": (retrieved_at - started) * 1000.0,
            }
            if metadata_dict.get("method") == "m2":
                consume_timing = getattr(retriever, "consume_timing_ms", None)
                if not callable(consume_timing):
                    raise RunContractError("M2 retriever does not expose stage timing")
                timing = consume_timing()
                if not isinstance(timing, Mapping) or set(timing) != {
                    "retrieval",
                    "reranking",
                }:
                    raise RunContractError("M2 stage timing is incomplete")
                stage_latency = {}
                for stage in ("retrieval", "reranking"):
                    numeric = float(timing[stage])
                    if not math.isfinite(numeric) or numeric < 0:
                        raise RunContractError("M2 stage timing must be finite and non-negative")
                    stage_latency[stage] = numeric
            hit_rows: list[dict[str, object]] = []
            retrieved_ids: list[str] = []
            for hit in hits:
                row = _hit_row(hit)
                retrieved_ids.append(str(row["chunk_id"]))
                hit_rows.append(row)
            if len(retrieved_ids) != len(set(retrieved_ids)):
                raise RunContractError("retriever returned duplicate chunk IDs")
            ended = clock()
            metrics = (
                retrieval_metrics(
                    retrieved_ids,
                    question.resolved_chunk_ids,
                    config.retrieval_cutoffs,
                    config.mrr_cutoff,
                    config.ndcg_cutoff,
                )
                if question.answerable
                else {}
            )
            rows[question.question_id] = {
                "schema_version": 1,
                "question_id": question.question_id,
                "split": split,
                "answerable": question.answerable,
                "status": "success",
                "hits": hit_rows,
                "metrics": metrics,
                "latency_ms": {
                    **stage_latency,
                    "end_to_end": (ended - started) * 1000.0,
                },
                "usage": {"input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0},
                "error_type": None,
                "error_message": None,
            }
        except Exception as error:  # noqa: BLE001 - per-question isolation is the contract
            ended = clock()
            rows[question.question_id] = {
                "schema_version": 1,
                "question_id": question.question_id,
                "split": split,
                "answerable": question.answerable,
                "status": "failed",
                "hits": [],
                "metrics": {},
                "latency_ms": {"end_to_end": (ended - started) * 1000.0},
                "usage": {"input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0},
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        _write_result_views(output_dir, rows)

    ordered = [rows[key] for key in sorted(rows)]
    results_bytes = (output_dir / "results.jsonl").read_bytes()
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": metadata_dict.get("run_id"),
        "split": split,
        "dataset_version": metadata_dict.get("dataset_version"),
        "dataset_sha256": metadata_dict.get("dataset_sha256"),
        "config_sha256": metadata_dict.get("config_sha256"),
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "question_count": len(questions),
        "retrieval": aggregate_retrieval_metrics(ordered),
        "operations": summarize_operations(ordered),
    }
    _atomic_text(output_dir / "summary.json", _json_line(summary))
    return summary
