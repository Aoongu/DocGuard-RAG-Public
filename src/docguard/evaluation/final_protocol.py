"""Strict, hash-bound protocol for the one-shot S12 Test evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SHA256 = re.compile(r"[0-9a-f]{64}")
_METHOD_ORDER = (
    "b0",
    "b1",
    "m1",
    "m2",
    "b2",
    "m3",
    "m4",
    "m4_no_version_filter",
    "m4_no_citation_validation",
)
_ABLATIONS = {
    "no_bm25": "b1",
    "no_dense": "b0",
    "no_cross_encoder": "m1",
    "prompt_only": "m3",
    "no_version_filter": "m4_no_version_filter",
    "no_citation_validation": "m4_no_citation_validation",
}
_TOP_FIELDS = {
    "schema_version",
    "protocol_version",
    "dataset",
    "frozen_inputs",
    "method_order",
    "methods",
    "ablations",
    "output_root",
    "max_retries",
    "existing_record_policy",
    "bootstrap",
}
_DATASET_FIELDS = {
    "version",
    "manifest_path",
    "manifest_sha256",
    "test_path",
    "test_sha256",
    "test_count",
    "review_status",
}
_METHOD_FIELDS = {"run_id", "output_dir", "parameters"}


@dataclass(frozen=True, slots=True)
class FrozenInput:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FinalDataset:
    version: str
    manifest_path: str
    manifest_sha256: str
    test_path: str
    test_sha256: str
    test_count: int
    review_status: str


@dataclass(frozen=True, slots=True)
class FinalMethod:
    name: str
    run_id: str
    output_dir: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BootstrapProtocol:
    samples: int
    seed: int
    confidence_level: float


@dataclass(frozen=True, slots=True)
class S12FinalProtocol:
    schema_version: int
    protocol_version: str
    dataset: FinalDataset
    frozen_inputs: tuple[FrozenInput, ...]
    method_order: tuple[str, ...]
    methods: Mapping[str, FinalMethod]
    ablations: Mapping[str, str]
    output_root: str
    max_retries: int
    existing_record_policy: str
    bootstrap: BootstrapProtocol


@dataclass(frozen=True, slots=True)
class S12TestAuthorization:
    """Hash-bound capability for exactly one pre-registered Test run."""

    protocol_version: str
    method: str
    run_id: str
    test_sha256: str
    output_dir: Path
    question_ids: tuple[str, ...]
    question_contract_sha256: str
    parameters: Mapping[str, object]

    def validate(
        self,
        output_dir: Path,
        metadata: Mapping[str, object],
        questions: Sequence[object],
    ) -> None:
        if self.protocol_version != "s12-final-v1":
            raise ValueError("authorization protocol identity is invalid")
        if output_dir.resolve() != self.output_dir:
            raise ValueError("authorization output directory does not match")
        if metadata.get("run_id") != self.run_id:
            raise ValueError("authorization run ID does not match")
        if metadata.get("split") != "test":
            raise ValueError("authorization only permits the Test split")
        if metadata.get("dataset_sha256") != self.test_sha256:
            raise ValueError("authorization Test SHA-256 does not match")
        if metadata.get("method") != self.method and not (
            self.method == "b2" and metadata.get("route") == "b2"
        ):
            raise ValueError("authorization method does not match")
        if metadata.get("parameters") != self.parameters:
            raise ValueError("authorization parameters do not match")
        try:
            rows = [question.to_dict() for question in questions]  # type: ignore[attr-defined]
        except AttributeError as error:
            raise ValueError("authorization questions must be serializable") from error
        identifiers = tuple(sorted(str(row.get("question_id")) for row in rows))
        if identifiers != self.question_ids:
            raise ValueError(
                "authorization requires the complete frozen Test question set"
            )
        canonical = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in sorted(rows, key=lambda item: str(item.get("question_id")))
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != self.question_contract_sha256:
            raise ValueError("authorization question content differs from frozen Test")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields do not match the frozen schema")


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def _expected_parameters() -> dict[str, dict[str, object]]:
    return {
        "b0": {"dense_enabled": False, "bm25_enabled": True},
        "b1": {"dense_enabled": True, "bm25_enabled": False},
        "m1": {
            "dense_enabled": True,
            "bm25_enabled": True,
            "dense_depth": 50,
            "bm25_depth": 20,
            "rrf_k": 20,
        },
        "m2": {
            "dense_enabled": True,
            "bm25_enabled": True,
            "dense_depth": 50,
            "bm25_depth": 20,
            "rrf_k": 20,
            "rerank_enabled": True,
            "rerank_depth": 50,
        },
        "b2": {
            "dense_enabled": True,
            "bm25_enabled": False,
            "rerank_enabled": False,
            "generation_enabled": True,
            "citation_validation_enabled": True,
            "top_k": 10,
        },
        "m3": {
            "refusal_enabled": False,
            "version_filter_enabled": True,
            "generation_enabled": True,
            "citation_validation_enabled": True,
        },
        "m4": {
            "refusal_enabled": True,
            "version_filter_enabled": True,
            "generation_enabled": True,
            "citation_validation_enabled": True,
        },
        "m4_no_version_filter": {
            "refusal_enabled": True,
            "version_filter_enabled": False,
            "generation_enabled": True,
            "citation_validation_enabled": True,
        },
        "m4_no_citation_validation": {
            "refusal_enabled": True,
            "version_filter_enabled": True,
            "generation_enabled": True,
            "citation_validation_enabled": False,
        },
    }


def load_final_protocol(path: Path) -> S12FinalProtocol:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "S12 config")
    _exact_fields(payload, _TOP_FIELDS, "S12 config")
    if payload["schema_version"] != 1 or payload["protocol_version"] != "s12-final-v1":
        raise ValueError("unsupported S12 protocol identity")

    dataset = _mapping(payload["dataset"], "dataset")
    _exact_fields(dataset, _DATASET_FIELDS, "dataset")
    if dataset["version"] != "s4-golden-v1":
        raise ValueError("dataset version must remain s4-golden-v1")
    if (
        dataset["test_sha256"]
        != "67bbafdb8841a15f58de0cd66cf14f488f142a2f1e44bff360ef17647e56adeb"
    ):
        raise ValueError("Test SHA-256 differs from the frozen dataset")
    if dataset["test_count"] != 60 or dataset["review_status"] != "double_reviewed":
        raise ValueError("Test must contain 60 double-reviewed questions")
    final_dataset = FinalDataset(
        version="s4-golden-v1",
        manifest_path=_relative_path(dataset["manifest_path"], "dataset.manifest_path"),
        manifest_sha256=_sha(dataset["manifest_sha256"], "dataset.manifest_sha256"),
        test_path=_relative_path(dataset["test_path"], "dataset.test_path"),
        test_sha256=_sha(dataset["test_sha256"], "dataset.test_sha256"),
        test_count=60,
        review_status="double_reviewed",
    )

    frozen_rows = payload["frozen_inputs"]
    if not isinstance(frozen_rows, list) or not frozen_rows:
        raise ValueError("frozen_inputs must be a non-empty array")
    frozen_inputs: list[FrozenInput] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(frozen_rows):
        item = _mapping(raw, f"frozen_inputs[{index}]")
        _exact_fields(item, {"path", "sha256"}, f"frozen_inputs[{index}]")
        relative = _relative_path(item["path"], f"frozen_inputs[{index}].path")
        if relative in seen_paths:
            raise ValueError("frozen_inputs paths must be unique")
        seen_paths.add(relative)
        frozen_inputs.append(FrozenInput(relative, _sha(item["sha256"], relative)))

    method_order = payload["method_order"]
    if tuple(method_order) != _METHOD_ORDER if isinstance(method_order, list) else True:
        raise ValueError("method_order does not match the frozen S12 order")
    methods_raw = _mapping(payload["methods"], "methods")
    if set(methods_raw) != set(_METHOD_ORDER):
        raise ValueError("methods do not match the frozen S12 matrix")
    expected = _expected_parameters()
    methods: dict[str, FinalMethod] = {}
    run_ids: set[str] = set()
    output_dirs: set[str] = set()
    for name in _METHOD_ORDER:
        method = _mapping(methods_raw[name], f"methods.{name}")
        _exact_fields(method, _METHOD_FIELDS, f"methods.{name}")
        run_id = method["run_id"]
        if (
            not isinstance(run_id, str)
            or run_id != f"s12-test-{name.replace('_', '-')}-v1"
        ):
            raise ValueError(f"{name} run ID does not match the frozen identity")
        output_dir = _relative_path(method["output_dir"], f"methods.{name}.output_dir")
        parameters = _mapping(method["parameters"], f"methods.{name}.parameters")
        if parameters != expected[name]:
            label = (
                "M1 parameters"
                if name == "m1"
                else "M2 parameters"
                if name == "m2"
                else f"{name} parameters"
            )
            raise ValueError(f"{label} differ from the frozen contract")
        if run_id in run_ids or output_dir in output_dirs:
            raise ValueError("method run ID and output directory must be unique")
        run_ids.add(run_id)
        output_dirs.add(output_dir)
        methods[name] = FinalMethod(name, run_id, output_dir, dict(parameters))

    ablations = _mapping(payload["ablations"], "ablations")
    if ablations != _ABLATIONS:
        raise ValueError("ablations do not match the frozen mapping")
    if payload["output_root"] != "reports/s12_final/raw":
        raise ValueError("output_root must remain reports/s12_final/raw")
    if payload["max_retries"] != 0:
        raise ValueError("max_retries must remain zero")
    if payload["existing_record_policy"] != "skip_any_record":
        raise ValueError("existing_record_policy must remain skip_any_record")
    bootstrap = _mapping(payload["bootstrap"], "bootstrap")
    _exact_fields(bootstrap, {"samples", "seed", "confidence_level"}, "bootstrap")
    if bootstrap != {"samples": 10000, "seed": 20261014, "confidence_level": 0.95}:
        raise ValueError("bootstrap contract differs from the frozen values")

    return S12FinalProtocol(
        schema_version=1,
        protocol_version="s12-final-v1",
        dataset=final_dataset,
        frozen_inputs=tuple(frozen_inputs),
        method_order=_METHOD_ORDER,
        methods=methods,
        ablations=dict(ablations),
        output_root="reports/s12_final/raw",
        max_retries=0,
        existing_record_policy="skip_any_record",
        bootstrap=BootstrapProtocol(10000, 20261014, 0.95),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_protocol_inputs(root: Path, protocol: S12FinalProtocol) -> None:
    manifest = root / protocol.dataset.manifest_path
    test_data = root / protocol.dataset.test_path
    if _sha256(manifest) != protocol.dataset.manifest_sha256:
        raise ValueError("dataset manifest hash differs from the frozen protocol")
    if _sha256(test_data) != protocol.dataset.test_sha256:
        raise ValueError("test dataset hash differs from the frozen protocol")
    rows = [line for line in test_data.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != protocol.dataset.test_count:
        raise ValueError("test dataset count differs from the frozen protocol")
    for item in protocol.frozen_inputs:
        if _sha256(root / item.path) != item.sha256:
            raise ValueError(f"frozen input hash differs: {item.path}")


def authorize_test_run(
    root: Path, protocol: S12FinalProtocol, method: str
) -> S12TestAuthorization:
    """Create a capability only after the exact frozen Test bytes are present."""
    if method not in protocol.methods:
        raise ValueError("method is not pre-registered in the S12 protocol")
    test_path = root / protocol.dataset.test_path
    if _sha256(test_path) != protocol.dataset.test_sha256:
        raise ValueError("Test dataset hash differs from the frozen protocol")
    rows = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != protocol.dataset.test_count or not all(
        isinstance(row, dict) and isinstance(row.get("question_id"), str)
        for row in rows
    ):
        raise ValueError("Test dataset does not match the frozen question contract")
    ordered = sorted(rows, key=lambda item: str(item["question_id"]))
    canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in ordered
    ).encode("utf-8")
    frozen = protocol.methods[method]
    output_dir = (root / protocol.output_root / frozen.output_dir).resolve()
    return S12TestAuthorization(
        protocol_version=protocol.protocol_version,
        method=method,
        run_id=frozen.run_id,
        test_sha256=protocol.dataset.test_sha256,
        output_dir=output_dir,
        question_ids=tuple(str(row["question_id"]) for row in ordered),
        question_contract_sha256=hashlib.sha256(canonical).hexdigest(),
        parameters=dict(frozen.parameters),
    )
