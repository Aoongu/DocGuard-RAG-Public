from __future__ import annotations

import importlib

from docguard.engineering.executors import EngineeringExecutors


def test_default_api_wires_s11_background_executors() -> None:
    module = importlib.import_module("docguard.api.main")
    assert isinstance(module.EXECUTORS, EngineeringExecutors)
    assert module.ARTIFACTS._rebuild_submitter is not None
    assert module.EVALUATIONS._executor == module.EXECUTORS.run_evaluation
    assert module.SERVICES.shutdown == module.EXECUTORS.close


def test_default_api_wires_persistent_document_import():
    from docguard.discovery.importer import DiscoveryService

    module = importlib.import_module("docguard.api.main")
    assert isinstance(module.SERVICES.discovery, DiscoveryService)
