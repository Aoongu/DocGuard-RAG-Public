"""Lazy local knowledge assembly; no model weights or paid API required at startup."""

import json
import os
import re
from pathlib import Path

from docguard.discovery.dns import configured_resolver
from docguard.discovery.fetch import SafeFetcher
from docguard.discovery.graph import AgentLimits
from docguard.discovery.importer import DiscoveryService
from docguard.discovery.providers import (
    BraveSearch,
    DeepSeekDecisionModel,
    SerpAPISearch,
)
from docguard.generation.client import DeepSeekResponsesClient
from docguard.generation.config import load_generation_config
from docguard.knowledge.local_models import LocalRetrievalModels
from docguard.knowledge.pipeline import KnowledgeBase
from docguard.processing.config import load_chunking_config
from docguard.processing.token_budget import TokenBudget


def build_discovery_service(
    project_root: Path, data_root: Path, *, asset_root: Path | None = None
) -> DiscoveryService:
    models = LocalRetrievalModels(project_root, asset_root or project_root)

    def factory():
        budget = TokenBudget.from_manifest(
            project_root / "data/tokenizers/manifest.json", project_root
        )
        config = load_chunking_config(
            project_root / "configs/chunking/s2-chunk-v1.json"
        )
        return KnowledgeBase(
            data_root / "knowledge",
            budget,
            config,
            embedder=models,
            embedding_id=models.embedding_id,
            reranker=models,
            reranker_id=models.reranker_id,
            default_method="full",
        )

    settings = json.loads(
        (project_root / "configs/discovery/s15-discovery-v1.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(settings, dict)
        or set(settings)
        != {
            "schema_version",
            "agent_version",
            "limits",
            "search_provider",
            "model_connection_config",
        }
        or settings["agent_version"] != "s15-discovery-agent-v1"
    ):
        raise ValueError("Unsupported discovery configuration fields")
    if (
        settings["schema_version"] != 1
        or settings["search_provider"] not in ("brave", "serpapi")
        or settings["model_connection_config"]
        != "configs/generation/s7-deepseek-flash-v1.json"
    ):
        raise ValueError("Unsupported discovery configuration")
    sites = json.loads(
        (project_root / "configs/discovery/sites.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(sites, dict)
        or not sites
        or any(
            not isinstance(name, str)
            or not isinstance(domains, list)
            or not domains
            or any(
                not isinstance(host, str)
                or re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", host) is None
                for host in domains
            )
            for name, domains in sites.items()
        )
    ):
        raise ValueError("Invalid official documentation domains")
    model_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    provider_name = (
        os.environ.get("DOCGUARD_SEARCH_PROVIDER", settings["search_provider"])
        .strip()
        .lower()
    )
    if provider_name not in ("brave", "serpapi"):
        raise ValueError("Unsupported search provider")
    search_key_env = (
        "SERPAPI_API_KEY" if provider_name == "serpapi" else "BRAVE_SEARCH_API_KEY"
    )
    search_key = os.environ.get(search_key_env, "").strip()
    model = (
        DeepSeekDecisionModel(
            load_generation_config(project_root / settings["model_connection_config"]),
            model_key,
        )
        if model_key
        else None
    )
    search_class = SerpAPISearch if provider_name == "serpapi" else BraveSearch
    search = search_class(search_key) if search_key else None
    service = DiscoveryService(
        data_root / "tasks",
        factory,
        fetcher=SafeFetcher(resolver=configured_resolver()),
        decision_model=model,
        search_provider=search,
        agent_limits=AgentLimits(**settings["limits"]),
        sites=sites,
        query_generator_factory=(
            lambda: DeepSeekResponsesClient(model.config, api_key=model_key)
        )
        if model is not None
        else None,
    )
    service.search_provider_name = provider_name
    service.document_dns_mode = (
        os.environ.get("DOCGUARD_DOCUMENT_DNS", "system").strip().lower()
    )
    return service
