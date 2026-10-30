"""Read-only project preflight; optional DNS, no paid API or page requests."""

import argparse
import asyncio
import ipaddress
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from docguard.discovery.dns import DohResolver
from docguard.discovery.runtime import build_discovery_service
from docguard.processing.config import load_chunking_config
from docguard.processing.token_budget import TokenBudget


def inspect_readiness(project_root, *, check_dns=False, resolver=None):
    root = Path(project_root)
    report = {
        "schema_version": 1,
        "checks": {},
        "capabilities": {},
        "local_import_configured": False,
        "agent_preflight_ready": False,
        "dns_status": "not_checked",
        "dns": {},
        "provider_authentication": "not_checked",
        "https_robots_html": "not_checked",
        "external_api_calls": 0,
        "credential_source": "process_environment_only",
        "scope": "configuration_assets_and_optional_dns_only",
    }

    def check(name, function):
        try:
            value = function()
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            sqlite3.Error,
        ) as error:
            # No exception text, local paths, credentials or provider body.
            report["checks"][name] = {
                "status": "failed",
                "error_type": type(error).__name__,
            }
            return None
        report["checks"][name] = {"status": "passed"}
        return value

    check(
        "tokenizers",
        lambda: TokenBudget.from_manifest(root / "data/tokenizers/manifest.json", root),
    )
    check(
        "chunking",
        lambda: load_chunking_config(root / "configs/chunking/s2-chunk-v1.json"),
    )

    def configuration():
        # Use the actual assembly validator, but never start tasks or providers.
        # All temporary SQLite state is outside the project and removed on exit.
        with TemporaryDirectory(prefix="docguard-preflight-") as temporary:
            service = build_discovery_service(root, Path(temporary))
            try:
                return (
                    service.capabilities(),
                    sorted(
                        {
                            host
                            for hosts in service.agent.sites.values()
                            for host in hosts
                        }
                    ),
                    service.fetcher.resolver,
                )
            finally:
                asyncio.run(service.close())

    configured = check("discovery_configuration", configuration)
    if configured is not None:
        report["capabilities"], hosts, active_resolver = configured
        report["document_dns_mode"] = (
            "doh" if isinstance(active_resolver, DohResolver) else "system"
        )
    else:
        hosts = []
    local_ok = all(item["status"] == "passed" for item in report["checks"].values())
    report["local_import_configured"] = local_ok
    if check_dns and hosts:
        for host in hosts:
            try:
                addresses = sorted(
                    {
                        str(ipaddress.ip_address(address))
                        for address in (resolver or active_resolver)(host)
                    }
                )
                status = (
                    "public_addresses_only"
                    if addresses
                    and all(
                        ipaddress.ip_address(address).is_global for address in addresses
                    )
                    else "blocked_non_public"
                )
                report["dns"][host] = {"status": status, "addresses": addresses}
            except (OSError, ValueError, TypeError) as error:
                report["dns"][host] = {
                    "status": "resolution_failed",
                    "error_type": type(error).__name__,
                }
        report["dns_status"] = (
            "passed"
            if all(
                item["status"] == "public_addresses_only"
                for item in report["dns"].values()
            )
            else "blocked"
        )
    if local_ok and report["capabilities"].get("agent_enabled"):
        report["agent_preflight_ready"] = (
            report["dns_status"] == "passed" if check_dns else None
        )
    report["blockers"] = []
    if not local_ok:
        report["blockers"].append("local_assets_or_configuration_invalid")
    if not report["capabilities"].get("decision_model_configured"):
        report["blockers"].append(
            "process_deepseek_key_missing_or_configuration_invalid"
        )
    if not report["capabilities"].get("search_configured"):
        provider = report["capabilities"].get("search_provider", "search")
        report["blockers"].append(
            f"process_{provider}_key_missing_or_configuration_invalid"
        )
    if report["dns_status"] == "blocked":
        report["blockers"].append("official_document_dns_blocked")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dns",
        action="store_true",
        help="Resolve configured official hosts; does not fetch pages",
    )
    parser.add_argument(
        "--require-agent",
        action="store_true",
        help="Exit 2 unless local assets, process keys and opt-in DNS checks pass",
    )
    args = parser.parse_args(argv)
    report = inspect_readiness(args.project_root, check_dns=args.dns)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return (
        0
        if report["local_import_configured"]
        and (not args.require_agent or report["agent_preflight_ready"] is True)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
