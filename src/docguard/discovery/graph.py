"""Bounded single-agent LangGraph loop, checkpointed in the P1 task ledger."""

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from langgraph.graph import END, START, StateGraph
from pydantic import Field

from docguard.discovery.collection import document_kind, scope_for_entry
from docguard.discovery.extract import extract_html
from docguard.discovery.fetch import FetchError, validate_url
from docguard.discovery.models import StrictInput
from docguard.discovery.providers import (
    ACTION_SCHEMA,
    SEARCH_FAILURE_MESSAGES,
    SYSTEM_PROMPT,
    ProviderError,
)
from docguard.discovery.store import Conflict


@dataclass(frozen=True)
class AgentLimits:
    max_model_calls: int = 8
    max_searches: int = 3
    max_reads: int = 10
    max_input_tokens: int = 50000
    max_output_tokens: int = 6400
    max_prompt_bytes: int = 20000
    max_model_cost_usd: float = 0.05
    max_active_seconds: float = 120

    def __post_init__(self):
        for field in (
            "max_model_calls",
            "max_searches",
            "max_reads",
            "max_input_tokens",
            "max_output_tokens",
            "max_prompt_bytes",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError("Agent count limits must be positive integers")
        if self.max_model_calls > 16:
            raise ValueError(
                "Agent model call limit cannot exceed graph recursion capacity"
            )
        for field in ("max_model_cost_usd", "max_active_seconds"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Agent resource limits must be finite and positive")


class Action(StrictInput):
    action: Literal[
        "search_documents",
        "read_page",
        "inspect_links",
        "assess_source",
        "ask_user",
        "present_candidates",
        "finish",
    ]
    query: str | None = Field(default=None, max_length=1000)
    candidate_id: str | None = Field(default=None, max_length=100)
    candidate_ids: list[str] = Field(default_factory=list, max_length=20)
    question: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default=None, max_length=240)


class GraphState(TypedDict):
    task_id: str
    execute: bool


DEFAULT_SITES = {
    "python": ["docs.python.org"],
    "redis": ["redis.io"],
    "postgresql": ["www.postgresql.org"],
    "java": ["docs.oracle.com"],
}


def initial_agent_state():
    return {
        "model_calls": 0,
        "searches": 0,
        "reads": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "model_cost_usd": 0.0,
        "reserved_input_tokens": 0,
        "reserved_output_tokens": 0,
        "reserved_model_cost_usd": 0.0,
        "usage_complete": True,
        "pool": [],
        "last_observation": {},
        "pending_action": None,
        "model_pending": False,
        "active_seconds": 0.0,
        "messages": [],
    }


def _failed_pages(row):
    return {
        key: value
        for key, value in row["agent"].get("failed_pages", {}).items()
        if value["revision"] == row["revision"]
    }


def _unread_ids(row):
    excluded = {c["candidate_id"] for c in row["candidates"]} | set(_failed_pages(row))
    return [
        c["candidate_id"]
        for c in row["agent"]["pool"]
        if c["candidate_id"] not in excluded
    ]


def _search_block(row, query):
    normalized = " ".join((query or "").casefold().split())
    history = row["agent"].get("search_history", [])
    if any(
        item["normalized_query"] == normalized and item["revision"] == row["revision"]
        for item in history
    ):
        return "duplicate_search_query"
    if (
        history
        and history[-1]["revision"] == row["revision"]
        and history[-1]["new_candidate_count"] == 0
        and _unread_ids(row)
    ):
        return "search_results_stalled_read_existing_candidates"
    return None


def _version_status(url, title, requested):
    if requested is None:
        return "not_requested"
    path = urlsplit(url).path
    match = re.search(r"/(?:v)?(\d+(?:\.\d+)*)(?:/|$)", path)
    if match is None:
        match = re.search(
            r"\b(?:Python|Redis|PostgreSQL)\s+(\d+(?:\.\d+)*)\b", title, re.IGNORECASE
        )
    if match is None:
        return "unknown"
    found = match.group(1)
    if found == requested or found.startswith(requested + "."):
        return "url_version_match"
    if requested.startswith(found + "."):
        return "unknown"
    return "version_mismatch"


class DiscoveryAgent:
    def __init__(
        self, store, model, search, fetcher, *, limits=None, sites=None, stopping=None
    ):
        self.store, self.model, self.search, self.fetcher = (
            store,
            model,
            search,
            fetcher,
        )
        self.limits, self.sites, self.stopping = (
            limits or AgentLimits(),
            sites or DEFAULT_SITES,
            stopping or (lambda: False),
        )
        builder = StateGraph(GraphState)
        builder.add_node("decide", self._decide)
        builder.add_node("tool", self._tool)
        builder.add_edge(START, "decide")
        builder.add_conditional_edges(
            "decide",
            lambda s: "tool" if s["execute"] else END,
            {"tool": "tool", END: END},
        )
        builder.add_conditional_edges(
            "tool",
            lambda s: (
                "decide"
                if not self.stopping()
                and self.store.get(s["task_id"])["status"] == "discovering"
                else END
            ),
            {"decide": "decide", END: END},
        )
        self.graph = builder.compile()

    def _change(self, task_id, change, kind, payload=None):
        def guarded(row):
            if row["status"] != "discovering":
                raise Conflict("Discovery task is no longer active")
            row.setdefault("agent", initial_agent_state())
            change(row)

        return self.store.mutate(task_id, guarded, kind, payload)

    def _stop(self, task_id, reason, details=None):
        def stop(row):
            row["stop_reason"] = reason
            candidates = [
                c
                for c in row["candidates"]
                if (
                    row["version"] is None
                    or c["version_status"] in ("url_version_match", "not_requested")
                )
                and (
                    not row.get("official_only", True)
                    or c["source_status"]
                    in ("official_domain", "agent_assessed_official")
                )
            ]
            if row.get("document_type", "topic") != "topic":
                candidates = [
                    c
                    for c in candidates
                    if scope_for_entry(c["url"]) is not None
                    and document_kind(c["url"], c["title"])
                    in (None, row["document_type"])
                ]
            for candidate in candidates:
                candidate["collection"] = scope_for_entry(candidate["url"])
            row["candidates"] = candidates
            row["status"] = "awaiting_selection" if candidates else "exhausted"

        row = self.store.get(task_id)
        usage = {
            key: row["agent"].get(key, 0)
            for key in (
                "model_calls",
                "searches",
                "reads",
                "input_tokens",
                "output_tokens",
                "model_cost_usd",
                "active_seconds",
                "reserved_input_tokens",
                "reserved_output_tokens",
                "reserved_model_cost_usd",
            )
        }
        self._change(
            task_id,
            stop,
            "agent_stopped",
            {
                "reason": reason,
                "usage": usage,
                "limits": asdict(self.limits),
                **(details or {}),
            },
        )

    async def run(self, task_id):
        connection = getattr(self.model, "config", None)
        identity = {
            "agent_version": "s15-discovery-agent-v1",
            "accounting_version": "settled-reservations-v2",
            "context_version": "network-bounded-collection-v1",
            "limits": asdict(self.limits),
            "sites": self.sites,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "action_schema": ACTION_SCHEMA,
            "model": getattr(connection, "model", "external_provider"),
            "base_url": getattr(connection, "default_base_url", None),
            "search_provider": getattr(
                self.search, "provider_id", type(self.search).__name__
            ),
        }
        configuration_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

        def begin(row):
            if row["status"] not in ("discovering", "awaiting_input"):
                raise Conflict("Agent task is no longer active")
            row["status"] = "discovering"
            row.setdefault("agent", initial_agent_state())
            if (
                row["agent"].get("configuration_id", configuration_id)
                != configuration_id
            ):
                row.update(status="failed", stop_reason="agent_configuration_changed")
                return
            row["agent"].update(
                configuration_id=configuration_id, configuration=identity
            )
            row.update(pending_question=None, stop_reason=None)

        try:
            self.store.mutate(task_id, begin, "agent_started")
        except Conflict:
            return
        if self.store.get(task_id)["status"] == "failed":
            return
        if self.model is None or self.search is None:

            def unavailable(row):
                row.update(
                    status="awaiting_input",
                    stop_reason="agent_dependencies_unavailable",
                    pending_question="文档搜索服务尚未配置，请提供文档网址，或配置模型和搜索服务后继续。",
                )

            self._change(task_id, unavailable, "agent_unavailable")
            return
        try:
            await self.graph.ainvoke(
                {"task_id": task_id, "execute": False}, config={"recursion_limit": 40}
            )
        except Conflict:
            return

    def _context(self, row):
        agent = row["agent"]
        read_ids = {c["candidate_id"] for c in row["candidates"]}
        unread_ids = _unread_ids(row)
        links_ids = sorted(
            i for i in read_ids if i not in agent.get("links_inspected", [])
        )
        options = []
        for candidate in row["candidates"]:
            if candidate["source_status"] != "unverified":
                continue
            authorities = [
                c["candidate_id"]
                for c in row["candidates"]
                if c["candidate_id"] != candidate["candidate_id"]
                and candidate["url"]
                in agent.get("page_links", {}).get(c["candidate_id"], [])
            ]
            if authorities:
                options.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "authority_candidate_ids": authorities[:5],
                    }
                )
        available = ["ask_user", "finish"]
        history = agent.get("search_history", [])
        stalled = (
            history
            and history[-1]["revision"] == row["revision"]
            and history[-1]["new_candidate_count"] == 0
            and unread_ids
        )
        if agent["searches"] < self.limits.max_searches and not stalled:
            available.append("search_documents")
        if unread_ids and agent["reads"] < self.limits.max_reads:
            available.append("read_page")
        if links_ids:
            available.append("inspect_links")
        if options:
            available.append("assess_source")
        if any(
            (
                not row.get("official_only", True)
                or c["source_status"] in ("official_domain", "agent_assessed_official")
            )
            and (
                row["version"] is None
                or c["version_status"] in ("url_version_match", "not_requested")
            )
            for c in row["candidates"]
        ):
            available.append("present_candidates")
        context = {
            "goal": row["goal"],
            "document_type": row.get("document_type", "user_guide"),
            "selection_instruction": "Choose the requested documentation collection entry. user_guide means comprehensive User Guide/manual, not Getting started or a single chapter. Follow actual navigation links. topic permits a specific chapter. Do not substitute an unrelated entry.",
            "product": row["product"],
            "version": row["version"],
            "official_only": row.get("official_only", True),
            "messages": agent["messages"][-5:],
            "pool": agent["pool"][:30],
            "candidates": row["candidates"],
            "last_observation": agent["last_observation"],
            "remaining_model_calls": self.limits.max_model_calls - agent["model_calls"],
            "remaining_searches": self.limits.max_searches - agent["searches"],
            "remaining_reads": self.limits.max_reads - agent["reads"],
            "official_sites": self.sites,
            "available_actions": available,
            "unread_candidate_ids": unread_ids,
            "inspectable_link_candidate_ids": links_ids,
            "source_assessment_options": options,
            "recent_actions": agent.get("recent_actions", [])[-6:],
            "search_history": history[-3:],
            "failed_pages": [
                {"candidate_id": key, **value}
                for key, value in _failed_pages(row).items()
            ],
        }
        # Persist full observations in the ledger; send a bounded working view
        # rather than repeating search results, URLs and navigation snippets.
        context["pool"] = [
            {
                "candidate_id": c["candidate_id"],
                "url": c["url"],
                "title": c["title"][:120],
                "snippet": c.get("snippet", "")[:120],
            }
            for c in context["pool"]
        ]
        observation = dict(context["last_observation"])
        if "results" in observation:
            observation["results"] = [
                {"candidate_id": c["candidate_id"], "title": c.get("title", "")[:120]}
                for c in observation["results"]
            ]
        navigation = observation.get("navigation_links", [])[:10]
        if navigation:
            observation.pop("links", None)
            observation["navigation_links"] = [
                {"candidate_id": c["candidate_id"], "title": c.get("title", "")[:120]}
                for c in navigation
                if c.get("candidate_id")
            ]
        if "excerpt" in observation:
            observation["excerpt"] = observation["excerpt"][:1000]
        context["last_observation"] = observation
        protected = read_ids | {
            c["candidate_id"] for c in navigation[:3] if c.get("candidate_id")
        }
        while len(SYSTEM_PROMPT.encode()) + len(
            json.dumps(context, ensure_ascii=False).encode()
        ) > min(self.limits.max_prompt_bytes, 12000):
            removable = [
                c for c in context["pool"] if c["candidate_id"] not in protected
            ]
            if not removable:
                break  # Never truncate user intent or source-assessment evidence.
            removed = removable[-1]["candidate_id"]
            context["pool"] = [
                c for c in context["pool"] if c["candidate_id"] != removed
            ]
            context["unread_candidate_ids"] = [
                i for i in context["unread_candidate_ids"] if i != removed
            ]
            if navigation:
                observation["navigation_links"] = [
                    c
                    for c in observation["navigation_links"]
                    if c["candidate_id"] != removed
                ]
        if not context["unread_candidate_ids"] and "read_page" in available:
            available.remove("read_page")
        return context

    async def _decide(self, state):
        task_id = state["task_id"]
        row = self.store.get(task_id)
        if self.stopping() or row["status"] != "discovering":
            return {"execute": False}
        agent = row["agent"]
        if agent["pending_action"] is not None:
            if agent["pending_action"]["status"] == "decided":
                return {"execute": True}

            def interrupted(r):
                r["agent"].update(
                    pending_action=None,
                    usage_complete=False,
                    last_observation={"reason": "interrupted_tool_result_unknown"},
                )

            self._change(task_id, interrupted, "agent_tool_interrupted")
            row = self.store.get(task_id)
            agent = row["agent"]
        if agent["model_pending"]:

            def model_interrupted(r):
                r["agent"].update(model_pending=False, usage_complete=False)

            self._change(task_id, model_interrupted, "agent_model_interrupted")
        context = self._context(row)
        size = len(SYSTEM_PROMPT.encode()) + len(
            json.dumps(context, ensure_ascii=False).encode()
        )
        checks = {
            "max_model_calls": agent["model_calls"] >= self.limits.max_model_calls,
            "max_active_seconds": agent["active_seconds"]
            >= self.limits.max_active_seconds,
            "max_prompt_bytes": size > self.limits.max_prompt_bytes,
            "max_input_tokens": max(
                agent["input_tokens"], agent.get("reserved_input_tokens", 0)
            )
            + size
            > self.limits.max_input_tokens,
            "max_output_tokens": max(
                agent["output_tokens"], agent.get("reserved_output_tokens", 0)
            )
            + 800
            > self.limits.max_output_tokens,
            "max_model_cost_usd": max(
                agent["model_cost_usd"], agent.get("reserved_model_cost_usd", 0)
            )
            + (size * 0.3 + 800 * 1.2) / 1_000_000
            > self.limits.max_model_cost_usd,
        }
        exhausted = [name for name, reached in checks.items() if reached]
        if exhausted:
            self._stop(
                task_id,
                "agent_budget_exhausted",
                {"exhausted_limits": exhausted, "prompt_bytes": size},
            )
            return {"execute": False}

        def reserve(r):
            r["agent"]["model_calls"] += 1
            r["agent"]["model_pending"] = True
            r["agent"]["reserved_input_tokens"] = (
                r["agent"].get("reserved_input_tokens", 0) + size
            )
            r["agent"]["reserved_output_tokens"] = (
                r["agent"].get("reserved_output_tokens", 0) + 800
            )
            r["agent"]["reserved_model_cost_usd"] = (
                r["agent"].get("reserved_model_cost_usd", 0)
                + size * 0.3 / 1_000_000
                + 800 * 1.2 / 1_000_000
            )

        self._change(task_id, reserve, "agent_model_started")
        started = time.monotonic()
        try:
            async with asyncio.timeout(
                max(0.1, self.limits.max_active_seconds - agent["active_seconds"])
            ):
                decision = await self.model.decide(context)
            action = Action.model_validate(decision.action)

            def record(r):
                a = r["agent"]
                a["model_pending"] = False
                a["active_seconds"] += time.monotonic() - started
                a["model"] = decision.model
                for field in ("input_tokens", "output_tokens", "cached_input_tokens"):
                    value = getattr(decision, field)
                    if value is None:
                        a["usage_complete"] = False
                    elif type(value) is not int or value < 0:
                        raise ValueError("Invalid provider usage")
                    else:
                        a[field] += value
                        if field in ("input_tokens", "output_tokens"):
                            estimate = size if field == "input_tokens" else 800
                            a["reserved_" + field] += value - estimate
                if decision.cost_usd is None:
                    a["usage_complete"] = False
                else:
                    if not math.isfinite(decision.cost_usd) or decision.cost_usd < 0:
                        raise ValueError("Invalid provider cost")
                    a["model_cost_usd"] += decision.cost_usd
                    a["reserved_model_cost_usd"] += (
                        decision.cost_usd - (size * 0.3 + 800 * 1.2) / 1_000_000
                    )
                a["pending_action"] = {
                    "status": "decided",
                    "action": action.model_dump(),
                }

            self._change(
                task_id,
                record,
                "agent_decision",
                {
                    "action": action.action,
                    "arguments": action.model_dump(
                        exclude={"action", "reason"}, exclude_none=True
                    ),
                    "reason": action.reason,
                    "model": decision.model,
                    "request_id": decision.request_id,
                    "provider_fingerprint": decision.provider_fingerprint,
                    "input_tokens": decision.input_tokens,
                    "output_tokens": decision.output_tokens,
                    "cached_input_tokens": decision.cached_input_tokens,
                    "model_cost_usd": decision.cost_usd,
                },
            )
            return {"execute": True}
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            failure_reason = (
                "model_action_validation_failed"
                if isinstance(error, ValueError)
                else "model_timeout"
                if isinstance(error, TimeoutError)
                else "model_provider_failed"
            )

            def failed(r):
                r.update(status="failed", stop_reason="agent_model_failed")
                r["agent"].update(model_pending=False, usage_complete=False)

            self._change(task_id, failed, "agent_failed", {"reason": failure_reason})
            return {"execute": False}

    async def _tool(self, state):
        if self.stopping():
            return {"execute": False}
        task_id = state["task_id"]
        row = self.store.get(task_id)
        action = Action.model_validate(row["agent"]["pending_action"]["action"])
        blocked_search = (
            _search_block(row, action.query)
            if action.action == "search_documents"
            else None
        )
        blocked_read = action.action == "read_page" and (
            action.candidate_id in _failed_pages(row)
            or any(c["candidate_id"] == action.candidate_id for c in row["candidates"])
        )
        for tool, counter, limit in (
            ("search_documents", "searches", self.limits.max_searches),
            ("read_page", "reads", self.limits.max_reads),
        ):
            if action.action == tool and row["agent"][counter] >= limit:
                self._stop(
                    task_id,
                    "search_budget_exhausted"
                    if counter == "searches"
                    else "read_budget_exhausted",
                )
                return {"execute": False}

        def reserve(r):
            a = r["agent"]
            a["pending_action"]["status"] = "executing"
            if action.action == "search_documents" and not blocked_search:
                if a["searches"] >= self.limits.max_searches:
                    raise ValueError("Search budget exhausted")
                a["searches"] += 1
            if action.action == "read_page" and not blocked_read:
                if a["reads"] >= self.limits.max_reads:
                    raise ValueError("Read budget exhausted")
                a["reads"] += 1

        started = time.monotonic()
        try:
            self._change(
                task_id, reserve, "agent_tool_started", {"action": action.action}
            )
            async with asyncio.timeout(
                max(
                    0.1, self.limits.max_active_seconds - row["agent"]["active_seconds"]
                )
            ):
                observation = await self._execute(task_id, action)
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            if isinstance(error, Conflict):
                raise
            observation = {"reason": "tool_failed", "error_type": type(error).__name__}
            if isinstance(error, FetchError) and action.action == "read_page":
                code = error.reason_code
                observation.update(
                    reason=code,
                    candidate_id=action.candidate_id,
                    retryable_in_same_revision=False,
                    network_attempts=error.attempts,
                )

                def page_failed(r):
                    r["agent"].setdefault("failed_pages", {})[action.candidate_id] = {
                        "reason": code,
                        "revision": r["revision"],
                    }
                    if code == "https_download_failed":
                        pool_entry = next(
                            (
                                candidate
                                for candidate in r["agent"]["pool"]
                                if candidate["candidate_id"] == action.candidate_id
                            ),
                            None,
                        )
                        host = (
                            urlsplit(pool_entry["url"]).hostname if pool_entry else None
                        )
                        failures = r["agent"].setdefault("host_network_failures", {})
                        previous = failures.get(host, {})
                        count = (
                            previous.get("count", 0) + 1
                            if previous.get("revision") == r["revision"]
                            else 1
                        )
                        failures[host] = {"count": count, "revision": r["revision"]}
                        if count >= 2:
                            r.update(
                                status="awaiting_input",
                                stop_reason=code,
                                pending_question="同一网站连续连接失败，已暂停搜索以避免继续消耗。下载器已进行有限重试和公网 IP 切换；网络恢复后可重新提交搜索，或补充可访问的官方网址。",
                            )
                    if code in ("dns_non_public", "dns_resolution_failed"):
                        r.update(
                            status="awaiting_input",
                            stop_reason=code,
                            pending_question="文档 DNS 解析失败或返回非公网地址，已暂停以避免重复消耗。请检查应用级 DoH 与代理配置；修复后提交补充信息继续当前任务。",
                        )

                self._change(task_id, page_failed, "agent_page_failed", observation)
            if isinstance(error, ProviderError) and action.action == "search_documents":
                code = str(error)
                if code not in SEARCH_FAILURE_MESSAGES:
                    code = "search_provider_unavailable"
                observation["reason"] = code
                self._change(
                    task_id,
                    lambda r: r.update(
                        status="awaiting_input",
                        stop_reason=code,
                        pending_question=SEARCH_FAILURE_MESSAGES[code]
                        + " 已暂停自动搜索，避免重复消耗。修复服务后可继续，或补充文档网址进行核查。",
                    ),
                    "agent_search_unavailable",
                    {"reason": code},
                )

        def finish(r):
            r["agent"].update(last_observation=observation, pending_action=None)
            r["agent"]["active_seconds"] += time.monotonic() - started
            history = r["agent"].setdefault("recent_actions", [])
            history.append(
                {
                    "action": action.action,
                    "candidate_id": action.candidate_id,
                    "reason": observation.get("reason", "completed"),
                }
            )
            r["agent"]["recent_actions"] = history[-6:]

        # Finalizing actions also persist their observation but no longer loop.
        current = self.store.get(task_id)
        if current["status"] == "discovering":
            self._change(
                task_id,
                finish,
                "agent_tool_completed",
                {"action": action.action, "observation": observation},
            )
        elif current["status"] != "cancelled":
            self.store.mutate(
                task_id,
                finish,
                "agent_tool_completed",
                {"action": action.action, "observation": observation},
            )
        return {"execute": False}

    def _pool_add(self, task_id, hits):
        def add(row):
            pool = row["agent"]["pool"]
            for hit in hits:
                url = validate_url(hit.url)
                if urlsplit(url).path.lower().endswith((".pdf", ".zip", ".gz", ".tar")):
                    continue
                if len(pool) >= 30:
                    break
                if any(c["url"] == url for c in pool):
                    continue
                pool.append(
                    {
                        "candidate_id": "candidate-"
                        + hashlib.sha256(url.encode()).hexdigest(),
                        "url": url,
                        "title": hit.title[:200],
                        "snippet": hit.snippet[:500],
                    }
                )

        return self._change(task_id, add, "agent_candidates_found")

    async def _execute(self, task_id, action):
        row = self.store.get(task_id)
        if action.action == "search_documents":
            if not action.query:
                raise ValueError("Search query required")
            blocked = _search_block(row, action.query)
            if blocked:
                return {
                    "reason": blocked,
                    "next_step": "Read an existing unread candidate, inspect links, or ask a clarification; do not repeat this query.",
                }
            previous_ids = {c["candidate_id"] for c in row["agent"]["pool"]}
            hits = await self.search.search(action.query, 5)
            row = self._pool_add(task_id, hits)
            found_ids = [
                c["candidate_id"]
                for c in row["agent"]["pool"]
                if c["url"] in {h.url for h in hits}
            ]
            new_count = len(set(found_ids) - previous_ids)
            self._change(
                task_id,
                lambda r: (
                    r["agent"]
                    .setdefault("search_history", [])
                    .append(
                        {
                            "query": action.query,
                            "normalized_query": " ".join(
                                action.query.casefold().split()
                            ),
                            "revision": r["revision"],
                            "new_candidate_count": new_count,
                        }
                    )
                ),
                "agent_search_recorded",
                {"new_candidate_count": new_count},
            )
            return {
                "reason": "search_completed"
                if new_count
                else "search_no_new_candidates",
                "results": [
                    c for c in row["agent"]["pool"] if c["url"] in {h.url for h in hits}
                ],
            }
        if action.action in ("read_page", "inspect_links"):
            if action.action == "read_page" and any(
                c["candidate_id"] == action.candidate_id for c in row["candidates"]
            ):
                return {
                    "reason": "page_already_read",
                    "candidate_id": action.candidate_id,
                    "next_step": "Choose a different unread_candidate_id. Real navigation links already have candidate IDs in the pool.",
                }
            if action.action == "read_page" and action.candidate_id in _failed_pages(
                row
            ):
                return {
                    "reason": "page_failed_previously",
                    "candidate_id": action.candidate_id,
                    "failure_reason": _failed_pages(row)[action.candidate_id]["reason"],
                    "next_step": "Read another candidate or clarify; do not retry this page in the same revision.",
                }
            candidate = next(
                (
                    c
                    for c in row["agent"]["pool"]
                    if c["candidate_id"] == action.candidate_id
                ),
                None,
            )
            if candidate is None:
                raise ValueError("Unknown candidate")
            domains = self.sites.get(row["product"].casefold(), [])
            official = urlsplit(candidate["url"]).hostname in domains
            try:
                snapshot = self.store.snapshot(task_id, candidate["candidate_id"])
            except LookupError:
                if action.action == "inspect_links":
                    return {"reason": "read_page_first"}
                snapshot = await asyncio.to_thread(self.fetcher.fetch, candidate["url"])
                self.store.save_snapshot(task_id, candidate["candidate_id"], snapshot)
            page = await asyncio.to_thread(
                extract_html, snapshot.body, snapshot.final_url
            )
            official = official and urlsplit(snapshot.final_url).hostname in domains
            from docguard.discovery.sources import discovery_links, source_links

            links = source_links(snapshot)
            navigation = discovery_links(snapshot)
            navigation.sort(
                key=lambda link: (
                    document_kind(link["url"], link["title"])
                    != row.get("document_type", "user_guide")
                )
            )
            navigation = navigation[:10]
            from docguard.discovery.providers import SearchHit

            # Make observed links directly actionable, without another paid
            # model turn merely to register URLs via inspect_links.
            updated = self._pool_add(
                task_id,
                [
                    SearchHit(
                        link["title"] or link["url"],
                        link["url"],
                        "Discovered in inspected page",
                    )
                    for link in navigation
                ],
            )
            pool_ids = {c["url"]: c["candidate_id"] for c in updated["agent"]["pool"]}
            navigation = [
                {**link, "candidate_id": pool_ids[link["url"]]}
                for link in navigation
                if link["url"] in pool_ids
            ]
            if action.action == "inspect_links":
                self._change(
                    task_id,
                    lambda r: (
                        r["agent"]
                        .setdefault("links_inspected", [])
                        .append(candidate["candidate_id"])
                    ),
                    "agent_links_inspected",
                    {"candidate_id": candidate["candidate_id"]},
                )
                return {
                    "links": [link["url"] for link in navigation],
                    "navigation_links": navigation,
                }
            status = _version_status(snapshot.final_url, page.title, row["version"])
            if status == "version_mismatch":
                return {"reason": "version_mismatch", "url": snapshot.final_url}

            def inspected(r):
                r["agent"].setdefault("host_network_failures", {}).pop(
                    urlsplit(snapshot.final_url).hostname, None
                )
                r["agent"].setdefault("page_links", {})[candidate["candidate_id"]] = (
                    links
                )
                value = {
                    "candidate_id": candidate["candidate_id"],
                    "requested_url": candidate["url"],
                    "url": snapshot.final_url,
                    "title": page.title,
                    "content_sha256": snapshot.content_sha256,
                    "version": row["version"],
                    "version_status": status,
                    "source_status": "official_domain" if official else "unverified",
                    "status": "inspected",
                }
                r["candidates"] = [
                    c
                    for c in r["candidates"]
                    if c["candidate_id"] != value["candidate_id"]
                ] + [value]

            self._change(
                task_id,
                inspected,
                "candidate_inspected",
                {"candidate_id": candidate["candidate_id"]},
            )
            return {
                "candidate_id": candidate["candidate_id"],
                "title": page.title,
                "version_status": status,
                "source_status": "official_domain" if official else "unverified",
                "links": [link["url"] for link in navigation],
                "navigation_links": navigation,
                "excerpt": "\n".join(b.text for b in page.blocks)[:1600],
            }
        if action.action == "assess_source":
            from docguard.discovery.sources import source_assessment

            inspected = {c["candidate_id"]: c for c in row["candidates"]}
            ids = list(dict.fromkeys(action.candidate_ids))
            if action.candidate_id not in inspected or any(
                i not in inspected for i in ids
            ):
                return {"reason": "source_assessment_requires_inspected_pages"}
            target = inspected[action.candidate_id]
            kind = document_kind(target["url"], target["title"])
            wanted = row.get("document_type", "user_guide")
            if wanted != "topic" and (
                (kind is not None and kind != wanted)
                or ("document_type" in row and scope_for_entry(target["url"]) is None)
            ):
                return {
                    "reason": "document_type_mismatch",
                    "requested_type": wanted,
                    "instruction": "Read the matching collection entry from actual navigation links",
                }
            target["collection"] = scope_for_entry(target["url"])
            try:
                proof = source_assessment(
                    self.store,
                    task_id,
                    inspected[action.candidate_id],
                    [inspected[i] for i in ids],
                    action.reason,
                )
                proof["model"] = row["agent"].get("model")
            except (ValueError, LookupError):
                return {"reason": "source_assessment_missing_link_evidence"}

            def assessed(r):
                for candidate in r["candidates"]:
                    if candidate["candidate_id"] == action.candidate_id:
                        candidate["collection"] = scope_for_entry(candidate["url"])
                        candidate["source_status"] = "agent_assessed_official"
                        candidate["source_assessment"] = proof
                        if r["version"] is None or candidate["version_status"] in (
                            "url_version_match",
                            "not_requested",
                        ):
                            # The assessment already nominates this document;
                            # handing it to the user needs no extra paid decision.
                            r.update(
                                status="awaiting_selection",
                                candidates=[candidate],
                                pending_question=None,
                                stop_reason=None,
                            )
                        break

            self._change(
                task_id,
                assessed,
                "agent_source_assessed",
                {"candidate_id": action.candidate_id, "assessment": proof},
            )
            return {
                "reason": "source_assessed_pending_user_confirmation",
                "candidate_id": action.candidate_id,
                "assessment": proof,
            }
        if action.action == "ask_user":
            if not action.question:
                raise ValueError("Clarification question required")
            self._change(
                task_id,
                lambda r: r.update(
                    status="awaiting_input", pending_question=action.question
                ),
                "agent_awaiting_input",
            )
            return {"reason": "awaiting_user"}
        if action.action == "present_candidates":
            mapping = {c["candidate_id"]: c for c in row["candidates"]}
            if not action.candidate_ids or any(
                c not in mapping for c in action.candidate_ids
            ):
                return {"reason": "uninspected_candidate"}
            selected = [mapping[c] for c in dict.fromkeys(action.candidate_ids)]
            wanted = row.get("document_type", "user_guide")
            if wanted != "topic" and any(
                document_kind(c["url"], c["title"]) not in (None, wanted)
                or ("document_type" in row and scope_for_entry(c["url"]) is None)
                for c in selected
            ):
                return {"reason": "document_type_mismatch", "requested_type": wanted}
            for candidate in selected:
                candidate["collection"] = scope_for_entry(candidate["url"])
            if row.get("official_only", True) and any(
                c["source_status"] not in ("official_domain", "agent_assessed_official")
                for c in selected
            ):
                return {
                    "reason": "source_not_assessed_read_supporting_page_then_assess_source"
                }
            if row["version"] is not None and any(
                c["version_status"] == "unknown" for c in selected
            ):
                return {"reason": "version_not_confirmed_ask_user"}
            self._change(
                task_id,
                lambda r: r.update(
                    status="awaiting_selection",
                    candidates=selected,
                    pending_question=None,
                ),
                "agent_candidates_ready",
            )
            return {"reason": "candidates_ready"}
        self._change(
            task_id,
            lambda r: r.update(status="failed", stop_reason="no_verified_candidates"),
            "agent_finished",
        )
        return row["agent"]["last_observation"]
