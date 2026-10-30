import hashlib

import pytest

from docguard.discovery.fetch import Snapshot
from docguard.discovery.graph import AgentLimits, DiscoveryAgent
from docguard.discovery.providers import Decision, SearchHit
from docguard.discovery.store import TaskStore

URL = "https://docs.python.org/3/library/configparser.html"
BODY = b'<title>Python ConfigParser</title><main><h1 id="config">ConfigParser</h1><p>Reads INI files.</p><a href="other.html">Other docs</a></main>'


class ScriptModel:
    def __init__(self, actions):
        self.actions, self.observations = iter(actions), []

    async def decide(self, context):
        self.observations.append(context)
        action = next(self.actions)
        if callable(action):
            action = action(context)
        return Decision(
            action=action, model="fixture", input_tokens=10, output_tokens=5
        )


class Search:
    def __init__(self, batches):
        self.batches, self.queries = iter(batches), []

    async def search(self, query, limit=5):
        self.queries.append(query)
        return next(self.batches)


class Fetcher:
    def __init__(self):
        self.urls = []

    def fetch(self, url):
        self.urls.append(url)
        return Snapshot(
            url,
            url,
            "2026-10-27T00:00:00Z",
            hashlib.sha256(BODY).hexdigest(),
            BODY,
            "https://docs.python.org/robots.txt",
            404,
            hashlib.sha256(b"").hexdigest(),
        )


def read_first(context):
    return {
        "action": "read_page",
        "candidate_id": context["pool"][0]["candidate_id"],
        "reason": "Inspect the actual page",
    }


def present(context):
    return {
        "action": "present_candidates",
        "candidate_ids": [context["candidates"][0]["candidate_id"]],
        "reason": "The inspected page is relevant",
    }


def task(store, version="3"):
    return store.create(
        {
            "goal": "Find Python INI configuration docs",
            "product": "Python",
            "version": version,
            "urls": [],
            "official_only": True,
        }
    )


@pytest.mark.asyncio
async def test_successful_model_usage_releases_conservative_reservations(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    store.mutate(
        row["task_id"],
        lambda r: r.update(goal="Find Python docs " + "文档" * 900),
        "long_goal",
    )
    model = ScriptModel([{"action": "ask_user", "question": "Which topic?"}] * 5)
    agent = DiscoveryAgent(store, model, Search([]), Fetcher())
    for _ in range(5):
        await agent.run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_input"
    assert result["agent"]["model_calls"] == 5
    assert result["agent"]["reserved_input_tokens"] == 50
    assert result["agent"]["reserved_output_tokens"] == 25


@pytest.mark.asyncio
async def test_dns_failure_pauses_with_diagnostic_without_retry(tmp_path):
    from docguard.discovery.fetch import FetchError

    class BrokenFetcher:
        def fetch(self, url):
            raise FetchError(
                "do not expose raw provider details", reason_code="dns_non_public"
            )

    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel([{"action": "search_documents", "query": "Python"}, read_first])
    row = task(store)
    await DiscoveryAgent(
        store, model, Search([[SearchHit("Docs", URL, "")]]), BrokenFetcher()
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_input"
    assert result["stop_reason"] == "dns_non_public"
    assert result["agent"]["reads"] == 1
    assert result["agent"]["last_observation"]["reason"] == "dns_non_public"
    assert "do not expose" not in str(result)


@pytest.mark.asyncio
async def test_budget_stop_identifies_prompt_limit_without_calling_the_model(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    await DiscoveryAgent(
        store,
        ScriptModel([]),
        Search([]),
        Fetcher(),
        limits=AgentLimits(max_prompt_bytes=3000),
    ).run(row["task_id"])
    stopped = next(
        e["data"] for e in store.events(row["task_id"]) if e["kind"] == "agent_stopped"
    )
    assert stopped["exhausted_limits"] == ["max_prompt_bytes"]
    assert stopped["prompt_bytes"] > 3000
    assert stopped["usage"]["model_calls"] == 0


@pytest.mark.asyncio
async def test_large_navigation_context_is_compact_without_losing_assessment_evidence(
    tmp_path,
):
    import json

    from docguard.discovery.graph import initial_agent_state
    from docguard.discovery.providers import SYSTEM_PROMPT

    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    pool = [
        {
            "candidate_id": f"candidate-{i:064x}",
            "url": f"https://docs.example/guide/{i}.html",
            "title": "Documentation " * 10,
            "snippet": "Search summary " * 35,
        }
        for i in range(30)
    ]
    read = [
        {**c, "source_status": "unverified", "version_status": "not_requested"}
        for c in pool[:3]
    ]
    agent = initial_agent_state()
    agent.update(
        pool=pool,
        page_links={read[0]["candidate_id"]: [read[1]["url"]]},
        last_observation={
            "candidate_id": read[2]["candidate_id"],
            "title": "User Guide",
            "excerpt": "Guide text " * 150,
            "links": [c["url"] for c in pool],
            "navigation_links": pool,
        },
    )
    row.update(agent=agent, candidates=read)
    context = DiscoveryAgent(store, None, None, None)._context(row)
    size = len(SYSTEM_PROMPT.encode()) + len(
        json.dumps(context, ensure_ascii=False).encode()
    )
    assert size <= 20000
    assert (
        context["source_assessment_options"][0]["candidate_id"]
        == read[1]["candidate_id"]
    )
    assert context["candidates"] == read
    assert set(context["unread_candidate_ids"]) <= {
        c["candidate_id"] for c in context["pool"]
    }
    assert len(row["agent"]["pool"]) == 30


@pytest.mark.asyncio
async def test_read_registers_real_navigation_without_inspect_step_and_blocks_repeat(
    tmp_path,
):
    store = TaskStore(tmp_path / "state.sqlite3")

    def after_read(context):
        other = next(c for c in context["pool"] if c["url"].endswith("other.html"))
        assert other["candidate_id"] in context["unread_candidate_ids"]
        assert (
            context["last_observation"]["navigation_links"][0]["candidate_id"]
            == other["candidate_id"]
        )
        return read_first(context)

    def after_repeat(context):
        assert context["last_observation"]["reason"] == "page_already_read"
        return {"action": "ask_user", "question": "Which topic?"}

    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python"},
            read_first,
            after_read,
            after_repeat,
        ]
    )
    fetcher = Fetcher()
    row = task(store)
    await DiscoveryAgent(
        store, model, Search([[SearchHit("Docs", URL, "")]]), fetcher
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_input"
    assert result["pending_question"] == "Which topic?"
    assert result["agent"]["reads"] == 1
    assert fetcher.urls == [URL]


@pytest.mark.asyncio
async def test_failed_page_is_not_fetched_again_and_other_pages_remain_available(
    tmp_path,
):
    from docguard.discovery.fetch import FetchError

    class BrokenFetcher:
        def __init__(self):
            self.urls = []

        def fetch(self, url):
            self.urls.append(url)
            raise FetchError("blocked", reason_code="robots_disallowed")

    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python"},
            read_first,
            read_first,
            {"action": "ask_user", "question": "Try another topic?"},
        ]
    )
    row = task(store)
    fetcher = BrokenFetcher()
    await DiscoveryAgent(
        store, model, Search([[SearchHit("Docs", URL, "")]]), fetcher
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert fetcher.urls == [URL]
    assert result["agent"]["reads"] == 1
    assert model.observations[2]["unread_candidate_ids"] == []
    assert model.observations[2]["failed_pages"][0]["reason"] == "robots_disallowed"


@pytest.mark.asyncio
async def test_duplicate_search_query_uses_existing_results_without_another_call(
    tmp_path,
):
    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python docs"},
            {"action": "search_documents", "query": " python   DOCS "},
            {"action": "ask_user", "question": "Which version?"},
        ]
    )
    search = Search([[]])
    row = task(store)
    await DiscoveryAgent(store, model, search, Fetcher()).run(row["task_id"])
    result = store.get(row["task_id"])
    assert search.queries == ["Python docs"]
    assert result["agent"]["searches"] == 1
    assert (
        model.observations[2]["last_observation"]["reason"] == "duplicate_search_query"
    )


@pytest.mark.asyncio
async def test_search_provider_failure_pauses_without_model_retry(tmp_path):
    from docguard.discovery.providers import ProviderError

    class BrokenSearch:
        async def search(self, query, limit=5):
            raise ProviderError("search_authentication_failed")

    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel([{"action": "search_documents", "query": "Java docs"}])
    row = task(store)
    await DiscoveryAgent(store, model, BrokenSearch(), Fetcher()).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_input"
    assert result["stop_reason"] == "search_authentication_failed"
    assert result["agent"]["model_calls"] == 1
    assert result["agent"]["searches"] == 1
    assert "密钥" in result["pending_question"]


@pytest.mark.asyncio
async def test_repeated_result_set_blocks_further_search_until_candidates_are_read(
    tmp_path,
):
    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": query}
            for query in ("first", "refined", "again")
        ]
        + [{"action": "ask_user", "question": "Which file format?"}]
    )
    hits = [SearchHit("Docs", URL, "")]
    search = Search([hits, hits])
    row = task(store)
    await DiscoveryAgent(store, model, search, Fetcher()).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["agent"]["searches"] == 2
    assert search.queries == ["first", "refined"]
    assert "search_documents" not in model.observations[2]["available_actions"]
    assert "read_page" in model.observations[2]["available_actions"]


@pytest.mark.asyncio
async def test_unknown_model_usage_keeps_reservation_after_success(tmp_path):
    class UnknownUsage:
        async def decide(self, context):
            return Decision(
                {"action": "ask_user", "question": "Which topic?"}, "fixture"
            )

    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    await DiscoveryAgent(store, UnknownUsage(), Search([]), Fetcher()).run(
        row["task_id"]
    )
    a = store.get(row["task_id"])["agent"]
    assert a["usage_complete"] is False
    assert a["reserved_input_tokens"] > 0
    assert a["reserved_output_tokens"] == 800


def test_search_stall_from_previous_revision_does_not_block_user_clarification():
    from docguard.discovery.graph import _search_block

    row = {
        "revision": 2,
        "candidates": [],
        "agent": {
            "pool": [{"candidate_id": "candidate-old"}],
            "search_history": [
                {
                    "query": "old",
                    "normalized_query": "old",
                    "new_candidate_count": 0,
                    "revision": 1,
                }
            ],
        },
    }
    assert _search_block(row, "new clarified topic") is None


@pytest.mark.asyncio
async def test_search_limit_does_not_loop_over_rejected_calls(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "first"},
            {"action": "search_documents", "query": "again"},
        ]
    )
    row = task(store)
    await DiscoveryAgent(
        store, model, Search([[]]), Fetcher(), limits=AgentLimits(max_searches=1)
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["stop_reason"] == "search_budget_exhausted"
    assert result["agent"]["model_calls"] == 2
    assert model.observations[1]["remaining_searches"] == 0
    stopped = next(
        e["data"] for e in store.events(row["task_id"]) if e["kind"] == "agent_stopped"
    )
    assert stopped["usage"]["searches"] == stopped["limits"]["max_searches"] == 1


@pytest.mark.asyncio
async def test_java_official_page_can_be_inspected(tmp_path):
    url = "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/Properties.html"
    store = TaskStore(tmp_path / "state.sqlite3")
    row = store.create(
        {
            "goal": "Java Properties docs",
            "product": "Java",
            "version": "21",
            "urls": [],
            "official_only": True,
        }
    )
    fetcher = Fetcher()
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Java 21 Properties official"},
            read_first,
            present,
        ]
    )
    await DiscoveryAgent(
        store, model, Search([[SearchHit("Properties", url, "Java docs")]]), fetcher
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_selection"
    assert result["candidates"][0]["source_status"] == "official_domain"


@pytest.mark.asyncio
async def test_unknown_product_can_discover_and_assess_source_without_user_url(
    tmp_path,
):
    home, doc = "https://example.org/", "https://example.org/2/guide.html"

    class Pages:
        def fetch(self, url):
            body = (
                f'<title>Widget project</title><main><p>Widget official documentation.</p><a href="{doc}">Documentation</a></main>'
            ).encode()
            return Snapshot(
                url,
                url,
                "2026-10-28T00:00:00Z",
                hashlib.sha256(body).hexdigest(),
                body,
                home + "robots.txt",
                404,
                hashlib.sha256(b"").hexdigest(),
            )

    def assess(context):
        candidates = {c["url"]: c["candidate_id"] for c in context["candidates"]}
        return {
            "action": "assess_source",
            "candidate_id": candidates[doc],
            "candidate_ids": [candidates[home]],
            "reason": "Widget project homepage links directly to the requested Widget documentation.",
        }

    def present_doc(context):
        return {
            "action": "present_candidates",
            "candidate_ids": [
                next(
                    c["candidate_id"] for c in context["candidates"] if c["url"] == doc
                )
            ],
        }

    store = TaskStore(tmp_path / "state.sqlite3")
    row = store.create(
        {
            "goal": "Find Widget 2 configuration docs",
            "product": "unspecified",
            "version": "2",
            "urls": [],
            "official_only": True,
        }
    )
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Widget official documentation"},
            lambda c: {
                "action": "read_page",
                "candidate_id": c["pool"][0]["candidate_id"],
            },
            lambda c: {
                "action": "read_page",
                "candidate_id": c["pool"][1]["candidate_id"],
            },
            assess,
            present_doc,
        ]
    )
    await DiscoveryAgent(
        store,
        model,
        Search([[SearchHit("Widget", home, "")]]),
        Pages(),
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_selection"
    assert result["candidates"][0]["source_status"] == "agent_assessed_official"
    proof = result["candidates"][0]["source_assessment"]
    assert proof["requires_user_confirmation"] is True
    assert proof["authority_pages"][0]["url"] == home
    assert proof["authority_pages"][0]["content_sha256"]
    assert "present_candidates" not in model.observations[3]["available_actions"]
    assert "assess_source" in model.observations[3]["available_actions"]
    assert len(model.observations) == 4
    assert result.get("stop_reason") is None
    assert model.observations[3]["recent_actions"][-1]["action"] == "read_page"


@pytest.mark.asyncio
async def test_unassessed_unknown_source_cannot_escape_via_budget_stop(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    model = ScriptModel(
        [{"action": "search_documents", "query": "Python docs"}, read_first]
    )
    await DiscoveryAgent(
        store,
        model,
        Search([[SearchHit("Community", "https://example.org/3/docs", "")]]),
        Fetcher(),
        limits=AgentLimits(max_model_calls=2),
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "exhausted"
    assert not result["candidates"]
    assert result["agent"]["reads"] == 1


@pytest.mark.asyncio
async def test_search_rewrite_read_then_present_uses_observations(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python INI"},
            {"action": "search_documents", "query": "Python ConfigParser official"},
            read_first,
            present,
        ]
    )
    search = Search(
        [[], [SearchHit("ConfigParser", URL, "configuration documentation")]]
    )
    fetcher = Fetcher()
    agent = DiscoveryAgent(store, model, search, fetcher)
    row = task(store)
    await agent.run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_selection"
    assert search.queries == ["Python INI", "Python ConfigParser official"]
    assert fetcher.urls == [URL]
    assert result["candidates"][0]["source_status"] == "official_domain"
    assert result["candidates"][0]["version_status"] == "url_version_match"
    assert model.observations[1]["last_observation"]["results"] == []
    assert result["agent"]["model_calls"] == 4
    assert result["agent"]["input_tokens"] == 40


@pytest.mark.asyncio
async def test_budget_stops_loop_and_never_creates_import_job(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "first"},
            {"action": "search_documents", "query": "second"},
        ]
    )
    search = Search([[], []])
    row = task(store)
    await DiscoveryAgent(
        store, model, search, Fetcher(), limits=AgentLimits(max_model_calls=2)
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "exhausted"
    assert result["job"] is None
    assert result["agent"]["model_calls"] == 2


@pytest.mark.asyncio
async def test_search_summary_cannot_be_presented_as_read_evidence(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python docs"},
            lambda c: {
                "action": "present_candidates",
                "candidate_ids": [c["pool"][0]["candidate_id"]],
            },
            {"action": "finish", "reason": "No inspected evidence"},
        ]
    )
    row = task(store)
    await DiscoveryAgent(
        store,
        model,
        Search([[SearchHit("Docs", URL, "a fabricated answer")]]),
        Fetcher(),
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["candidates"] == []
    assert result["status"] == "failed"
    assert result["agent"]["last_observation"]["reason"] == "uninspected_candidate"


@pytest.mark.asyncio
async def test_wrong_version_is_excluded_and_agent_can_ask_user(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    wrong = "https://docs.python.org/2/library/configparser.html"
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python 3 docs"},
            read_first,
            {
                "action": "ask_user",
                "question": "Which documentation version do you need?",
            },
        ]
    )
    row = task(store)
    await DiscoveryAgent(
        store, model, Search([[SearchHit("Python 2", wrong, "old")]]), Fetcher()
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_input"
    assert result["candidates"] == []
    assert model.observations[2]["last_observation"]["reason"] == "version_mismatch"


@pytest.mark.asyncio
async def test_unavailable_providers_pause_without_fake_results(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    await DiscoveryAgent(store, None, None, Fetcher()).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "awaiting_input"
    assert result["stop_reason"] == "agent_dependencies_unavailable"
    assert not result["candidates"]


@pytest.mark.asyncio
async def test_cancelled_task_is_not_resurrected_by_scheduled_agent(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    store.mutate(
        row["task_id"], lambda r: r.update(status="cancelled"), "task_cancelled"
    )
    await DiscoveryAgent(store, ScriptModel([]), Search([]), Fetcher()).run(
        row["task_id"]
    )
    assert store.get(row["task_id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_interrupted_search_is_not_replayed_on_restart(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    await DiscoveryAgent(
        store,
        ScriptModel([{"action": "ask_user", "question": "Which version?"}]),
        Search([]),
        Fetcher(),
    ).run(row["task_id"])

    def interrupted(r):
        r["status"] = "discovering"
        r["agent"]["pending_action"] = {
            "status": "executing",
            "action": {"action": "search_documents", "query": "already sent"},
        }

    store.mutate(row["task_id"], interrupted, "interrupted")
    search = Search([])
    model = ScriptModel(
        [{"action": "finish", "reason": "Search result was interrupted"}]
    )
    await DiscoveryAgent(store, model, search, Fetcher()).run(row["task_id"])
    assert not search.queries
    assert not store.get(row["task_id"])["agent"]["usage_complete"]
    assert (
        model.observations[0]["last_observation"]["reason"]
        == "interrupted_tool_result_unknown"
    )


@pytest.mark.asyncio
async def test_budget_does_not_present_unknown_version_as_verified(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store, version="3.11")
    model = ScriptModel(
        [{"action": "search_documents", "query": "Python 3.11 config"}, read_first]
    )
    await DiscoveryAgent(
        store,
        model,
        Search([[SearchHit("Python docs", URL, "major version only")]]),
        Fetcher(),
        limits=AgentLimits(max_model_calls=2),
    ).run(row["task_id"])
    result = store.get(row["task_id"])
    assert result["status"] == "exhausted"
    assert not result["candidates"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_reads": 0},
        {"max_model_calls": 17},
        {"max_active_seconds": float("nan")},
        {"max_input_tokens": True},
    ],
)
def test_agent_resource_limits_are_validated(kwargs):
    with pytest.raises(ValueError):
        AgentLimits(**kwargs)


@pytest.mark.asyncio
async def test_requested_guide_does_not_accept_getting_started_even_at_budget_stop(
    tmp_path,
):
    from docguard.discovery.graph import Action, initial_agent_state

    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store, version=None)
    candidate = {
        "candidate_id": "intro",
        "url": "https://docs.example/getting_started/index.html",
        "title": "Getting started",
        "source_status": "official_domain",
        "version_status": "not_requested",
    }
    store.mutate(
        row["task_id"],
        lambda r: r.update(
            status="discovering",
            document_type="user_guide",
            candidates=[candidate],
            agent=initial_agent_state(),
        ),
        "fixture",
    )
    agent = DiscoveryAgent(store, ScriptModel([]), Search([]), Fetcher())
    result = await agent._execute(
        row["task_id"], Action(action="present_candidates", candidate_ids=["intro"])
    )
    assert result["reason"] == "document_type_mismatch"
    agent._stop(row["task_id"], "agent_budget_exhausted")
    assert store.get(row["task_id"])["candidates"] == []


@pytest.mark.asyncio
async def test_same_host_network_failures_pause_before_spending_remaining_model_budget(
    tmp_path,
):
    from docguard.discovery.fetch import FetchError

    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)

    class FailedFetcher:
        def fetch(self, url):
            raise FetchError(
                "HTTPS download failed", reason_code="https_download_failed"
            )

    def read_second(context):
        return {
            "action": "read_page",
            "candidate_id": context["pool"][1]["candidate_id"],
        }

    model = ScriptModel(
        [
            {"action": "search_documents", "query": "Python docs"},
            read_first,
            read_second,
        ]
    )
    await DiscoveryAgent(
        store,
        model,
        Search(
            [
                [
                    SearchHit("First", URL, ""),
                    SearchHit(
                        "Second", "https://docs.python.org/3/library/json.html", ""
                    ),
                ]
            ]
        ),
        FailedFetcher(),
    ).run(row["task_id"])
    final = store.get(row["task_id"])
    assert final["status"] == "awaiting_input"
    assert final["stop_reason"] == "https_download_failed"
    assert final["agent"]["model_calls"] == 3


@pytest.mark.asyncio
async def test_html_discovery_does_not_offer_pdf_as_readable_candidate(tmp_path):
    store = TaskStore(tmp_path / "state.sqlite3")
    row = task(store)
    model = ScriptModel(
        [
            {"action": "search_documents", "query": "docs"},
            {"action": "ask_user", "question": "Which guide?"},
        ]
    )
    await DiscoveryAgent(
        store,
        model,
        Search(
            [
                [
                    SearchHit("PDF guide", "https://docs.example/guide.pdf", ""),
                    SearchHit(
                        "HTML guide", "https://docs.example/guide/index.html", ""
                    ),
                ]
            ]
        ),
        Fetcher(),
    ).run(row["task_id"])
    assert [candidate["url"] for candidate in model.observations[1]["pool"]] == [
        "https://docs.example/guide/index.html"
    ]
