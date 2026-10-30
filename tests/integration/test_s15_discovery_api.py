import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docguard.api.discovery import discovery_router
from docguard.discovery.fetch import Snapshot
from docguard.discovery.importer import DiscoveryService
from docguard.knowledge.pipeline import KnowledgeBase
from docguard.processing.config import load_chunking_config
from docguard.processing.token_budget import TokenBudget

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("goal", [None, "", "   "])
def test_product_only_request_derives_official_document_goal(goal):
    from docguard.discovery.models import CreateTask

    values = {"product": " pandas ", "version": " 2.3 "}
    if goal is not None:
        values["goal"] = goal
    request = CreateTask(**values)
    assert request.product == "pandas"
    assert request.goal == "查找 pandas 2.3 的官方文档"


@pytest.mark.parametrize("product", [None, "", "  ", "unspecified"])
def test_document_task_requires_a_real_product(product):
    from docguard.discovery.models import CreateTask

    values = {"goal": "Find docs"}
    if product is not None:
        values["product"] = product
    with pytest.raises(ValueError):
        CreateTask(**values)


def test_explicit_requirement_is_preserved():
    from docguard.discovery.models import CreateTask

    request = CreateTask(product="Python", goal="  只查找配置解析资料  ")
    assert request.goal == "只查找配置解析资料"


def test_http_product_only_creation_and_missing_product_validation(setup_service):
    service, _, _, _ = setup_service
    app = FastAPI()
    app.include_router(discovery_router(service), prefix="/api/v1")
    with TestClient(app) as client:
        missing = client.post(
            "/api/v1/discovery/tasks", json={"goal": "Find official docs"}
        )
        assert missing.status_code == 422
        response = client.post("/api/v1/discovery/tasks", json={"product": "pandas"})
        assert response.status_code == 202
        assert response.json()["product"] == "pandas"
        assert response.json()["goal"] == "查找 pandas 的官方文档"


URL = "https://example.com/docs/config.html"
BODY = b'<title>Configuration</title><main><h1 id="config">ConfigParser</h1><p>ConfigParser reads INI configuration files.</p></main>'


class Fetcher:
    def __init__(self):
        self.calls = []

    def fetch(self, url):
        self.calls.append(url)
        return Snapshot(
            url,
            url,
            "2026-10-27T00:00:00Z",
            hashlib.sha256(BODY).hexdigest(),
            BODY,
            "https://example.com/robots.txt",
            404,
            hashlib.sha256(b"").hexdigest(),
        )


@pytest.fixture
def setup_service(tmp_path):
    budget = TokenBudget.from_manifest(ROOT / "data/tokenizers/manifest.json", ROOT)
    config = load_chunking_config(ROOT / "configs/chunking/s2-chunk-v1.json")
    fetcher = Fetcher()

    def factory():
        return KnowledgeBase(tmp_path / "knowledge", budget, config)

    service = DiscoveryService(tmp_path / "tasks", factory, fetcher=fetcher)
    return service, fetcher, factory, tmp_path


async def settled(service, task_id):
    for _ in range(200):
        task = service.get_task(task_id)
        if task["status"] not in ("discovering", "importing"):
            return task
        await asyncio.sleep(0.01)
    pytest.fail("task did not settle")


@pytest.mark.asyncio
async def test_delete_imported_base_hides_it_blocks_query_and_preserves_snapshot(
    setup_service,
):
    service, fetcher, factory, _ = setup_service
    info = await factory().import_urls(
        "delete-me", [URL], product="Python", fetcher=fetcher
    )
    app = FastAPI()
    app.include_router(discovery_router(service), prefix="/api/v1")
    with TestClient(app) as client:
        deleted = client.delete("/api/v1/knowledge-bases/delete-me")
        assert deleted.status_code == 200
        assert deleted.json()["status"] == "deleted"
        assert client.get("/api/v1/knowledge-bases").json()["knowledge_bases"] == []
        assert (
            client.post(
                "/api/v1/knowledge-bases/delete-me/query", json={"question": "INI?"}
            ).status_code
            == 404
        )
        assert client.delete("/api/v1/knowledge-bases/delete-me").status_code == 404
        assert client.delete("/api/v1/knowledge-bases/POSTGRESQL").status_code == 422
    assert (
        factory().root / "delete-me" / info["snapshot_id"] / "manifest.json"
    ).is_file()
    restored = await factory().import_urls(
        "delete-me", [URL], product="Python", fetcher=fetcher
    )
    assert restored["snapshot_id"] == info["snapshot_id"]
    assert service.list_knowledge_bases()[0]["knowledge_base_id"] == "delete-me"
    await service.close()


@pytest.mark.asyncio
async def test_delete_refuses_a_base_with_active_import(setup_service):
    from docguard.discovery.store import Conflict

    service, fetcher, factory, _ = setup_service
    await factory().import_urls("busy-base", [URL], product="Python", fetcher=fetcher)
    task = service.store.create(
        {"goal": "Docs", "product": "Python", "version": None, "urls": []}
    )
    service.store.mutate(
        task["task_id"],
        lambda row: row.update(
            job={"knowledge_base_id": "busy-base", "status": "running"}
        ),
        "fixture_active_import",
    )
    with pytest.raises(Conflict):
        service.delete_knowledge_base("busy-base")
    assert factory().info("busy-base")["document_count"] == 1
    await service.close()


@pytest.mark.asyncio
async def test_dns_failure_preserves_safe_reason_code(setup_service):
    from docguard.discovery.fetch import FetchError

    service, fetcher, _, _ = setup_service

    def failure(url):
        raise FetchError("Do not persist this secret", reason_code="dns_non_public")

    fetcher.fetch = failure
    try:
        task = await service.create("Python config", "Python", "3", [URL])
        task = await settled(service, task["task_id"])
        assert task["status"] == "failed"
        assert task["failures"][0]["reason_code"] == "dns_non_public"
        assert "secret" not in json.dumps(task)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_replace_failed_url_in_same_task_preserves_failure_event(setup_service):
    from docguard.discovery.fetch import FetchError

    service, fetcher, _, _ = setup_service
    original_fetch = fetcher.fetch

    def failed(url):
        raise FetchError("blocked", reason_code="dns_non_public")

    fetcher.fetch = failed
    try:
        first = await service.create("Python config", "Python", "2", [URL])
        failed_task = await settled(service, first["task_id"])
        fetcher.fetch = original_fetch
        resumed = await service.reply(first["task_id"], [URL], version="3")
        final = await settled(service, first["task_id"])
        assert final["task_id"] == failed_task["task_id"]
        assert resumed["revision"] == failed_task["revision"] + 1
        assert final["status"] == "awaiting_selection"
        assert final["version"] == "3"
        updated = next(
            e
            for e in service.events(first["task_id"])
            if e["kind"] == "sources_updated"
        )
        assert (
            updated["data"]["previous_failures"][0]["reason_code"] == "dns_non_public"
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_knowledge_query_invokes_generator_and_closes_client(setup_service):
    from docguard.generation.models import GenerationCompletion

    service, fetcher, factory, _ = setup_service

    class Generator:
        closed = False

        async def generate(self, request):
            assert "evidence" in request.messages[-1]["content"]
            return GenerationCompletion(
                '{"action":"answer","answer":"Use ConfigParser to read INI files [E01].","citations":["E01"]}',
                "model-query-fixture",
                "mock-deepseek",
                "stop",
                120,
                0,
                24,
            )

        async def close(self):
            self.closed = True

    generator = Generator()
    service.query_generator_factory = lambda: generator
    await factory().import_urls(
        "python-docs", [URL], product="Python", version="3", fetcher=fetcher
    )
    answer = await service.query("python-docs", "ConfigParser INI files", generate=True)
    assert answer["mode"] == "generated"
    assert answer["model_info"]["request_id"] == "model-query-fixture"
    assert answer["decision"]["confidence"] is None
    assert answer["decision"]["triggered_rule"] == "generic_citations_validated"
    assert generator.closed


@pytest.mark.asyncio
async def test_requested_generation_without_model_never_falls_back_to_excerpts(
    setup_service,
):
    service, _, _, _ = setup_service
    with pytest.raises(RuntimeError, match="generation"):
        await service.query("python-docs", "INI files", generate=True)


@pytest.mark.asyncio
async def test_confirmation_uses_inspected_bytes_and_duplicate_is_idempotent(
    setup_service,
):
    service, fetcher, factory, _ = setup_service
    await service.start()
    task = await service.create("Find configuration docs", "Python", "3", [URL])
    task = await settled(service, task["task_id"])
    assert task["status"] == "awaiting_selection"
    candidate_id = task["candidates"][0]["candidate_id"]
    job = await service.approve(
        task["task_id"], [candidate_id], task["revision"], "python", "request-1"
    )
    duplicate = await service.approve(
        task["task_id"], [candidate_id], task["revision"], "python", "request-1"
    )
    assert job["job_id"] == duplicate["job_id"]
    task = await settled(service, task["task_id"])
    assert task["status"] == "ready"
    assert fetcher.calls == [URL]
    assert service.get_job(job["job_id"])["status"] == "succeeded"
    assert (await factory().answer("python", "ConfigParser"))["action"] == "answer"
    await service.close()


@pytest.mark.asyncio
async def test_stale_revision_unknown_candidate_and_reused_key_conflict(setup_service):
    from docguard.discovery.store import Conflict

    service, _, _, _ = setup_service
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    selected = [task["candidates"][0]["candidate_id"]]
    with pytest.raises(Conflict):
        await service.approve(
            task["task_id"], selected, task["revision"] + 1, "python", "stale"
        )
    with pytest.raises(Conflict):
        await service.approve(
            task["task_id"], ["missing"], task["revision"], "python", "bad"
        )
    await service.approve(task["task_id"], selected, task["revision"], "python", "same")
    with pytest.raises(Conflict):
        await service.approve(
            task["task_id"], selected, task["revision"], "another-kb", "same"
        )
    await settled(service, task["task_id"])
    await service.close()


@pytest.mark.asyncio
async def test_clarification_persists_across_restart_and_events_resume(setup_service):
    service, fetcher, factory, root = setup_service
    task = await service.create("Need Python docs", "Python", None, [])
    assert task["status"] == "awaiting_input"
    events = service.events(task["task_id"], 0)
    await service.close()
    restored = DiscoveryService(root / "tasks", factory, fetcher=fetcher)
    await restored.start()
    assert restored.get_task(task["task_id"])["pending_question"]
    await restored.reply(task["task_id"], [URL])
    task = await settled(restored, task["task_id"])
    assert task["status"] == "awaiting_selection"
    more = restored.events(task["task_id"], events[-1]["event_id"])
    assert more and all(e["event_id"] > events[-1]["event_id"] for e in more)
    await restored.close()


def app_for(service):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        yield
        await service.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(discovery_router(service), prefix="/api/v1")
    return app


def test_http_validation_404_503_and_event_replay(setup_service):
    service, _, _, _ = setup_service
    with TestClient(app_for(service)) as client:
        assert client.get("/api/v1/discovery/tasks/missing").status_code == 404
        assert (
            client.post(
                "/api/v1/discovery/tasks",
                json={
                    "goal": "docs",
                    "product": "Python",
                    "urls": ["http://localhost/"],
                },
            ).status_code
            == 422
        )
        created = client.post(
            "/api/v1/discovery/tasks", json={"goal": "docs", "product": "Python"}
        )
        assert created.status_code == 202
        task = created.json()
        response = client.get(
            f"/api/v1/discovery/tasks/{task['task_id']}/events",
            headers={"Last-Event-ID": "0"},
        )
        assert response.status_code == 200
        assert "event: task_created" in response.text
        assert "id: " in response.text
        assert (
            client.post(f"/api/v1/discovery/tasks/{task['task_id']}/cancel").json()[
                "status"
            ]
            == "cancelled"
        )
    app = FastAPI()
    app.include_router(discovery_router(None), prefix="/api/v1")
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/discovery/tasks", json={"goal": "docs", "product": "Python"}
            ).status_code
            == 503
        )


def test_sqlite_store_reopens_without_losing_events(tmp_path):
    from docguard.discovery.store import TaskStore

    store = TaskStore(tmp_path / "state.sqlite3")
    task = store.create(
        {"goal": "docs", "product": "Python", "version": None, "urls": []}
    )
    restored = TaskStore(tmp_path / "state.sqlite3")
    assert restored.get(task["task_id"]) == task
    assert restored.events(task["task_id"], 0)[0]["kind"] == "task_created"
    assert "body" not in json.dumps(task)


@pytest.mark.asyncio
async def test_agent_discovery_flows_into_existing_approval_and_import(setup_service):
    from docguard.discovery.providers import Decision, SearchHit

    _, fetcher, factory, root = setup_service
    url = "https://docs.python.org/3/library/configparser.html"

    class Model:
        async def decide(self, context):
            if not context["pool"]:
                action = {
                    "action": "search_documents",
                    "query": "Python ConfigParser docs",
                }
            elif not context["candidates"]:
                action = {
                    "action": "read_page",
                    "candidate_id": context["pool"][0]["candidate_id"],
                }
            else:
                action = {
                    "action": "present_candidates",
                    "candidate_ids": [context["candidates"][0]["candidate_id"]],
                }
            return Decision(action, "fixture", 10, 5, 0, 0.00001)

    class Search:
        async def search(self, query, limit):
            return [SearchHit("Python ConfigParser", url, "INI documentation")]

    service = DiscoveryService(
        root / "agent-tasks",
        factory,
        fetcher=fetcher,
        decision_model=Model(),
        search_provider=Search(),
    )
    task = await service.create(
        "帮我找 Python 配置文件文档", "python", "3", [], document_type="topic"
    )
    task = await settled(service, task["task_id"])
    assert task["status"] == "awaiting_selection"
    assert task["product"] == "python"
    assert task["agent"]["model_calls"] == 3
    job = await service.approve(
        task["task_id"],
        [task["candidates"][0]["candidate_id"]],
        task["revision"],
        "agent-python",
        "agent-request",
    )
    assert (await settled(service, task["task_id"]))["status"] == "ready"
    assert service.get_job(job["job_id"])["status"] == "succeeded"
    assert fetcher.calls == [url]
    await service.close()


@pytest.mark.asyncio
async def test_model_failure_accepts_text_reply_and_preserves_budget(setup_service):
    from docguard.discovery.providers import Decision

    _, fetcher, factory, root = setup_service

    class Model:
        calls = 0

        async def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return Decision({"type": "search_documents"}, "fixture")
            return Decision(
                {"action": "ask_user", "question": "Which version?"},
                "fixture",
                10,
                5,
                0,
                0.00001,
            )

    service = DiscoveryService(
        root / "model-recovery",
        factory,
        fetcher=fetcher,
        decision_model=Model(),
        search_provider=object(),
    )
    try:
        task = await service.create("Find pandas docs", "pandas", None, [])
        failed = await settled(service, task["task_id"])
        assert failed["stop_reason"] == "agent_model_failed"
        reserved = failed["agent"]["reserved_input_tokens"]
        events = service.events(task["task_id"])
        assert any(
            e["kind"] == "agent_failed"
            and e["data"]["reason"] == "model_action_validation_failed"
            for e in events
        )
        await service.reply(task["task_id"], [], message="Continue the task")
        final = await settled(service, task["task_id"])
        assert final["status"] == "awaiting_input"
        assert final["revision"] == 2
        assert final["agent"]["model_calls"] == 2
        assert final["agent"]["reserved_input_tokens"] >= reserved
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_agent_text_clarification_resumes_after_restart(setup_service):
    from docguard.discovery.providers import Decision, SearchHit

    _, fetcher, factory, root = setup_service

    class Ask:
        async def decide(self, context):
            return Decision(
                {"action": "ask_user", "question": "Which Python version?"},
                "fixture",
                10,
                5,
                0,
                0.00001,
            )

    class Search:
        async def search(self, query, limit):
            return [
                SearchHit(
                    "Python docs",
                    "https://docs.python.org/3/library/configparser.html",
                    "ConfigParser",
                )
            ]

    service = DiscoveryService(
        root / "clarify-agent",
        factory,
        fetcher=fetcher,
        decision_model=Ask(),
        search_provider=Search(),
    )
    task = await service.create("Find Python docs", "Python", None, [])
    task = await settled(service, task["task_id"])
    assert task["status"] == "awaiting_input"
    await service.close()

    class Finish:
        async def decide(self, context):
            assert context["version"] == "3"
            assert context["messages"] == ["Use version 3"]
            return Decision(
                {"action": "finish", "reason": "clarification received"},
                "fixture",
                10,
                5,
                0,
                0.00001,
            )

    restored = DiscoveryService(
        root / "clarify-agent",
        factory,
        fetcher=fetcher,
        decision_model=Finish(),
        search_provider=Search(),
    )
    await restored.start()
    await restored.reply(task["task_id"], [], message="Use version 3", version="3")
    task = await settled(restored, task["task_id"])
    assert task["version"] == "3" and task["agent"]["model_calls"] == 2
    assert task["status"] == "failed"
    await restored.close()


@pytest.mark.asyncio
async def test_cancel_running_import_blocks_publication(setup_service):
    import threading

    service, _, factory, root = setup_service
    entered, release = threading.Event(), threading.Event()
    original_base = factory()

    class PausedBase:
        config = original_base.config
        embedding_id = original_base.embedding_id

        async def import_urls(self, *args, **kwargs):
            entered.set()
            release.wait(5)
            return await original_base.import_urls(*args, **kwargs)

    service.knowledge_factory = lambda: PausedBase()
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    job = await service.approve(
        task["task_id"],
        [task["candidates"][0]["candidate_id"]],
        task["revision"],
        "cancel-test",
        "cancel-key",
    )
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    try:
        assert (await service.cancel(task["task_id"]))["status"] == "cancelled"
    finally:
        release.set()
    await service.close()
    assert service.get_job(job["job_id"])["status"] == "cancelled"
    assert not (root / "knowledge/cancel-test/active.json").exists()


@pytest.mark.asyncio
async def test_restart_resumes_queued_approved_snapshot_without_network(setup_service):
    service, fetcher, factory, root = setup_service
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    await service.close()
    job = await service.approve(
        task["task_id"],
        [task["candidates"][0]["candidate_id"]],
        task["revision"],
        "resumed",
        "restart-key",
    )
    assert service.get_job(job["job_id"])["status"] == "queued"
    restored = DiscoveryService(root / "tasks", factory, fetcher=fetcher)
    await restored.start()
    assert (await settled(restored, task["task_id"]))["status"] == "ready"
    assert len(fetcher.calls) == 1
    assert restored.get_job(job["job_id"])["status"] == "succeeded"
    await restored.close()


@pytest.mark.asyncio
async def test_failed_worker_is_terminal_and_provider_message_is_not_saved(
    setup_service,
):
    service, _, _, _ = setup_service

    class BrokenFactory:
        config = load_chunking_config(ROOT / "configs/chunking/s2-chunk-v1.json")
        embedding_id = None

        async def import_urls(self, *args, **kwargs):
            raise RuntimeError("secret=do-not-save /private/path")

    service.knowledge_factory = BrokenFactory
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    job = await service.approve(
        task["task_id"],
        [task["candidates"][0]["candidate_id"]],
        task["revision"],
        "broken",
        "broken-key",
    )
    assert (await settled(service, task["task_id"]))["status"] == "failed"
    stored = service.get_job(job["job_id"])
    assert stored["failure_type"] == "RuntimeError"
    assert "do-not-save" not in json.dumps(stored)
    await service.close()


def test_full_http_import_and_query_journey(setup_service):
    import time

    service, fetcher, _, _ = setup_service
    with TestClient(app_for(service)) as client:
        task = client.post(
            "/api/v1/discovery/tasks",
            json={
                "goal": "Import configuration docs",
                "product": "Python",
                "version": "3",
                "urls": [URL],
            },
        ).json()
        for _ in range(200):
            task = client.get(f"/api/v1/discovery/tasks/{task['task_id']}").json()
            if task["status"] == "awaiting_selection":
                break
            time.sleep(0.01)
        assert task["status"] == "awaiting_selection"
        assert "body" not in json.dumps(task)
        payload = {
            "candidate_ids": [task["candidates"][0]["candidate_id"]],
            "revision": task["revision"] + 1,
            "knowledge_base_id": "http-python",
            "idempotency_key": "http-key",
        }
        endpoint = f"/api/v1/discovery/tasks/{task['task_id']}/approve"
        assert client.post(endpoint, json=payload).status_code == 409
        payload["revision"] = task["revision"]
        response = client.post(endpoint, json=payload)
        assert response.status_code == 202
        job = response.json()
        for _ in range(200):
            final = client.get(f"/api/v1/imports/{job['job_id']}").json()
            if final["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert final["status"] == "succeeded"
        assert client.post(endpoint, json=payload).json()["job_id"] == job["job_id"]
        result = client.post(
            "/api/v1/knowledge-bases/http-python/query",
            json={"question": "ConfigParser INI"},
        ).json()
        assert result["action"] == "answer"
        assert result["citations"][0]["url"].endswith("#config")
        assert (
            client.get("/api/v1/knowledge-bases").json()["knowledge_bases"][0][
                "knowledge_base_id"
            ]
            == "http-python"
        )
        assert (
            client.get(
                f"/api/v1/discovery/tasks/{task['task_id']}/events",
                headers={"Last-Event-ID": "-1"},
            ).status_code
            == 422
        )
        assert fetcher.calls == [URL]


@pytest.mark.asyncio
async def test_snapshot_tampering_fails_without_publication(setup_service):
    service, _, _, root = setup_service
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    candidate_id = task["candidates"][0]["candidate_id"]
    with service.store.connection() as db:
        db.execute(
            "UPDATE snapshots SET body=? WHERE task_id=? AND candidate_id=?",
            (b"changed", task["task_id"], candidate_id),
        )
    job = await service.approve(
        task["task_id"], [candidate_id], task["revision"], "tamper", "tamper-key"
    )
    assert (await settled(service, task["task_id"]))["status"] == "failed"
    assert service.get_job(job["job_id"])["failure_type"] == "ValueError"
    assert not (root / "knowledge/tamper/active.json").exists()
    await service.close()


@pytest.mark.asyncio
async def test_publication_claim_rejects_late_cancellation(setup_service):
    from docguard.discovery.store import Conflict

    service, _, _, _ = setup_service
    task = await service.create("Find docs", "Python", None, [])

    def simulate_publishing(row):
        row["status"] = "importing"
        row["job"] = {"status": "publishing"}

    service.store.mutate(task["task_id"], simulate_publishing, "publication_started")
    with pytest.raises(Conflict):
        await service.cancel(task["task_id"])
    assert service.get_task(task["task_id"])["status"] == "importing"
    await service.close()


@pytest.mark.asyncio
async def test_restart_skips_completed_inspection_failures(setup_service):
    service, fetcher, factory, root = setup_service
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    await service.close()
    failed_url = "https://example.com/docs/failed.html"

    def interrupted(row):
        row["urls"] = [URL, failed_url]
        row["status"] = "discovering"
        row["inspected_urls"] = [URL, failed_url]
        row["failures"] = [
            {
                "url": failed_url,
                "reason_code": "source_inspection_failed",
                "error_type": "FetchError",
            }
        ]

    service.store.mutate(task["task_id"], interrupted, "inspection_interrupted")
    restored = DiscoveryService(root / "tasks", factory, fetcher=fetcher)
    await restored.start()
    assert (await settled(restored, task["task_id"]))["status"] == "awaiting_selection"
    assert fetcher.calls == [URL]
    await restored.close()


@pytest.mark.asyncio
async def test_changed_processing_configuration_requires_new_approval(setup_service):
    from dataclasses import replace

    service, fetcher, factory, root = setup_service
    task = await service.create("Find docs", "Python", None, [URL])
    task = await settled(service, task["task_id"])
    await service.close()
    job = await service.approve(
        task["task_id"],
        [task["candidates"][0]["candidate_id"]],
        task["revision"],
        "changed-config",
        "config-key",
    )

    def changed_factory():
        base = factory()
        base.config = replace(base.config, min_chunk_tokens=65)
        return base

    restored = DiscoveryService(root / "tasks", changed_factory, fetcher=fetcher)
    await restored.start()
    assert (await settled(restored, task["task_id"]))["status"] == "failed"
    assert restored.get_job(job["job_id"])["failure_type"] == "Conflict"
    assert not (root / "knowledge/changed-config/active.json").exists()
    await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [False, True])
async def test_approved_collection_imports_more_than_twenty_or_publishes_nothing(
    setup_service, broken
):
    from dataclasses import replace

    service, fetcher, factory, directory = setup_service
    root = "https://example.com/docs/guide/index.html"
    original_fetch = fetcher.fetch

    def fetch(url):
        if broken and url.endswith("chapter24.html"):
            raise RuntimeError("fixture fetch failure")
        snapshot = original_fetch(url)
        body = (
            "<title>User Guide</title><main><h1>Guide</h1><p>Usage documentation.</p>"
            + (
                "".join(f'<a href="chapter{i}.html">Chapter {i}</a>' for i in range(25))
                + '<a href="../api/index.html">Outside</a>'
                if url == root
                else "<p>Chapter body.</p>"
            )
            + "</main>"
        ).encode()
        return replace(
            snapshot, body=body, content_sha256=hashlib.sha256(body).hexdigest()
        )

    fetcher.fetch = fetch
    task = await service.create(
        None, "Example", None, [root], document_type="user_guide"
    )
    task = await settled(service, task["task_id"])
    assert task["document_type"] == "user_guide"
    await service.approve(
        task["task_id"],
        [task["candidates"][0]["candidate_id"]],
        task["revision"],
        "whole-guide",
        "collection-test",
        import_scope="collection",
    )
    for _ in range(1000):
        result = service.get_task(task["task_id"])
        if result["status"] != "importing":
            break
        await asyncio.sleep(0.02)
    if broken:
        assert result["status"] == "failed"
        assert not (directory / "knowledge/whole-guide/active.json").exists()
        approved_id = result["job"]["approved_manifest_id"]
        broken = False
        retry = await service.retry_import(result["job"]["job_id"])
        assert retry["approved_manifest_id"] == approved_id
        for _ in range(1000):
            result = service.get_task(task["task_id"])
            if result["status"] != "importing":
                break
            await asyncio.sleep(0.02)
        assert result["status"] == "ready", result["job"]
        assert result["job"]["result"]["document_count"] == 26
        assert len(fetcher.calls) == 26
        assert result["job"]["failure_history"][0]["failure_type"] == "RuntimeError"
        from docguard.discovery.store import Conflict

        with pytest.raises(Conflict):
            await service.retry_import(result["job"]["job_id"])
    else:
        assert result["status"] == "ready", result["job"]
        assert result["job"]["result"]["document_count"] == 26
        assert factory().info("whole-guide")["document_count"] == 26
        assert result["job"]["progress"]["phase"] == "completed"
        phases = {
            event["data"].get("phase")
            for event in service.events(task["task_id"])
            if event["kind"] == "import_progress"
        }
        assert {"collecting", "parsing", "chunking", "bm25", "publishing"} <= phases
        assert len(fetcher.calls) == 26
    assert not any("/api/" in url for url in fetcher.calls)
    await service.close()
