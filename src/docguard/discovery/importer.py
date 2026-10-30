"""Single-process workers with durable approval and replay of inspected bytes."""

import asyncio
import hashlib
import logging
import re
import threading
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from docguard.discovery.collection import CollectionError, collect, scope_for_entry
from docguard.discovery.extract import extract_html
from docguard.discovery.fetch import FetchError, SafeFetcher
from docguard.discovery.graph import DiscoveryAgent, initial_agent_state
from docguard.discovery.models import Approval, CreateTask, Reply
from docguard.discovery.store import Conflict, NotFound, TaskStore, encoded, timestamp


class WorkerInterrupted(RuntimeError):
    """Keep the durable job pending until the next application start."""


class DiscoveryService:
    def __init__(
        self,
        root: Path,
        knowledge_factory,
        *,
        fetcher=None,
        decision_model=None,
        search_provider=None,
        agent_limits=None,
        sites=None,
        query_generator_factory=None,
    ):
        self.store = TaskStore(root / "state.sqlite3")
        self.knowledge_factory = knowledge_factory
        self.query_generator_factory = query_generator_factory
        self.fetcher = fetcher or SafeFetcher()
        self._workers = {}
        self._stopping = threading.Event()
        self.agent = DiscoveryAgent(
            self.store,
            decision_model,
            search_provider,
            self.fetcher,
            limits=agent_limits,
            sites=sites,
            stopping=self._stopping.is_set,
        )

    def capabilities(self):
        return {
            "document_dns_mode": getattr(self, "document_dns_mode", "injected"),
            "document_doh_provider": getattr(
                getattr(self.fetcher, "resolver", None), "provider", None
            ),
            "document_doh_active_provider": getattr(
                getattr(self.fetcher, "resolver", None), "last_provider", None
            ),
            "search_provider": getattr(
                self,
                "search_provider_name",
                getattr(self.agent.search, "provider_id", "injected"),
            ),
            "html_import": True,
            "decision_model_configured": self.agent.model is not None,
            "search_configured": self.agent.search is not None,
            "agent_enabled": self.agent.model is not None
            and self.agent.search is not None,
        }

    def _launch(self, key, coroutine):
        if self._stopping.is_set():
            coroutine.close()
            return
        current = self._workers.get(key)
        if current is not None and not current.done():
            current.add_done_callback(lambda _: self._launch(key, coroutine))
            return
        self._workers[key] = asyncio.create_task(coroutine)

    async def start(self):
        self._stopping.clear()
        for task in self.store.list_tasks():
            if task["status"] == "discovering":
                self._launch(
                    task["task_id"],
                    self.agent.run(task["task_id"])
                    if task.get("mode") == "agent"
                    else self._inspect(task["task_id"]),
                )
            elif task["status"] == "importing":
                self._launch(task["task_id"], self._import(task["task_id"]))

    async def close(self):
        self._stopping.set()
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)

    async def create(
        self,
        goal,
        product,
        version,
        urls,
        *,
        official_only=True,
        document_type="user_guide",
    ):
        normalized = CreateTask(
            goal=goal,
            product=product,
            version=version,
            urls=urls,
            official_only=official_only,
            document_type=document_type,
        )
        goal, product, version = normalized.goal, normalized.product, normalized.version
        if version is None and product != "unspecified":
            match = re.search(
                re.escape(product) + r"\s*(?:v(?:ersion)?\s*)?(\d+(?:\.\d+)*)",
                goal,
                re.IGNORECASE,
            )
            if match:
                version = match.group(1)
        request = CreateTask(
            goal=goal,
            product=product,
            version=version,
            urls=urls,
            official_only=official_only,
            document_type=document_type,
        )
        task = self.store.create(
            {**request.model_dump(), "mode": "urls" if urls else "agent"}
        )
        if not urls and self.capabilities()["agent_enabled"]:
            task = self.store.mutate(
                task["task_id"],
                lambda r: r.update(status="discovering", pending_question=None),
                "agent_queued",
            )
            self._launch(task["task_id"], self.agent.run(task["task_id"]))
            return task
        if not urls:
            return self.store.mutate(
                task["task_id"],
                lambda r: r.update(stop_reason="agent_dependencies_unavailable"),
                "agent_unavailable",
            )
        if task["status"] == "discovering":
            self._launch(task["task_id"], self._inspect(task["task_id"]))
        return task

    def get_task(self, task_id):
        return self.store.get(task_id)

    def get_job(self, job_id):
        return self.store.get_job(job_id)

    def events(self, task_id, after=0):
        return self.store.events(task_id, after)

    async def reply(
        self,
        task_id,
        urls,
        *,
        message=None,
        version=None,
        product=None,
        allow_unversioned=False,
    ):
        request = Reply(
            urls=urls,
            message=message,
            version=version,
            product=product,
            allow_unversioned=allow_unversioned,
        )

        if not request.urls:

            def clarify(row):
                if (
                    row.get("mode") != "agent"
                    or (
                        row["status"] not in ("awaiting_input", "awaiting_selection")
                        and not (
                            row["status"] == "failed"
                            and row.get("stop_reason") == "agent_model_failed"
                        )
                    )
                    or row["job"] is not None
                ):
                    raise Conflict("Task cannot accept agent clarification")
                row.setdefault("agent", initial_agent_state())
                if request.version is not None or request.allow_unversioned:
                    row["version"] = (
                        None if request.allow_unversioned else request.version
                    )
                    row["candidates"] = []
                if request.product is not None:
                    row["product"] = request.product
                    row["candidates"] = []
                if "agent" in row and request.message:
                    if len(row["agent"]["messages"]) >= 8:
                        raise Conflict("Clarification limit reached")
                    row["agent"]["messages"].append(request.message)
                row.update(
                    revision=row["revision"] + 1,
                    status="discovering"
                    if self.capabilities()["agent_enabled"]
                    else "awaiting_input",
                    pending_question=None
                    if self.capabilities()["agent_enabled"]
                    else "请配置模型与搜索服务，或直接提供文档网址。",
                    stop_reason=None,
                )

            task = self.store.mutate(task_id, clarify, "user_clarification")
            if task["status"] == "discovering":
                self._launch(task_id, self.agent.run(task_id))
            return task

        update_event = {"operation": "replace_sources"}

        def change(task):
            if (
                task["status"] not in ("awaiting_input", "awaiting_selection", "failed")
                or task["job"] is not None
            ):
                raise Conflict("Task cannot accept new sources")
            update_event.update(
                previous_revision=task["revision"],
                previous_urls=task["urls"],
                previous_failures=task["failures"],
            )
            if request.product is not None:
                task["product"] = request.product
            if request.version is not None or request.allow_unversioned:
                task["version"] = None if request.allow_unversioned else request.version
            task.update(
                mode="urls",
                urls=request.urls,
                candidates=[],
                inspected_urls=[],
                downloaded_bytes=0,
                failures=[],
                pending_question=None,
                status="discovering",
                revision=task["revision"] + 1,
            )

        task = self.store.mutate(task_id, change, "sources_updated", update_event)
        self._launch(task_id, self._inspect(task_id))
        return task

    async def _inspect(self, task_id):
        task = self.store.get(task_id)
        revision = task["revision"]
        try:
            total_bytes = task.get("downloaded_bytes", 0)
            for url in task["urls"]:
                current = self.store.get(task_id)
                if self._stopping.is_set() or current["status"] != "discovering":
                    return
                if url in current.get("inspected_urls", []) or any(
                    c["requested_url"] == url for c in current["candidates"]
                ):
                    continue
                if total_bytes >= 50 * 1024 * 1024:
                    break
                try:
                    snapshot = await asyncio.to_thread(self.fetcher.fetch, url)
                    total_bytes += len(snapshot.body)
                    if total_bytes > 50 * 1024 * 1024:
                        raise ValueError("Task byte limit exceeded")
                    page = await asyncio.to_thread(
                        extract_html, snapshot.body, snapshot.final_url
                    )
                    candidate_id = (
                        "candidate-"
                        + hashlib.sha256(
                            (snapshot.final_url + snapshot.content_sha256).encode()
                        ).hexdigest()
                    )
                    self.store.save_snapshot(task_id, candidate_id, snapshot)
                    candidate = {
                        "candidate_id": candidate_id,
                        "requested_url": url,
                        "url": snapshot.final_url,
                        "collection": scope_for_entry(snapshot.final_url),
                        "title": page.title,
                        "content_sha256": snapshot.content_sha256,
                        "version": task["version"],
                        "version_status": "user_supplied_unverified",
                        "source_status": "user_supplied_unverified",
                        "status": "inspected",
                    }

                    def record(
                        row, candidate=candidate, url=url, downloaded=total_bytes
                    ):
                        if (
                            row["status"] != "discovering"
                            or row["revision"] != revision
                        ):
                            raise Conflict("Inspection is no longer current")
                        row.setdefault("inspected_urls", []).append(url)
                        row["downloaded_bytes"] = downloaded
                        if not any(
                            c["candidate_id"] == candidate["candidate_id"]
                            for c in row["candidates"]
                        ):
                            row["candidates"].append(candidate)

                    self.store.mutate(
                        task_id,
                        record,
                        "candidate_inspected",
                        {"candidate_id": candidate_id},
                    )
                except Conflict:
                    return
                except (OSError, RuntimeError, ValueError) as error:
                    error_type = type(error).__name__
                    reason_code = (
                        error.reason_code
                        if isinstance(error, FetchError)
                        else "source_inspection_failed"
                    )

                    network_attempts = getattr(error, "attempts", [])

                    def failed(
                        row,
                        error_type=error_type,
                        reason_code=reason_code,
                        url=url,
                        downloaded=total_bytes,
                        network_attempts=network_attempts,
                    ):
                        if row["status"] != "discovering":
                            raise Conflict("Task stopped")
                        row.setdefault("inspected_urls", []).append(url)
                        row["downloaded_bytes"] = downloaded
                        row["failures"].append(
                            {
                                "url": url,
                                "reason_code": reason_code,
                                "error_type": error_type,
                                "network_attempts": network_attempts,
                            }
                        )

                    self.store.mutate(task_id, failed, "source_failed")
            if self._stopping.is_set():
                return

            def finish(row):
                if row["status"] != "discovering" or row["revision"] != revision:
                    raise Conflict("Task stopped")
                row["status"] = "awaiting_selection" if row["candidates"] else "failed"

            self.store.mutate(task_id, finish, "inspection_completed")
        except Conflict:
            return

    async def approve(
        self,
        task_id,
        candidate_ids,
        revision,
        kb_id,
        idempotency_key,
        *,
        import_scope="pages",
    ):
        request = Approval(
            candidate_ids=candidate_ids,
            revision=revision,
            knowledge_base_id=kb_id,
            idempotency_key=idempotency_key,
            import_scope=import_scope,
        )
        data = request.model_dump()
        data["candidate_ids"] = sorted(set(candidate_ids))
        signature = hashlib.sha256(encoded(data).encode()).hexdigest()
        base = await asyncio.to_thread(self.knowledge_factory)
        processing_identity = {
            "packing": asdict(base.config),
            "embedding_id": base.embedding_id,
            "parser_version": "generic-html-v1",
            "chunking_version": "generic-chunk-v1",
        }

        def approve(row):
            if row["job"] is not None:
                if row["job"]["request_signature"] == signature:
                    return
                raise Conflict("Task already has an approved import")
            if row["status"] != "awaiting_selection" or row["revision"] != revision:
                raise Conflict(
                    "Candidate revision is stale or task is not awaiting selection"
                )
            mapping = {c["candidate_id"]: c for c in row["candidates"]}
            if any(c not in mapping for c in data["candidate_ids"]):
                raise Conflict("Only inspected candidates can be approved")
            manifest = {
                "processing_identity": processing_identity,
                "task_id": task_id,
                "revision": revision,
                "knowledge_base_id": kb_id,
                "product": row["product"],
                "version": row["version"],
                "candidates": [mapping[c] for c in data["candidate_ids"]],
            }
            if import_scope == "collection":
                if len(manifest["candidates"]) != 1:
                    raise Conflict("Select one documentation collection")
                scope = scope_for_entry(manifest["candidates"][0]["url"])
                if scope is None:
                    raise Conflict("This page is not a bounded collection entry")
                manifest["collection"] = scope
            row["approved_manifest"] = manifest
            row["job"] = {
                "job_id": "import-" + uuid4().hex,
                "task_id": task_id,
                "status": "queued",
                "knowledge_base_id": kb_id,
                "request_signature": signature,
                "idempotency_key": idempotency_key,
                "approved_manifest_id": hashlib.sha256(
                    encoded(manifest).encode()
                ).hexdigest(),
                "result": None,
                "failure_type": None,
            }
            row["status"] = "importing"

        task = self.store.mutate(task_id, approve, "import_approved")
        if task["status"] == "importing":
            self._launch(task_id, self._import(task_id))
        return task["job"]

    async def retry_import(self, job_id):
        """Retry the same approved import, reusing immutable cached HTML pages."""
        task_id = self.store.get_job(job_id)["task_id"]

        def retry(row):
            job = row["job"]
            if row["status"] != "failed" or job["status"] != "failed":
                raise Conflict("Only failed imports can be retried")
            if (
                hashlib.sha256(encoded(row["approved_manifest"]).encode()).hexdigest()
                != job["approved_manifest_id"]
            ):
                raise Conflict("Approval manifest hash mismatch")
            job.setdefault("failure_history", []).append(
                {
                    "failure_type": job.get("failure_type"),
                    "failure_reason": job.get("failure_reason"),
                    "failure_url": job.get("failure_url"),
                    "retried_at": timestamp(),
                }
            )
            job.update(
                status="queued",
                failure_type=None,
                failure_reason=None,
                failure_url=None,
            )
            row["status"] = "importing"

        row = self.store.mutate(task_id, retry, "import_retry_requested")
        self._launch(task_id, self._import(task_id))
        return row["job"]

    def _publication_guard(self, task_id):
        def claim(row):
            if row["status"] == "cancelled":
                raise Conflict("Import was cancelled before publication")
            if self._stopping.is_set() and row["job"]["status"] != "publishing":
                raise WorkerInterrupted("Worker is stopping")
            if row["status"] != "importing":
                raise Conflict("Import is no longer active")
            row["job"]["status"] = "publishing"

        self.store.mutate(task_id, claim, "publication_started")

    async def _import(self, task_id):
        try:
            current = self.store.get(task_id)
            if current["status"] != "importing" or self._stopping.is_set():
                return
            approved = current["approved_manifest"]
            if (
                hashlib.sha256(encoded(approved).encode()).hexdigest()
                != current["job"]["approved_manifest_id"]
            ):
                raise Conflict("Approval manifest hash mismatch")
            snapshots = {
                c["requested_url"]: self.store.snapshot(task_id, c["candidate_id"])
                for c in approved["candidates"]
            }

            class CachedFetcher:
                def fetch(self, url):
                    if url not in snapshots:
                        raise Conflict("URL is outside approved manifest")
                    return snapshots[url]

            def mark_running(row):
                if row["status"] != "importing":
                    raise Conflict("Import stopped")
                if row["job"]["status"] != "publishing":
                    row["job"]["status"] = "running"

            self.store.mutate(task_id, mark_running, "import_started")

            def run():
                nonlocal snapshots

                def check():
                    if self._stopping.is_set():
                        raise WorkerInterrupted("Worker is stopping")
                    if self.store.get(task_id)["status"] != "importing":
                        raise Conflict("Import stopped")

                def progress(value):
                    check()
                    value = {**value, "updated_at": timestamp()}
                    self.store.mutate(
                        task_id,
                        lambda row: row["job"].update(progress=value),
                        "import_progress",
                        value,
                    )

                limits = {}
                if scope := approved.get("collection"):

                    def load(url):
                        for snapshot in snapshots.values():
                            if snapshot.final_url == url:
                                return snapshot
                        key = "collection-" + hashlib.sha256(url.encode()).hexdigest()
                        try:
                            return self.store.snapshot(task_id, key)
                        except NotFound:
                            snapshot = self.fetcher.fetch(url)
                            self.store.save_snapshot(task_id, key, snapshot)
                            return snapshot

                    snapshots = collect(scope, load, progress=progress, check=check)
                    limits = {
                        "max_pages": scope["max_pages"],
                        "max_total_bytes": scope["max_bytes"],
                    }
                base = self.knowledge_factory()
                actual_identity = {
                    "packing": asdict(base.config),
                    "embedding_id": base.embedding_id,
                    "parser_version": "generic-html-v1",
                    "chunking_version": "generic-chunk-v1",
                }
                if actual_identity != approved["processing_identity"]:
                    raise Conflict("Processing configuration changed after approval")
                return asyncio.run(
                    base.import_urls(
                        approved["knowledge_base_id"],
                        list(snapshots),
                        product=approved["product"],
                        version=approved["version"],
                        fetcher=CachedFetcher(),
                        before_publish=lambda: self._publication_guard(task_id),
                        progress=progress,
                        **limits,
                    )
                )

            result = await asyncio.to_thread(run)

            def complete(row):
                row["status"] = "ready"
                row["job"].update(
                    status="succeeded",
                    progress={
                        "phase": "completed",
                        "completed": 1,
                        "total": 1,
                        "unit": "steps",
                        "updated_at": timestamp(),
                    },
                    result={
                        "snapshot_id": result["snapshot_id"],
                        "document_count": result["document_count"],
                        "chunk_count": result["chunk_count"],
                    },
                )

            self.store.mutate(task_id, complete, "import_completed")
        except WorkerInterrupted:
            return
        except (OSError, RuntimeError, ValueError) as error:
            error_type = type(error).__name__
            reason = (
                str(error)
                if isinstance(error, CollectionError)
                else getattr(error, "reason_code", "import_failed")
            )

            failure_url = getattr(error, "source_url", None)
            logging.getLogger(__name__).exception(
                "Import failed task_id=%s source_url=%s reason=%s",
                task_id,
                failure_url,
                reason,
            )

            def fail(row):
                if row["status"] != "cancelled":
                    row["status"] = "failed"
                    row["job"].update(
                        status="failed",
                        failure_type=error_type,
                        failure_reason=reason,
                        failure_url=failure_url,
                    )

            self.store.mutate(task_id, fail, "import_failed")

    async def cancel(self, task_id):
        def cancel(row):
            if row["status"] in ("ready", "failed", "cancelled"):
                return
            if row["job"] is not None:
                if row["job"]["status"] == "publishing":
                    raise Conflict("Publication has started; wait for the final result")
                row["job"]["status"] = "cancelled"
            row.update(status="cancelled", pending_question=None)

        return self.store.mutate(task_id, cancel, "task_cancelled")

    def list_knowledge_bases(self):
        base = self.knowledge_factory()
        result = []
        for directory in sorted(base.root.iterdir()):
            if directory.is_dir() and (directory / "active.json").is_file():
                try:
                    info = base.info(directory.name)
                except FileNotFoundError:
                    # A concurrent deletion may unpublish after the list scan.
                    continue
                result.append(
                    {
                        k: info[k]
                        for k in (
                            "knowledge_base_id",
                            "snapshot_id",
                            "product",
                            "version",
                            "document_count",
                            "chunk_count",
                        )
                    }
                )
        return result

    def delete_knowledge_base(self, kb_id):
        for task in self.store.list_tasks():
            job = task.get("job")
            if (
                job
                and job["knowledge_base_id"] == kb_id
                and job["status"]
                in (
                    "queued",
                    "running",
                    "publishing",
                )
            ):
                raise Conflict("Knowledge base import is active")
        try:
            return self.knowledge_factory().delete(kb_id)
        except FileNotFoundError:
            raise NotFound("Knowledge base does not exist") from None

    async def query(self, kb_id, question, *, generate=False):
        def run():
            async def answer():
                if generate and self.query_generator_factory is None:
                    raise RuntimeError("knowledge_generation_not_configured")
                client = self.query_generator_factory() if generate else None
                try:
                    return await self.knowledge_factory().answer(
                        kb_id, question, generator=client
                    )
                finally:
                    if client is not None:
                        await client.close()

            return asyncio.run(answer())

        try:
            return await asyncio.to_thread(run)
        except FileNotFoundError:
            raise NotFound("Knowledge base does not exist") from None
