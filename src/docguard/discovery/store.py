"""Transactional task, approval, snapshot and event ledger for the local worker."""

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from docguard.discovery.fetch import Snapshot
from docguard.fs_safety import ensure_directory, validate_regular_file


class NotFound(LookupError):
    """Requested task or job does not exist."""


class Conflict(ValueError):
    """The action conflicts with the durable task state."""


def timestamp():
    return datetime.now(UTC).isoformat()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TaskStore:
    def __init__(self, path: Path):
        ensure_directory(path.parent)
        if path.exists():
            validate_regular_file(path)
        self.path = path
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_task ON events(task_id,event_id);
                CREATE TABLE IF NOT EXISTS snapshots(task_id TEXT NOT NULL,candidate_id TEXT NOT NULL,metadata TEXT NOT NULL,body BLOB NOT NULL,PRIMARY KEY(task_id,candidate_id));
            """)

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.execute("PRAGMA busy_timeout=5000")
            with db:
                yield db

    @staticmethod
    def _get(db, task_id):
        row = db.execute(
            "SELECT payload FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise NotFound("Task does not exist")
        return json.loads(row[0])

    def create(self, request):
        task_id = "discover-" + uuid4().hex
        task = {
            **request,
            "task_id": task_id,
            "status": "discovering" if request["urls"] else "awaiting_input",
            "revision": 1,
            "candidates": [],
            "inspected_urls": [],
            "downloaded_bytes": 0,
            "failures": [],
            "pending_question": None
            if request["urls"]
            else "请提供要导入的技术文档网址。",
            "job": None,
            "created_at": timestamp(),
            "updated_at": timestamp(),
        }
        with self.connection() as db:
            db.execute("INSERT INTO tasks VALUES (?,?)", (task_id, encoded(task)))
            db.execute(
                "INSERT INTO events(task_id,kind,payload) VALUES (?,?,?)",
                (task_id, "task_created", encoded({"status": task["status"]})),
            )
        return task

    def get(self, task_id):
        with self.connection() as db:
            return self._get(db, task_id)

    def list_tasks(self):
        with self.connection() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT payload FROM tasks ORDER BY rowid")
            ]

    def mutate(self, task_id, change, kind, payload=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            task = self._get(db, task_id)
            change(task)
            task["updated_at"] = timestamp()
            db.execute(
                "UPDATE tasks SET payload=? WHERE task_id=?", (encoded(task), task_id)
            )
            db.execute(
                "INSERT INTO events(task_id,kind,payload) VALUES (?,?,?)",
                (task_id, kind, encoded(payload or {"status": task["status"]})),
            )
            return task

    def save_snapshot(self, task_id, candidate_id, snapshot):
        metadata = asdict(snapshot)
        body = metadata.pop("body")
        if hashlib.sha256(body).hexdigest() != snapshot.content_sha256:
            raise ValueError("Snapshot hash mismatch")
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?)",
                (task_id, candidate_id, encoded(metadata), body),
            )

    def snapshot(self, task_id, candidate_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT metadata,body FROM snapshots WHERE task_id=? AND candidate_id=?",
                (task_id, candidate_id),
            ).fetchone()
        if row is None:
            raise NotFound("Candidate snapshot does not exist")
        snapshot = Snapshot(**json.loads(row[0]), body=bytes(row[1]))
        if hashlib.sha256(snapshot.body).hexdigest() != snapshot.content_sha256:
            raise ValueError("Snapshot hash mismatch")
        return snapshot

    def get_job(self, job_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM tasks WHERE json_extract(payload,'$.job.job_id')=?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise NotFound("Import job does not exist")
        return json.loads(row[0])["job"]

    def events(self, task_id, after=0):
        self.get(task_id)
        if after < 0:
            raise ValueError("Event cursor cannot be negative")
        with self.connection() as db:
            rows = db.execute(
                "SELECT event_id,kind,payload FROM events WHERE task_id=? AND event_id>? ORDER BY event_id LIMIT 500",
                (task_id, after),
            ).fetchall()
        return [
            {"event_id": row[0], "kind": row[1], "data": json.loads(row[2])}
            for row in rows
        ]
