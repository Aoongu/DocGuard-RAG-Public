"""Export one durable discovery trace without loading models or calling APIs."""

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def export_log(database: Path, task_id: str, output: Path) -> dict:
    if re.fullmatch(r"discover-[0-9a-f]{32}", task_id) is None:
        raise ValueError("Invalid discovery task ID")
    # Read-only even when invoked against a currently running application's DB.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        row = db.execute(
            "SELECT payload FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Task does not exist")
        events = [
            {"event_id": event_id, "kind": kind, "data": json.loads(payload)}
            for event_id, kind, payload in db.execute(
                "SELECT event_id,kind,payload FROM events WHERE task_id=? ORDER BY event_id",
                (task_id,),
            )
        ]
        payload = {
            "exported_at": datetime.now(UTC).isoformat(),
            "task": json.loads(row[0]),
            "events": events,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return payload


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "reports/generated/s15-discovery/tasks/state.sqlite3",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or root / "reports/s15_ingestion/logs" / (
        args.task_id + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    try:
        payload = export_log(args.database, args.task_id, output)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Export failed: {type(error).__name__}\n")
    print(f"Exported {len(payload['events'])} events to {output.resolve()}")


if __name__ == "__main__":
    main()
