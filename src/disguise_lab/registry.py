"""Append-only trial identity with explicit transitions and bounded execution."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class Registry:
    def __init__(self, path: Path, max_trials: int = 24):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.max_trials = path, max_trials
        with self.connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS trials "
                "(id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL, "
                "created TEXT NOT NULL, result TEXT)"
            )

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path) as db:
            yield db

    def start(self, config: dict) -> str:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM trials").fetchone()[0] >= self.max_trials:
                raise RuntimeError("development trial budget exhausted")
            identity = str(uuid4())
            db.execute(
                "INSERT INTO trials VALUES (?,?,?,?,NULL)",
                (
                    identity,
                    json.dumps(config, sort_keys=True),
                    "running",
                    datetime.now(UTC).isoformat(),
                ),
            )
            return identity

    def finish(self, identity: str, status: str, result: dict):
        if status not in ["succeeded", "failed", "cancelled", "budget_exhausted"]:
            raise ValueError("invalid terminal status")
        with self.connection() as db:
            updated = db.execute(
                "UPDATE trials SET status=?,result=? WHERE id=? AND status='running'",
                (status, json.dumps(result, sort_keys=True), identity),
            ).rowcount
            if updated != 1:
                raise ValueError("terminal records cannot be overwritten")

    def records(self) -> list[dict]:
        with self.connection() as db:
            return [
                {
                    "trial_id": i,
                    "config": json.loads(c),
                    "status": s,
                    "created": t,
                    "result": json.loads(r) if r else None,
                }
                for i, c, s, t, r in db.execute("SELECT * FROM trials ORDER BY created")
            ]
