"""Append-only SQLite provenance store for the SWE-bench Long Memory arm."""

from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import ReflectionCase, TaskRecord


SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    if len(blob) % 4:
        raise ValueError("invalid float32 embedding payload")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class MemoryStore:
    """Owns a new database only for an explicitly enabled Long Memory run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
              key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectories (
              trajectory_id TEXT PRIMARY KEY,
              instance_id TEXT NOT NULL,
              order_id TEXT NOT NULL,
              sequence_index INTEGER NOT NULL,
              issue_text TEXT NOT NULL,
              run_config_json TEXT NOT NULL,
              artifact_paths_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS trajectories_order_position
              ON trajectories(order_id, sequence_index);
            CREATE TABLE IF NOT EXISTS reflection_cases (
              case_id TEXT PRIMARY KEY,
              trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              instance_id TEXT NOT NULL,
              outcome TEXT NOT NULL,
              trigger TEXT NOT NULL,
              evidence TEXT NOT NULL,
              hypothesized_cause TEXT NOT NULL,
              preventative_action TEXT NOT NULL,
              limitations TEXT NOT NULL,
              confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
              embedding_text TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              embedding_dim INTEGER NOT NULL,
              embedding BLOB NOT NULL,
              status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'retired')),
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS reflection_cases_lookup
              ON reflection_cases(status, embedding_model, embedding_dim);
            CREATE TABLE IF NOT EXISTS memory_sources (
              source_id TEXT PRIMARY KEY,
              case_id TEXT NOT NULL REFERENCES reflection_cases(case_id),
              trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              source_kind TEXT NOT NULL,
              locator TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS retrieval_events (
              event_id TEXT PRIMARY KEY,
              instance_id TEXT NOT NULL,
              order_id TEXT NOT NULL,
              sequence_index INTEGER NOT NULL,
              candidate_case_ids_json TEXT NOT NULL,
              selected_case_ids_json TEXT NOT NULL,
              guidance TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def record_trajectory(self, task: TaskRecord) -> str:
        trajectory_id = str(uuid.uuid4())
        self.connection.execute(
            """INSERT INTO trajectories(trajectory_id, instance_id, order_id, sequence_index,
               issue_text, run_config_json, artifact_paths_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                trajectory_id, task.instance_id, task.order_id, task.sequence_index,
                task.issue_text, _json(task.run_config), _json(task.artifact_paths),
            ),
        )
        self.connection.commit()
        return trajectory_id

    def admit_case(
        self, case: ReflectionCase, *, embedding_model: str, vector: Sequence[float]
    ) -> None:
        if not vector:
            raise ValueError("memory embeddings must not be empty")
        self.connection.execute(
            """INSERT INTO reflection_cases(case_id, trajectory_id, instance_id, outcome, trigger,
                evidence, hypothesized_cause, preventative_action, limitations, confidence,
                embedding_text, embedding_model, embedding_dim, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.case_id, case.trajectory_id, case.instance_id, case.outcome, case.trigger,
                case.evidence, case.hypothesized_cause, case.preventative_action,
                case.limitations, case.confidence, case.embedding_text, embedding_model,
                len(vector), _pack(vector),
            ),
        )
        self.connection.executemany(
            "INSERT INTO memory_sources(source_id, case_id, trajectory_id, source_kind, locator) VALUES (?, ?, ?, ?, ?)",
            [
                (str(uuid.uuid4()), case.case_id, case.trajectory_id, "agent_visible", ref)
                for ref in case.source_refs
            ],
        )
        self.connection.commit()

    def candidate_cases(
        self, *, task: TaskRecord, embedding_model: str, dimension: int
    ) -> Iterable[tuple[ReflectionCase, list[float]]]:
        rows = self.connection.execute(
            """SELECT * FROM reflection_cases
               WHERE status='active' AND embedding_model=? AND embedding_dim=?
                 AND instance_id != ?
               ORDER BY created_at ASC""",
            (embedding_model, dimension, task.instance_id),
        ).fetchall()
        for row in rows:
            yield (
                ReflectionCase(
                    case_id=row["case_id"], trajectory_id=row["trajectory_id"],
                    instance_id=row["instance_id"], outcome=row["outcome"], trigger=row["trigger"],
                    evidence=row["evidence"], hypothesized_cause=row["hypothesized_cause"],
                    preventative_action=row["preventative_action"], limitations=row["limitations"],
                    confidence=float(row["confidence"]), embedding_text=row["embedding_text"],
                    source_refs=(),
                ),
                _unpack(row["embedding"]),
            )

    def record_retrieval(
        self, *, task: TaskRecord, candidate_case_ids: Sequence[str],
        selected_case_ids: Sequence[str], guidance: str, prompt_version: str
    ) -> str:
        event_id = str(uuid.uuid4())
        self.connection.execute(
            """INSERT INTO retrieval_events(event_id, instance_id, order_id, sequence_index,
                candidate_case_ids_json, selected_case_ids_json, guidance, prompt_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, task.instance_id, task.order_id, task.sequence_index,
                _json(candidate_case_ids), _json(selected_case_ids), guidance, prompt_version,
            ),
        )
        self.connection.commit()
        return event_id
