"""Small local SQLite store for dashboard assessment and review history."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "dashboard_monitoring.sqlite3"


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _database(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the assessment history table if it does not exist."""
    with _database(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_assessments (
                thread_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                created_at TEXT NOT NULL,
                risk_probability REAL,
                composite_risk_score REAL,
                risk_level TEXT NOT NULL,
                review_required INTEGER NOT NULL,
                final_status TEXT NOT NULL,
                review_decision TEXT NOT NULL,
                review_reason TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dashboard_assessments_created_at
            ON dashboard_assessments(created_at)
            """
        )


def save_assessment(
    thread_id: str,
    state: dict[str, Any],
    next_nodes: tuple[str, ...] | list[str] = (),
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Insert or update the summary fields used by the monitoring page."""
    initialize_store(db_path)
    quant_report = state.get("quant_report") or {}
    probability = quant_report.get("risk_probability")
    score = state.get("composite_risk_score")
    probability = probability if isinstance(probability, (int, float)) else None
    score = score if isinstance(score, (int, float)) else None

    review_required = bool(state.get("human_approval_required"))
    verdict = state.get("final_verdict") or {}
    final_status = verdict.get("final_status")
    if "human_governance" in next_nodes:
        final_status = "PENDING_HUMAN_REVIEW"
        review_decision = "PENDING"
    elif review_required and state.get("human_approved") is True:
        review_decision = "APPROVED"
    elif review_required and state.get("human_approved") is False:
        review_decision = "REJECTED"
    else:
        review_decision = "NOT_REQUIRED"

    with _database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO dashboard_assessments (
                thread_id, ticker, created_at, risk_probability,
                composite_risk_score, risk_level, review_required,
                final_status, review_decision, review_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                risk_probability = excluded.risk_probability,
                composite_risk_score = excluded.composite_risk_score,
                risk_level = excluded.risk_level,
                review_required = excluded.review_required,
                final_status = excluded.final_status,
                review_decision = excluded.review_decision,
                review_reason = excluded.review_reason
            """,
            (
                thread_id,
                state.get("ticker", "UNKNOWN"),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                probability,
                score,
                state.get("risk_level") or "UNKNOWN",
                int(review_required),
                final_status or "ASSESSMENT_INCOMPLETE",
                review_decision,
                state.get("rejection_reason"),
            ),
        )


def get_assessment_history(
    limit: int = 500,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return newest assessment summaries first."""
    if limit < 1:
        raise ValueError("limit must be positive")
    initialize_store(db_path)
    with _database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT thread_id, ticker, created_at, risk_probability,
                   composite_risk_score, risk_level, review_required,
                   final_status, review_decision, review_reason
            FROM dashboard_assessments
            ORDER BY created_at DESC, thread_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
