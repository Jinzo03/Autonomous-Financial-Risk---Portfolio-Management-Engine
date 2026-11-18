"""Tests for durable dashboard assessment and review history."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.monitoring_store import get_assessment_history, save_assessment


class DashboardMonitoringStoreTests(unittest.TestCase):
    def test_pending_review_is_updated_after_decision(self) -> None:
        with TemporaryDirectory(prefix="dashboard-monitoring-") as temp_dir:
            database = Path(temp_dir) / "history.sqlite3"
            pending_state = {
                "ticker": "AAPL",
                "quant_report": {"risk_probability": 0.62},
                "composite_risk_score": 0.62,
                "risk_level": "HIGH",
                "human_approval_required": True,
                "human_approved": None,
                "rejection_reason": None,
                "final_verdict": None,
            }

            save_assessment("thread-1", pending_state, ("human_governance",), database)
            first_record = get_assessment_history(db_path=database)[0]
            self.assertEqual(first_record["review_decision"], "PENDING")
            self.assertEqual(first_record["final_status"], "PENDING_HUMAN_REVIEW")

            rejected_state = {
                **pending_state,
                "human_approved": False,
                "rejection_reason": "Evidence is incomplete.",
                "final_verdict": {
                    "final_status": "BLOCKED_BY_REVIEW: Evidence is incomplete."
                },
            }
            save_assessment("thread-1", rejected_state, (), database)

            history = get_assessment_history(db_path=database)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["review_decision"], "REJECTED")
            self.assertEqual(
                history[0]["final_status"],
                "BLOCKED_BY_REVIEW: Evidence is incomplete.",
            )
            self.assertEqual(history[0]["review_reason"], "Evidence is incomplete.")
            self.assertEqual(history[0]["created_at"], first_record["created_at"])

    def test_non_review_assessment_is_marked_not_required(self) -> None:
        with TemporaryDirectory(prefix="dashboard-monitoring-") as temp_dir:
            database = Path(temp_dir) / "history.sqlite3"
            save_assessment(
                "thread-2",
                {
                    "ticker": "MSFT",
                    "quant_report": {"risk_probability": 0.18},
                    "composite_risk_score": 0.18,
                    "risk_level": "LOW",
                    "human_approval_required": False,
                    "final_verdict": {"final_status": "READY_FOR_EXECUTION"},
                },
                (),
                database,
            )

            record = get_assessment_history(db_path=database)[0]
            self.assertEqual(record["review_decision"], "NOT_REQUIRED")
            self.assertEqual(record["final_status"], "READY_FOR_EXECUTION")


if __name__ == "__main__":
    unittest.main(verbosity=2)
