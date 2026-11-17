"""Workflow tests for quantitative gating and human approval decisions."""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic_pipeline import graph as workflow


class AgenticWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.quant_result = {
            "status": "available",
            "risk_probability": 0.2,
        }

        patches = (
            patch.object(
                workflow,
                "quant_agent_node",
                side_effect=lambda state: {"quant_report": self.quant_result},
            ),
            patch.object(
                workflow,
                "audit_agent_node",
                return_value={
                    "audit_report": {"available": True, "documents": [], "count": 0}
                },
            ),
            patch.object(
                workflow,
                "sentiment_agent_node",
                return_value={
                    "sentiment_report": {
                        "news": [],
                        "result_count": 0,
                        "sentiment_label": "UNASSESSED",
                    }
                },
            ),
        )
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    @staticmethod
    def initial_state() -> dict:
        return {
            "ticker": "TEST",
            "messages": [],
            "quant_report": None,
            "audit_report": None,
            "sentiment_report": None,
            "composite_risk_score": None,
            "risk_level": None,
            "human_approval_required": False,
            "human_approved": None,
            "rejection_reason": None,
            "final_verdict": None,
        }

    def invoke(self, thread_id: str):
        app = workflow.create_multi_agent_graph()
        config = {"configurable": {"thread_id": thread_id}}
        app.invoke(self.initial_state(), config=config)
        return app, config

    def test_low_risk_assessment_reaches_execution_without_review(self) -> None:
        app, config = self.invoke("low-risk")

        snapshot = app.get_state(config)
        self.assertEqual(snapshot.next, ())
        self.assertEqual(snapshot.values["risk_level"], "LOW")
        self.assertFalse(snapshot.values["human_approval_required"])
        self.assertEqual(
            snapshot.values["final_verdict"]["final_status"],
            "READY_FOR_EXECUTION",
        )

    def test_high_risk_pauses_then_approval_resumes_workflow(self) -> None:
        self.quant_result = {"status": "available", "risk_probability": 0.6}
        app, config = self.invoke("high-risk-approved")

        paused = app.get_state(config)
        self.assertEqual(paused.values["risk_level"], "HIGH")
        self.assertIn("human_governance", paused.next)
        self.assertIsNone(paused.values["final_verdict"])

        app.invoke(
            Command(resume={"approved": True, "reason": None}),
            config=config,
        )
        completed = app.get_state(config)
        self.assertEqual(completed.next, ())
        self.assertTrue(completed.values["human_approved"])
        self.assertEqual(
            completed.values["final_verdict"]["final_status"],
            "APPROVED_FOR_EXECUTION",
        )

    def test_rejection_records_reason_and_blocks_execution(self) -> None:
        self.quant_result = {"status": "available", "risk_probability": 0.6}
        app, config = self.invoke("high-risk-rejected")

        app.invoke(
            Command(resume={"approved": False, "reason": "Evidence is incomplete."}),
            config=config,
        )
        verdict = app.get_state(config).values["final_verdict"]
        self.assertEqual(
            verdict["final_status"],
            "BLOCKED_BY_REVIEW: Evidence is incomplete.",
        )

    def test_unavailable_quantitative_result_blocks_execution(self) -> None:
        self.quant_result = {
            "status": "unavailable",
            "risk_probability": None,
            "error": "prediction service unavailable",
        }
        app, config = self.invoke("missing-quant")

        snapshot = app.get_state(config)
        self.assertEqual(snapshot.next, ())
        self.assertEqual(snapshot.values["risk_level"], "UNKNOWN")
        self.assertFalse(snapshot.values["human_approval_required"])
        self.assertEqual(
            snapshot.values["final_verdict"]["final_status"],
            "BLOCKED_MISSING_QUANTITATIVE_DATA",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
