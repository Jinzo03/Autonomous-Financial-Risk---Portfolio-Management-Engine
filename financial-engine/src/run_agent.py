"""Command-line runner for the multi-source financial risk workflow."""

import argparse
import json
import logging
from uuid import uuid4

from langgraph.types import Command

from src.agentic_pipeline.graph import create_multi_agent_graph


logger = logging.getLogger(__name__)


def _print_updates(app, initial_state: dict, config: dict) -> dict:
    result = app.invoke(initial_state, config=config)
    for message in result.get("messages", []):
        logger.info("%s: %s", message["role"], message["content"])
    return result


def run_agentic_workflow(
    ticker: str = "AAPL",
    thread_id: str | None = None,
    decision: str | None = None,
) -> dict:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be empty")

    thread_id = thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    app = create_multi_agent_graph()
    initial_state = {
        "ticker": ticker,
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

    print("\n--- Starting financial risk workflow ---")
    _print_updates(app, initial_state, config)

    snapshot = app.get_state(config)
    if "human_governance" in snapshot.next:
        print(
            f"\nReview required for {ticker}: "
            f"{snapshot.values.get('risk_level')} risk "
            f"(score {snapshot.values.get('composite_risk_score')})."
        )
        response = decision or input("Approve execution? (yes/no): ").strip().lower()
        approved = response in {"yes", "y", "approve", "approved"}
        reason = None if approved else "Rejected by human reviewer."
        app.invoke(
            Command(resume={"approved": approved, "reason": reason}),
            config=config,
        )

    final_state = app.get_state(config).values
    verdict = final_state.get("final_verdict")
    print("\n--- Final risk decision ---")
    print(json.dumps(verdict, indent=2, default=str))
    return verdict or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--thread-id", help="Optional checkpoint thread ID")
    parser.add_argument(
        "--decision",
        choices=("approve", "reject"),
        help="Human decision for a high-risk case; prompts interactively if omitted",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_agentic_workflow(
        ticker=args.ticker,
        thread_id=args.thread_id,
        decision=args.decision,
    )


if __name__ == "__main__":
    main()
