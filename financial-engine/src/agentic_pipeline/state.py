from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class AgentMessage(TypedDict):
    role: str
    content: str


class MultiAgentState(TypedDict):
    """Shared state passed between the risk workflow nodes."""

    ticker: str
    messages: Annotated[list[AgentMessage], operator.add]
    quant_report: dict[str, Any] | None
    audit_report: dict[str, Any] | None
    sentiment_report: dict[str, Any] | None
    composite_risk_score: float | None
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] | None
    human_approval_required: bool
    human_approved: bool | None
    rejection_reason: str | None
    final_verdict: dict[str, Any] | None


# Backward-compatible name for code that imported the previous state type.
FinancialAgentState = MultiAgentState
