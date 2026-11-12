import logging
from math import isfinite
from typing import Any

from langgraph.types import interrupt

from src.agentic_pipeline.state import MultiAgentState
from src.agentic_pipeline.tools import (
    get_quantitative_risk_tool,
    query_sec_rag_tool,
    search_market_sentiment_tool
)

logger = logging.getLogger(__name__)

# --- Agent 1: Quantitative Risk Agent ---
def quant_agent_node(state: MultiAgentState) -> dict[str, Any]:
    ticker = state["ticker"]
    logger.info(f"[QuantAgent] Invoking get_quantitative_risk_tool for {ticker}...")
    
    # Execute tool
    tool_output = get_quantitative_risk_tool(ticker)
    
    return {
        "quant_report": tool_output,
        "messages": [{
            "role": "QuantAgent",
            "content": f"Executed quantitative tool: Risk Score = {tool_output.get('risk_probability', 0.5):.4f}"
        }]
    }


# --- Agent 2: Audit Agent (SEC Qualitative RAG) ---
def audit_agent_node(state: MultiAgentState) -> dict[str, Any]:
    ticker = state["ticker"]
    logger.info(f"[AuditAgent] Invoking query_sec_rag_tool for {ticker}...")
    
    # Execute tool
    docs = query_sec_rag_tool(
        ticker, query="supply chain litigation regulatory exposure"
    )
    
    return {
        "audit_report": {"documents": docs, "count": len(docs)},
        "messages": [{
            "role": "AuditAgent",
            "content": f"Executed SEC RAG tool: Retrieved {len(docs)} qualitative risk chunks."
        }]
    }


# --- Agent 3: Market Sentiment Agent ---
def sentiment_agent_node(state: MultiAgentState) -> dict[str, Any]:
    ticker = state["ticker"]
    logger.info(f"[SentimentAgent] Invoking search_market_sentiment_tool for {ticker}...")
    
    # Execute tool
    sentiment_data = search_market_sentiment_tool(ticker)
    
    return {
        "sentiment_report": sentiment_data,
        "messages": [{
            "role": "SentimentAgent",
            "content": (
                f"Searched live news: {sentiment_data.get('result_count', 0)} results; "
                f"sentiment = {sentiment_data.get('sentiment_label', 'UNAVAILABLE')} "
                "(not scored by the search tool)."
            )
        }]
    }


# --- Lead Orchestrator: Chief Risk Officer (CRO) ---
def lead_orchestrator_node(state: MultiAgentState) -> dict[str, Any]:
    """Synthesizes inputs across all three worker agents (Quant + Audit + Sentiment)."""
    logger.info("[LeadOrchestrator] Synthesizing reports from Quant, Audit, and Sentiment agents...")
    
    quant = state.get("quant_report") or {}
    sentiment = state.get("sentiment_report") or {}
    
    quant_prob = quant.get("risk_probability", 0.5)
    if not isinstance(quant_prob, (int, float)) or not isfinite(quant_prob):
        quant_prob = 0.5
    quant_prob = min(1.0, max(0.0, float(quant_prob)))

    raw_sentiment_score = sentiment.get("sentiment_score")
    # Search and retrieved disclosure counts are evidence, not validated risk
    # scores. Only include sentiment if a separate scorer supplies a real value.
    weighted_score = quant_prob * 0.5
    available_weight = 0.5
    if (
        isinstance(raw_sentiment_score, (int, float))
        and isfinite(raw_sentiment_score)
    ):
        sentiment_risk_factor = min(1.0, max(0.0, -float(raw_sentiment_score)))
        weighted_score += sentiment_risk_factor * 0.25
        available_weight += 0.25

    composite_score = round(weighted_score / available_weight, 4)
    
    if composite_score >= 0.70:
        risk_level = "CRITICAL"
        requires_human = True
    elif composite_score >= 0.50:
        risk_level = "HIGH"
        requires_human = True
    elif composite_score >= 0.30:
        risk_level = "MEDIUM"
        requires_human = False
    else:
        risk_level = "LOW"
        requires_human = False

    return {
        "composite_risk_score": composite_score,
        "risk_level": risk_level,
        "human_approval_required": requires_human,
        "messages": [{
            "role": "LeadOrchestrator",
            "content": f"Synthesis Complete | Composite Score: {composite_score} | Level: {risk_level} | Require Human: {requires_human}"
        }]
    }


# --- Human Governance Node ---
def human_governance_node(state: MultiAgentState) -> dict[str, Any]:
    decision = interrupt({
        "kind": "human_approval",
        "ticker": state["ticker"],
        "risk_level": state["risk_level"],
        "risk_score": state["composite_risk_score"],
        "prompt": "Approve proceeding? Resume with {'approved': true|false}.",
    })
    if isinstance(decision, bool):
        approved = decision
        reason = None if approved else "Rejected by human reviewer."
    elif isinstance(decision, dict):
        approved = decision.get("approved") is True
        reason = decision.get("reason")
        if not approved and not reason:
            reason = "Rejected by human reviewer."
    else:
        approved = False
        reason = "Invalid approval response; execution denied."

    return {
        "human_approved": approved,
        "rejection_reason": reason,
        "messages": [{
            "role": "GovernanceNode",
            "content": (
                "Human reviewer approved execution."
                if approved
                else f"Human reviewer rejected execution: {reason}"
            ),
        }]
    }


# --- Execution Node ---
def execution_node(state: MultiAgentState) -> dict[str, Any]:
    approved = (
        state.get("human_approved") is True
        if state.get("human_approval_required")
        else True
    )
    
    if approved:
        status = (
            "APPROVED_FOR_EXECUTION"
            if state.get("human_approval_required")
            else "READY_FOR_EXECUTION"
        )
    else:
        status = f"BLOCKED_BY_REVIEW: {state.get('rejection_reason', 'Risk threshold exceeded')}"
        
    final_verdict = {
        "ticker": state["ticker"],
        "final_status": status,
        "composite_risk_score": state["composite_risk_score"],
        "risk_score_basis": (
            "Quantitative model probability, optionally combined with a separately "
            "scored sentiment value. Retrieved filings and unscored search results "
            "are contextual evidence only."
        ),
        "risk_level": state["risk_level"],
        "agent_outputs": {
            "quant_ml_metrics": state.get("quant_report"),
            "audit_sec_context": state.get("audit_report"),
            "market_sentiment": state.get("sentiment_report")
        }
    }
    
    return {"final_verdict": final_verdict}
