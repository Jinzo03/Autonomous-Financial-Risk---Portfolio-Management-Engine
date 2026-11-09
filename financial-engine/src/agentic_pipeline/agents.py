import logging
from typing import Dict, Any
from src.agentic_pipeline.state import MultiAgentState
from src.agentic_pipeline.tools import (
    get_quantitative_risk_tool,
    query_sec_rag_tool,
    search_market_sentiment_tool
)

logger = logging.getLogger(__name__)

# --- Agent 1: Quantitative Risk Agent ---
def quant_agent_node(state: MultiAgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    logger.info(f"[QuantAgent] Invoking get_quantitative_risk_tool for {ticker}...")
    
    # Execute tool
    tool_output = get_quantitative_risk_tool.invoke({"ticker": ticker})
    
    return {
        "quant_report": tool_output,
        "messages": [{
            "role": "QuantAgent",
            "content": f"Executed quantitative tool: Risk Score = {tool_output.get('risk_probability', 0.5):.4f}"
        }]
    }


# --- Agent 2: Audit Agent (SEC Qualitative RAG) ---
def audit_agent_node(state: MultiAgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    logger.info(f"[AuditAgent] Invoking query_sec_rag_tool for {ticker}...")
    
    # Execute tool
    docs = query_sec_rag_tool.invoke({"ticker": ticker, "query": "supply chain litigation regulatory exposure"})
    
    return {
        "audit_report": {"documents": docs, "count": len(docs)},
        "messages": [{
            "role": "AuditAgent",
            "content": f"Executed SEC RAG tool: Retrieved {len(docs)} qualitative risk chunks."
        }]
    }


# --- Agent 3: Market Sentiment Agent ---
def sentiment_agent_node(state: MultiAgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    logger.info(f"[SentimentAgent] Invoking search_market_sentiment_tool for {ticker}...")
    
    # Execute tool
    sentiment_data = search_market_sentiment_tool.invoke({"ticker": ticker})
    
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
def lead_orchestrator_node(state: MultiAgentState) -> Dict[str, Any]:
    """Synthesizes inputs across all three worker agents (Quant + Audit + Sentiment)."""
    logger.info("[LeadOrchestrator] Synthesizing reports from Quant, Audit, and Sentiment agents...")
    
    quant = state.get("quant_report") or {}
    audit = state.get("audit_report") or {}
    sentiment = state.get("sentiment_report") or {}
    
    quant_prob = quant.get("risk_probability", 0.5)
    audit_docs_count = audit.get("count", 0)
    raw_sentiment_score = sentiment.get("sentiment_score")
    # Live search returns articles, not a validated sentiment score.
    sentiment_score = (
        float(raw_sentiment_score)
        if isinstance(raw_sentiment_score, (int, float))
        else 0.0
    )
    
    # Multi-Factor Weighted Risk Calculation across all 3 agent outputs:
    # 50% Quant ML Score + 25% SEC Audit Disclosures + 25% Negative Market Sentiment
    sentiment_risk_factor = max(0.0, -sentiment_score)  # Convert negative sentiment to positive risk
    audit_risk_factor = min(1.0, audit_docs_count / 2.0)
    
    composite_score = (quant_prob * 0.50) + (audit_risk_factor * 0.25) + (sentiment_risk_factor * 0.25)
    composite_score = round(min(1.0, composite_score), 4)
    
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
def human_governance_node(state: MultiAgentState) -> Dict[str, Any]:
    return {
        "messages": [{
            "role": "GovernanceNode",
            "content": "Execution paused by Lead Orchestrator. Awaiting compliance approval."
        }]
    }


# --- Execution Node ---
def execution_node(state: MultiAgentState) -> Dict[str, Any]:
    approved = state.get("human_approved", True)
    
    if approved:
        status = "TRADE_EXECUTED_AND_LOGGED"
    else:
        status = f"TRADE_CANCELLED: {state.get('rejection_reason', 'Risk threshold exceeded')}"
        
    final_verdict = {
        "ticker": state["ticker"],
        "final_status": status,
        "composite_risk_score": state["composite_risk_score"],
        "risk_level": state["risk_level"],
        "agent_outputs": {
            "quant_ml_metrics": state.get("quant_report"),
            "audit_sec_context": state.get("audit_report"),
            "market_sentiment": state.get("sentiment_report")
        }
    }
    
    return {"final_verdict": final_verdict}
