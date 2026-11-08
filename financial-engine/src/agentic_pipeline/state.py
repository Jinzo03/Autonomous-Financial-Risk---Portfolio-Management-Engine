from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
import operator


class FinancialAgentState(TypedDict):
    ticker: str
    thread_id: str
    messages: Annotated[List[Dict[str, str]], operator.add]
    
    # Agent Outputs
    quant_report: Optional[Dict[str, Any]]
    qual_report: Optional[Dict[str, Any]]
    
    # Risk Assessment
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]]
    risk_score: float
    
    # Governance & Checkpointing
    human_approval_required: bool
    human_approved: Optional[bool]
    rejection_reason: Optional[str]
    
    final_verdict: Optional[Dict[str, Any]]