from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agentic_pipeline.state import MultiAgentState
from src.agentic_pipeline.agents import (
    quant_agent_node,
    audit_agent_node,
    sentiment_agent_node,
    lead_orchestrator_node,
    human_governance_node,
    execution_node
)

def route_governance(state: MultiAgentState) -> str:
    if state.get("human_approval_required", False):
        return "human_governance"
    return "execution"

def build_multi_agent_system():
    workflow = StateGraph(MultiAgentState)
    
    # 1. Add Agent Nodes
    workflow.add_node("quant_agent", quant_agent_node)
    workflow.add_node("audit_agent", audit_agent_node)
    workflow.add_node("sentiment_agent", sentiment_agent_node)
    workflow.add_node("lead_orchestrator", lead_orchestrator_node)
    workflow.add_node("human_governance", human_governance_node)
    workflow.add_node("execution", execution_node)
    
    # 2. Parallel Worker Fan-out
    workflow.set_entry_point("quant_agent")
    workflow.add_edge("quant_agent", "audit_agent")
    workflow.add_edge("audit_agent", "sentiment_agent")
    
    # 3. Fan-in into Lead Orchestrator
    workflow.add_edge("sentiment_agent", "lead_orchestrator")
    
    # 4. Conditional Edge from Orchestrator based on Tool Synthesis Risk Score
    workflow.add_conditional_edges(
        "lead_orchestrator",
        route_governance,
        {
            "human_governance": "human_governance",
            "execution": "execution"
        }
    )
    
    workflow.add_edge("human_governance", "execution")
    workflow.add_edge("execution", END)
    
    # 5. Checkpointer for HITL Interrupts
    checkpointer = MemorySaver()
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_governance"]
    )