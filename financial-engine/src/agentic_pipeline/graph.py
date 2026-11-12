from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agentic_pipeline.agents import (
    audit_agent_node,
    execution_node,
    human_governance_node,
    lead_orchestrator_node,
    quant_agent_node,
    sentiment_agent_node,
)
from src.agentic_pipeline.state import MultiAgentState


def route_governance(
    state: MultiAgentState,
) -> Literal["human_governance", "execution"]:
    return "human_governance" if state["human_approval_required"] else "execution"


def create_multi_agent_graph(checkpointer=None):
    """Build the risk workflow; default checkpointing is for local demos only."""
    builder = StateGraph(MultiAgentState)
    builder.add_node("quant_agent", quant_agent_node)
    builder.add_node("audit_agent", audit_agent_node)
    builder.add_node("sentiment_agent", sentiment_agent_node)
    builder.add_node("lead_orchestrator", lead_orchestrator_node)
    builder.add_node("human_governance", human_governance_node)
    builder.add_node("execution", execution_node)

    # Independent data sources can be queried concurrently.
    for worker in ("quant_agent", "audit_agent", "sentiment_agent"):
        builder.add_edge(START, worker)
        builder.add_edge(worker, "lead_orchestrator")

    builder.add_conditional_edges(
        "lead_orchestrator",
        route_governance,
        {
            "human_governance": "human_governance",
            "execution": "execution",
        },
    )
    builder.add_edge("human_governance", "execution")
    builder.add_edge("execution", END)

    # InMemorySaver loses all checkpoints when the process exits; production
    # deployments should inject a durable saver such as PostgresSaver.
    if checkpointer is None:
        checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Retain the old factory name used by the initial runner.
build_multi_agent_system = create_multi_agent_graph
