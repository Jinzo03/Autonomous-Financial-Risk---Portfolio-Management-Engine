import json
import logging
from src.agentic_pipeline.graph import create_multi_agent_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_agentic_workflow():
    app = create_multi_agent_graph()
    
    # Unique thread context for state checkpointing
    thread_config = {"configurable": {"thread_id": "session_portfolio_001"}}
    
    initial_state = {
        "ticker": "AAPL",
        "thread_id": "session_portfolio_001",
        "messages": [],
        "quant_report": None,
        "qual_report": None,
        "risk_level": None,
        "risk_score": 0.0,
        "human_approval_required": False,
        "human_approved": None,
        "rejection_reason": None,
        "final_verdict": None
    }
    
    print("\n--- PHASE 1: STARTING MULTI-AGENT PIPELINE ---")
    # Execute workflow until complete or paused by HITL interrupt
    for event in app.stream(initial_state, config=thread_config):
        for node_name, node_state in event.items():
            print(f"\n[Node Executed]: {node_name}")
            if "messages" in node_state and node_state["messages"]:
                latest_msg = node_state["messages"][-1]
                print(f"  └─ Message ({latest_msg['role']}): {latest_msg['content']}")
                
    # Check current thread state in MemorySaver checkpointer
    current_state = app.get_state(thread_config)
    
    if current_state.next and "human_governance" in current_state.next:
        print("\n========================================================")
        print(" [HUMAN-IN-THE-LOOP INTERRUPT TRIGGERED]")
        print(f" Thread ID: {thread_config['configurable']['thread_id']}")
        print(f" Current Risk Level: {current_state.values.get('risk_level')}")
        print(f" Composite Risk Score: {current_state.values.get('risk_score')}")
        print(" Execution held for human compliance review.")
        print("========================================================")
        
        # Simulate Human Manager Input (Approve or Reject)
        human_decision = input("\nApprove trade execution? (yes/no): ").strip().lower()
        
        is_approved = human_decision == "yes"
        rejection_reason = None if is_approved else "Human Risk Officer rejected high volatility allocation."
        
        # Resume workflow by updating state checkpoint
        app.update_state(
            thread_config,
            {
                "human_approved": is_approved,
                "rejection_reason": rejection_reason
            },
            as_node="human_governance"
        )
        
        print("\n--- PHASE 2: RESUMING WORKFLOW FROM CHECKPOINT ---")
        for event in app.stream(None, config=thread_config):
            for node_name, node_state in event.items():
                print(f"\n[Node Executed]: {node_name}")
                
    # Fetch final summary
    final_state = app.get_state(thread_config).values
    print("\n========================================================")
    print("         FINAL MULTI-AGENT VERDICT & AUDIT TRAIL        ")
    print("========================================================")
    print(json.dumps(final_state.get("final_verdict"), indent=2))

if __name__ == "__main__":
    run_agentic_workflow()