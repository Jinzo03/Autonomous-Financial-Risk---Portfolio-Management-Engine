from src.llm_pipeline.rag_engine import FinancialRAGEngine

def test_rag_retrieval():
    rag = FinancialRAGEngine()
    
    print("=== Test 1: NVDA Export Control Query ===")
    results_nvda = rag.retrieve_context("GPU export constraints and regulations", ticker="NVDA")
    for doc in results_nvda:
        print(f"Score: {doc['score']} | {doc['doc_id']} | Text: {doc['text']}")
        
    print("\n=== Test 2: AAPL Supply Chain Query ===")
    results_aapl = rag.retrieve_context("semiconductor component shortages", ticker="AAPL")
    for doc in results_aapl:
        print(f"Score: {doc['score']} | {doc['doc_id']} | Text: {doc['text']}")

if __name__ == "__main__":
    test_rag_retrieval()