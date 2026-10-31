import sys
from pathlib import Path

# Allow direct execution from the tests directory as well as from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_pipeline.rag_engine import FinancialRAGEngine

def test_rag_retrieval():
    rag = FinancialRAGEngine()
    try:
        print("=== Test 1: NVDA Export Control Query ===")
        results_nvda = rag.retrieve_context(
            "GPU export constraints and regulations", ticker="NVDA"
        )
        for doc in results_nvda:
            print(f"Score: {doc['score']} | {doc['doc_id']} | Text: {doc['text']}")

        print("\n=== Test 2: AAPL Supply Chain Query ===")
        results_aapl = rag.retrieve_context(
            "semiconductor component shortages", ticker="AAPL"
        )
        for doc in results_aapl:
            print(f"Score: {doc['score']} | {doc['doc_id']} | Text: {doc['text']}")
    finally:
        rag.close()

if __name__ == "__main__":
    rag = FinancialRAGEngine()
    try:
        print("=== Test 1: NVDA Export Control Query ===")
        results_nvda = rag.retrieve_context(
            "GPU export constraints and regulations", ticker="NVDA"
        )
        for doc in results_nvda:
            print(f"Score: {doc['score']} | {doc['doc_id']} | Text: {doc['text']}")

        print("\n=== Test 2: AAPL Supply Chain Query ===")
        results_aapl = rag.retrieve_context(
            "semiconductor component shortages", ticker="AAPL"
        )
        for doc in results_aapl:
            print(f"Score: {doc['score']} | {doc['doc_id']} | Text: {doc['text']}")
    finally:
        rag.close()
