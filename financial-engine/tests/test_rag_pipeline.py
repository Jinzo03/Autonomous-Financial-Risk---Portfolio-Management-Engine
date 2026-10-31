"""Small end-to-end checks for document indexing and filtered retrieval."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_pipeline.ingest_docs import prepare_document_chunks
from src.llm_pipeline.rag_engine import FinancialRAGEngine
from src.llm_pipeline.vector_store import VectorStoreManager


def test_rag_retrieval() -> None:
    chunks = prepare_document_chunks()
    with TemporaryDirectory(prefix="financial-rag-test-") as storage_path:
        with VectorStoreManager(storage_path=storage_path) as vector_store:
            assert vector_store.index_chunks(chunks) == len(chunks)

        with FinancialRAGEngine(storage_path=storage_path) as rag:
            nvda_results = rag.retrieve_context(
                "GPU export constraints and regulations",
                ticker="nvda",
            )
            aapl_results = rag.retrieve_context(
                "semiconductor component shortages",
                ticker="AAPL",
            )

            assert nvda_results
            assert all(result["ticker"] == "NVDA" for result in nvda_results)
            assert nvda_results[0]["doc_id"] == "NVDA-2026-10K"
            assert aapl_results
            assert all(result["ticker"] == "AAPL" for result in aapl_results)
            assert "semiconductor" in aapl_results[0]["text"].lower()


if __name__ == "__main__":
    test_rag_retrieval()
    print("RAG indexing and retrieval checks passed.")
