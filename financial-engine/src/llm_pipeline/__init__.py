"""Document ingestion and retrieval without eagerly loading ML runtimes."""

from typing import Any

__all__ = ["prepare_document_chunks", "VectorStoreManager", "FinancialRAGEngine"]


def __getattr__(name: str) -> Any:
    """Load heavier pipeline components only when they are requested."""
    if name == "prepare_document_chunks":
        from .ingest_docs import prepare_document_chunks

        return prepare_document_chunks
    if name == "VectorStoreManager":
        from .vector_store import VectorStoreManager

        return VectorStoreManager
    if name == "FinancialRAGEngine":
        from .rag_engine import FinancialRAGEngine

        return FinancialRAGEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
