"""Metadata-filtered semantic retrieval over the local or remote Qdrant index."""

import logging
import os
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

if __package__:
    from .config import (
        COLLECTION_NAME,
        DEFAULT_STORAGE_PATH,
        EMBEDDING_MODEL_NAME,
        create_qdrant_client,
    )
else:  # Support `python rag_engine.py` from this directory.
    from config import (
        COLLECTION_NAME,
        DEFAULT_STORAGE_PATH,
        EMBEDDING_MODEL_NAME,
        create_qdrant_client,
    )


logger = logging.getLogger(__name__)


class FinancialRAGEngine:
    def __init__(
        self,
        storage_path: str | os.PathLike = DEFAULT_STORAGE_PATH,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.client = create_qdrant_client(storage_path)
        try:
            if not self.client.collection_exists(COLLECTION_NAME):
                raise RuntimeError(
                    f"Qdrant collection {COLLECTION_NAME!r} does not exist. "
                    "Run vector_store.py to index the documents first."
                )

            self.encoder = SentenceTransformer(embedding_model)
            vector_size = self.encoder.get_embedding_dimension()
            vectors = self.client.get_collection(COLLECTION_NAME).config.params.vectors
            if isinstance(vectors, dict) or vector_size != vectors.size:
                raise ValueError(
                    f"Embedding model {embedding_model!r} does not match the vector "
                    f"configuration of collection {COLLECTION_NAME!r}. Re-index using "
                    "the same embedding model and collection settings."
                )
            if vectors.distance.value != "Cosine":
                raise ValueError(
                    f"Collection {COLLECTION_NAME!r} must use cosine distance."
                )
        except Exception:
            self.client.close()
            raise

    def retrieve_context(
        self,
        query: str,
        ticker: str | None = None,
        top_k: int = 2,
    ) -> list[dict[str, Any]]:
        """Return the most relevant chunks, optionally filtered by ticker."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query_vector = self.encoder.encode_query(
            query.strip(),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        query_filter = None
        if ticker is not None:
            normalized_ticker = ticker.strip().upper()
            if not normalized_ticker:
                raise ValueError("ticker must be non-empty when provided")
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="ticker",
                        match=MatchValue(value=normalized_ticker),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=[
                "chunk_id",
                "chunk_index",
                "start_index",
                "source_doc_id",
                "source_path",
                "page_number",
                "row_number",
                "ticker",
                "doc_id",
                "section",
                "text",
            ],
        )

        return [
            {
                "score": float(point.score),
                **(point.payload or {}),
            }
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "FinancialRAGEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    with FinancialRAGEngine() as rag:
        results = rag.retrieve_context(
            query="What supply chain or component risks exist?",
            ticker="AAPL",
            top_k=2,
        )
        for result in results:
            print(
                f"\n[Score: {result['score']:.4f}] "
                f"{result.get('ticker')} ({result.get('doc_id')}):"
            )
            print(f"Text: {result.get('text', '')}")
