"""Index text chunks into Qdrant using Sentence Transformers embeddings."""

import logging
import os
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

if __package__:
    from .config import (
        COLLECTION_NAME,
        DEFAULT_STORAGE_PATH,
        EMBEDDING_MODEL_NAME,
        create_qdrant_client,
    )
else:  # Support `python vector_store.py` from this directory.
    from config import (
        COLLECTION_NAME,
        DEFAULT_STORAGE_PATH,
        EMBEDDING_MODEL_NAME,
        create_qdrant_client,
    )


logger = logging.getLogger(__name__)
INDEX_BATCH_SIZE = 64


class VectorStoreManager:
    def __init__(
        self,
        storage_path: str | os.PathLike = DEFAULT_STORAGE_PATH,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.embedding_model = embedding_model
        self.client = create_qdrant_client(storage_path)
        try:
            self.encoder = SentenceTransformer(self.embedding_model)
            self.vector_size = self.encoder.get_embedding_dimension()
            if self.vector_size is None:
                raise ValueError(
                    f"Could not determine embedding dimensions for {embedding_model!r}"
                )
            self._init_collection()
        except Exception:
            self.client.close()
            raise

    def _init_collection(self) -> None:
        """Create the collection or verify it matches this embedding model."""
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", COLLECTION_NAME)
            return

        info = self.client.get_collection(COLLECTION_NAME)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} uses named vectors; this pipeline "
                "expects one unnamed vector. Configure a separate collection."
            )
        if vectors.size != self.vector_size:
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} has {vectors.size}-dimensional vectors, "
                f"but model {self.embedding_model!r} produces {self.vector_size}. "
                "Use a new collection or rebuild the index with the selected model."
            )
        if vectors.distance != Distance.COSINE:
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} must use cosine distance."
            )

    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Upsert chunks with deterministic point IDs; return indexed count."""
        if not chunks:
            logger.info("No chunks to index.")
            return 0

        required_fields = {"chunk_id", "ticker", "doc_id", "section", "text"}
        for index, chunk in enumerate(chunks):
            missing = required_fields.difference(chunk)
            if missing:
                raise ValueError(
                    f"Chunk {index} is missing required fields: {sorted(missing)}"
                )
            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                raise ValueError(f"Chunk {index} has empty or invalid text")

        for start in range(0, len(chunks), INDEX_BATCH_SIZE):
            batch = chunks[start : start + INDEX_BATCH_SIZE]
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.encoder.encode_document(
                texts,
                batch_size=INDEX_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            points = []
            for chunk, embedding in zip(batch, embeddings, strict=True):
                point_id = str(uuid5(NAMESPACE_URL, chunk["chunk_id"]))
                payload = {
                    key: chunk[key]
                    for key in (
                        "chunk_id",
                        "chunk_index",
                        "start_index",
                        "ticker",
                        "doc_id",
                        "section",
                        "text",
                    )
                    if key in chunk
                }
                payload["ticker"] = str(chunk["ticker"]).strip().upper()
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

        logger.info("Indexed %d chunks into Qdrant collection %s", len(chunks), COLLECTION_NAME)
        return len(chunks)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "VectorStoreManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _load_chunks() -> list[dict[str, Any]]:
    if __package__:
        from .ingest_docs import prepare_document_chunks
    else:
        from ingest_docs import prepare_document_chunks

    return prepare_document_chunks()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chunks = _load_chunks()
    with VectorStoreManager() as vector_store:
        vector_store.index_chunks(chunks)
