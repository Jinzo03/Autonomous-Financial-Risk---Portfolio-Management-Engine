import os
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "financial_unstructured_docs"
VECTOR_DIM = 384  # Dimension for all-MiniLM-L6-v2
DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent / "data" / "qdrant_db"

class VectorStoreManager:
    def __init__(self, storage_path: str | os.PathLike = DEFAULT_STORAGE_PATH):
        os.makedirs(storage_path, exist_ok=True)
        self.client = QdrantClient(path=storage_path)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_collection()

    def _init_collection(self):
        """Creates vector collection if not present."""
        collections = [col.name for col in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' created.")

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Embeds text chunks and writes vectors with payloads to Qdrant."""
        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "ticker": chunk["ticker"],
                        "doc_id": chunk["doc_id"],
                        "section": chunk["section"],
                        "text": chunk["text"]
                    }
                )
            )

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[Qdrant] Indexed {len(points)} vector records into '{COLLECTION_NAME}'.")

    def close(self):
        """Close the local Qdrant client cleanly."""
        self.client.close()

if __name__ == "__main__":
    # Support both `python vector_store.py` from this directory and package
    # execution such as `python -m src.llm_pipeline.vector_store`.
    if __package__:
        from .ingest_docs import prepare_document_chunks
    else:
        from ingest_docs import prepare_document_chunks
    
    chunks = prepare_document_chunks()
    vm = VectorStoreManager()
    try:
        vm.index_chunks(chunks)
    finally:
        vm.close()
