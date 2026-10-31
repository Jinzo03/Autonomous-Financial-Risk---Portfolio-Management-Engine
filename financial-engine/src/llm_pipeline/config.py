"""Shared settings and Qdrant client construction for the retrieval pipeline."""

import os
from pathlib import Path

from qdrant_client import QdrantClient


PIPELINE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "financial_unstructured_docs_v2",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_STORAGE_PATH = Path(
    os.getenv(
        "QDRANT_STORAGE_PATH",
        str(PIPELINE_DIR / "data" / "qdrant_db"),
    )
).expanduser()
if not DEFAULT_STORAGE_PATH.is_absolute():
    DEFAULT_STORAGE_PATH = (PIPELINE_DIR / DEFAULT_STORAGE_PATH).resolve()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def create_qdrant_client(storage_path: str | os.PathLike = DEFAULT_STORAGE_PATH) -> QdrantClient:
    """Use Qdrant server when configured, otherwise use local embedded storage."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    path = Path(storage_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))
