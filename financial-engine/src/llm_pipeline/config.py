"""Shared settings and Qdrant client construction for the retrieval pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parents[1]
load_dotenv(PROJECT_ROOT / ".env")

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.6-flash,gemini-2.5-flash",
    ).split(",")
    if model.strip()
)
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_OUTPUT_TOKENS = int(
    os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")
)


def create_qdrant_client(storage_path: str | os.PathLike = DEFAULT_STORAGE_PATH) -> QdrantClient:
    """Use Qdrant server when configured, otherwise use local embedded storage."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    path = Path(storage_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def create_gemini_client():
    """Create a Gemini client only when answer generation is requested."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your key to financial-engine/.env."
        )

    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)
