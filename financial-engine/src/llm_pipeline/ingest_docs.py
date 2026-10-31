import json
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def generate_mock_sec_filings() -> list[dict[str, str]]:
    """Return synthetic disclosure examples for local development and demos."""
    return [
        {
            "ticker": "AAPL",
            "doc_id": "AAPL-2026-10K",
            "section": "Item 1A - Risk Factors",
            "content": (
                "Apple Inc. faces ongoing supply chain constraints in semiconductor sourcing. "
                "Global logistical bottlenecks and component shortages may increase production costs "
                "and delay device shipments. Furthermore, foreign exchange fluctuations continue to "
                "impact foreign net revenue gains."
            )
        },
        {
            "ticker": "NVDA",
            "doc_id": "NVDA-2026-10K",
            "section": "Item 1A - Risk Factors",
            "content": (
                "NVIDIA Corporation operations depend heavily on advanced wafer fabrication facilities. "
                "Export control regulations restricting high-performance GPU hardware shipments to foreign "
                "markets present a regulatory headwind that could impair quarterly enterprise revenue."
            )
        },
        {
            "ticker": "AAPL",
            "doc_id": "AAPL-2026-Q2-EARNINGS",
            "section": "Management Discussion",
            "content": (
                "Services revenue grew 14% year-over-year. However, regulatory headwinds regarding app store "
                "fee structures in foreign markets remain an active financial risk."
            )
        }
    ]

def prepare_document_chunks(
    chunk_size: int = 250,
    chunk_overlap: int = 40,
) -> list[dict[str, Any]]:
    """Split demo disclosures into chunks with stable IDs and source metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    docs = generate_mock_sec_filings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )

    chunked_records = []
    for doc in docs:
        source_metadata = {
            "ticker": doc["ticker"],
            "doc_id": doc["doc_id"],
            "section": doc["section"],
        }
        splits = text_splitter.create_documents(
            [doc["content"]],
            metadatas=[source_metadata],
        )
        for idx, split in enumerate(splits):
            chunked_records.append({
                "chunk_id": f"{doc['doc_id']}:chunk:{idx}",
                "chunk_index": idx,
                "start_index": split.metadata["start_index"],
                **split.metadata,
                "text": split.page_content,
            })

    logger.info(
        "Produced %d chunks from %d synthetic disclosure examples.",
        len(chunked_records),
        len(docs),
    )
    return chunked_records

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chunks = prepare_document_chunks()
    print("Sample Chunk Payload:", json.dumps(chunks[0], indent=2))
