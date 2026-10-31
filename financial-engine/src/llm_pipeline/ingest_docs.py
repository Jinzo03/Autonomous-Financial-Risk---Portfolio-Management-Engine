import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_mock_sec_filings() -> List[Dict[str, Any]]:
    """Generates sample unstructured financial disclosures."""
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

def prepare_document_chunks(chunk_size: int = 250, chunk_overlap: int = 40) -> List[Dict[str, Any]]:
    """Splits raw text into metadata-rich chunks for vector embedding."""
    docs = generate_mock_sec_filings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunked_records = []
    for doc in docs:
        splits = text_splitter.split_text(doc["content"])
        for idx, split_text in enumerate(splits):
            chunked_records.append({
                "chunk_id": f"{doc['doc_id']}_chunk_{idx}",
                "ticker": doc["ticker"],
                "doc_id": doc["doc_id"],
                "section": doc["section"],
                "text": split_text
            })
            
    print(f"[LLMOps] Produced {len(chunked_records)} text chunks from {len(docs)} input documents.")
    return chunked_records

if __name__ == "__main__":
    chunks = prepare_document_chunks()
    print("Sample Chunk Payload:", json.dumps(chunks[0], indent=2))