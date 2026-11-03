import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".csv"}
TEXT_COLUMNS = (
    "text",
    "content",
    "description",
    "risk_factors",
    "narrative",
    "summary",
    "notes",
)
METADATA_COLUMNS = {
    "ticker",
    "symbol",
    "doc_id",
    "document_id",
    "id",
    "section",
    "title",
    "type",
    "category",
}


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


def _filename_ticker(path: Path) -> str:
    """Use a ticker prefix in a filename when available, else mark unknown."""
    match = re.match(r"^([A-Z][A-Z0-9]{0,4})(?:[-_. ]|$)", path.stem.upper())
    return match.group(1) if match else "UNKNOWN"


def _relative_source(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _read_text_documents(path: Path, relative_path: str) -> list[dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        logger.warning("Reading %s as Windows-1252 after UTF-8 decoding failed", relative_path)
        content = path.read_text(encoding="cp1252").strip()
    if not content:
        return []

    if path.suffix.lower() not in {".md", ".markdown"}:
        return [{
            "ticker": _filename_ticker(path),
            "doc_id": relative_path,
            "source_doc_id": relative_path,
            "source_path": relative_path,
            "section": path.stem,
            "content": content,
        }]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )
    sections = markdown_splitter.split_text(content)
    documents = []
    for index, section_doc in enumerate(sections):
        headings = [
            section_doc.metadata[key]
            for key in ("h1", "h2", "h3", "h4")
            if section_doc.metadata.get(key)
        ]
        section = " > ".join(headings) if headings else path.stem
        text = section_doc.page_content.strip()
        if not text:
            continue
        documents.append({
            "ticker": _filename_ticker(path),
            "doc_id": f"{relative_path}#section:{index}",
            "source_doc_id": relative_path,
            "source_path": relative_path,
            "section": section,
            "content": text,
        })
    return documents


def _read_pdf_documents(path: Path, relative_path: str) -> list[dict[str, Any]]:
    # PyMuPDF preserves page boundaries and can sort extracted text into a
    # more natural top-to-bottom reading order.
    import pymupdf

    documents = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            content = page.get_text("text", sort=True).strip()
            if not content:
                logger.warning(
                    "No selectable text on %s page %d; scanned pages need OCR.",
                    relative_path,
                    page_number,
                )
                continue
            documents.append({
                "ticker": _filename_ticker(path),
                "doc_id": f"{relative_path}#page:{page_number}",
                "source_doc_id": relative_path,
                "source_path": relative_path,
                "section": f"Page {page_number}",
                "page_number": page_number,
                "content": content,
            })
    return documents


def _read_csv_documents(path: Path, relative_path: str) -> list[dict[str, Any]]:
    documents = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.DictReader(csv_file), start=1):
            normalized = {
                str(key).strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            if not any(normalized.values()):
                continue

            narrative_fields = [
                (key, normalized[key])
                for key in TEXT_COLUMNS
                if normalized.get(key)
            ]
            if narrative_fields:
                content = "\n".join(
                    f"{key.replace('_', ' ').title()}: {value}"
                    for key, value in narrative_fields
                )
            else:
                # Keep tabular rows separate and readable instead of flattening
                # an entire CSV into one long, ambiguous text block.
                content_fields = [
                    (key, value)
                    for key, value in normalized.items()
                    if value and key not in METADATA_COLUMNS
                ]
                content = "; ".join(
                    f"{key.replace('_', ' ')}: {value}"
                    for key, value in content_fields
                )
            if not content:
                continue

            ticker = (
                normalized.get("ticker")
                or normalized.get("symbol")
                or _filename_ticker(path)
            ).upper()
            source_doc_id = (
                normalized.get("doc_id")
                or normalized.get("document_id")
                or normalized.get("id")
                or f"row-{row_number}"
            )
            documents.append({
                "ticker": ticker,
                "doc_id": f"{relative_path}#{source_doc_id}",
                "source_doc_id": source_doc_id,
                "source_path": relative_path,
                "section": (
                    normalized.get("section")
                    or normalized.get("title")
                    or normalized.get("type")
                    or normalized.get("category")
                    or f"Row {row_number}"
                ),
                "row_number": row_number,
                "content": content,
            })
    return documents


def load_documents(source_path: str | Path) -> list[dict[str, Any]]:
    """Load supported UTF-8 text, Markdown, PDF, or CSV files from a path."""
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Document source does not exist: {source}")
    if source.is_file() and source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported source format {source.suffix!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    root = source if source.is_dir() else source.parent
    files = (
        sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if source.is_dir()
        else [source]
    )
    documents = []
    for path in files:
        relative_path = _relative_source(path, root)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            loaded = _read_text_documents(path, relative_path)
        elif suffix == ".pdf":
            loaded = _read_pdf_documents(path, relative_path)
        else:  # .csv
            loaded = _read_csv_documents(path, relative_path)
        documents.extend(loaded)

    if not documents:
        raise ValueError(f"No readable documents found under {source}")
    logger.info("Loaded %d source documents from %s", len(documents), source)
    return documents


def prepare_document_chunks(
    chunk_size: int = 250,
    chunk_overlap: int = 40,
    source_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load optional source files and split each document with provenance."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    docs = (
        load_documents(source_path)
        if source_path is not None
        else generate_mock_sec_filings()
    )
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
        for metadata_key in (
            "source_doc_id",
            "source_path",
            "page_number",
            "row_number",
        ):
            if metadata_key in doc:
                source_metadata[metadata_key] = doc[metadata_key]
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
        "Produced %d chunks from %d source documents.",
        len(chunked_records),
        len(docs),
    )
    return chunked_records

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_path",
        nargs="?",
        help="File or directory containing .txt, .md, .pdf, or .csv files. "
        "Omit to use synthetic demo documents.",
    )
    args = parser.parse_args()
    chunks = prepare_document_chunks(source_path=args.source_path)
    print("Sample Chunk Payload:", json.dumps(chunks[0], indent=2))
