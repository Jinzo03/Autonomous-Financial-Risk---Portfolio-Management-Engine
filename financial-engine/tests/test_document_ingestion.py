"""Format tests for the local document ingestion pipeline."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf

from src.llm_pipeline.ingest_docs import load_documents, prepare_document_chunks


class DocumentIngestionTests(unittest.TestCase):
    def test_loads_text_markdown_pdf_and_csv_with_metadata(self):
        with TemporaryDirectory(prefix="financial-doc-ingestion-") as directory:
            source_dir = Path(directory)
            (source_dir / "NVDA_report.txt").write_text(
                "GPU supply constraints may affect deliveries.", encoding="utf-8"
            )
            (source_dir / "AAPL_risks.md").write_text(
                "# Supply Chain\n\nComponent shortages may delay shipments.",
                encoding="utf-8",
            )
            (source_dir / "risks.csv").write_text(
                "ticker,doc_id,section,description\n"
                "MSFT,msft-risk-1,Operations,Cloud infrastructure costs may rise.\n",
                encoding="utf-8",
            )
            pdf_path = source_dir / "GOOGL_filing.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text(
                    (72, 72),
                    "Data center energy demand may increase operating costs.",
                )
                pdf.save(pdf_path)

            documents = load_documents(source_dir)
            by_ticker = {document["ticker"]: document for document in documents}

            self.assertIn("GPU supply", by_ticker["NVDA"]["content"])
            self.assertEqual(by_ticker["AAPL"]["section"], "Supply Chain")
            self.assertIn("Cloud infrastructure", by_ticker["MSFT"]["content"])
            self.assertEqual(by_ticker["GOOGL"]["page_number"], 1)
            self.assertIn("Data center energy", by_ticker["GOOGL"]["content"])

            chunks = prepare_document_chunks(source_path=source_dir)
            pdf_chunk = next(
                chunk for chunk in chunks if chunk["ticker"] == "GOOGL"
            )
            self.assertEqual(pdf_chunk["page_number"], 1)
            self.assertEqual(pdf_chunk["source_path"], "GOOGL_filing.pdf")

    def test_rejects_unsupported_file_extension(self):
        with TemporaryDirectory(prefix="financial-doc-ingestion-") as directory:
            unsupported = Path(directory) / "notes.docx"
            unsupported.write_text("not supported", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported source format"):
                load_documents(unsupported)


if __name__ == "__main__":
    unittest.main(verbosity=2)
