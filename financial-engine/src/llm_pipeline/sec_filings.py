"""Fetch recent public SEC filings for a ticker and index them in Qdrant."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from src.llm_pipeline.ingest_docs import chunk_documents, load_documents


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEC_CACHE_DIR = PROJECT_ROOT / "data" / "sec_filings"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
CACHE_MAX_AGE = timedelta(days=7)
REQUEST_INTERVAL_SECONDS = 0.12

_request_lock = threading.Lock()
_last_request_at = 0.0


class SECFilingError(RuntimeError):
    """Raised when ticker lookup or public filing retrieval fails."""


def _user_agent() -> str:
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent or "your-email" in user_agent.lower():
        raise SECFilingError(
            "Set SEC_USER_AGENT in financial-engine/.env to an app name and "
            "contact email, for example 'FinancialRiskDashboard name@example.com'."
        )
    return user_agent


def _get(session: requests.Session, url: str) -> requests.Response:
    global _last_request_at
    with _request_lock:
        delay = REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if delay > 0:
            time.sleep(delay)
        response = session.get(url, timeout=(5, 30))
        _last_request_at = time.monotonic()
    if response.status_code == 403:
        raise SECFilingError(
            "The SEC denied the request. Check SEC_USER_AGENT and try again later."
        )
    response.raise_for_status()
    return response


@lru_cache(maxsize=4)
def _ticker_map(user_agent: str) -> dict[str, dict[str, Any]]:
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        })
        payload = _get(session, TICKER_MAP_URL).json()

    mapping: dict[str, dict[str, Any]] = {}
    for record in payload.values():
        ticker = str(record.get("ticker", "")).strip().upper()
        if ticker:
            mapping[ticker] = record
    return mapping


def _resolve_company(ticker: str, user_agent: str) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,11}", normalized):
        raise SECFilingError("Enter a valid ticker symbol.")

    mapping = _ticker_map(user_agent)
    company = mapping.get(normalized)
    if company is None:
        compact = re.sub(r"[^A-Z0-9]", "", normalized)
        matches = [
            record
            for key, record in mapping.items()
            if re.sub(r"[^A-Z0-9]", "", key) == compact
        ]
        if len(matches) == 1:
            company = matches[0]
    if company is None:
        raise SECFilingError(
            f"SEC EDGAR does not list a public filer for ticker {normalized}."
        )
    return company


def _recent_filings(submissions: dict[str, Any]) -> list[dict[str, str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filings = []
    selected_forms: set[str] = set()
    categories = (
        {"10-K", "20-F", "40-F"},
        {"10-Q", "6-K"},
    )
    for category in categories:
        for index, form in enumerate(forms):
            if form not in category or form in selected_forms:
                continue
            accession = recent.get("accessionNumber", [])[index]
            primary = recent.get("primaryDocument", [])[index]
            filed = recent.get("filingDate", [])[index]
            if not accession or not primary or "/" in primary or "\\" in primary:
                continue
            filings.append({
                "accession": accession,
                "form": form,
                "filing_date": filed,
                "primary_document": primary,
            })
            selected_forms.add(form)
            break
    return filings


def _manifest_path(ticker: str) -> Path:
    return SEC_CACHE_DIR / ticker / "manifest.json"


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_is_fresh(manifest: dict[str, Any] | None) -> bool:
    if not manifest:
        return False
    try:
        checked_at = datetime.fromisoformat(manifest["checked_at"])
        return datetime.now(timezone.utc) - checked_at < CACHE_MAX_AGE
    except (KeyError, TypeError, ValueError):
        return False


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def _download_filing(
    session: requests.Session,
    cik: int,
    ticker: str,
    filing: dict[str, str],
) -> dict[str, str]:
    accession_directory = filing["accession"].replace("-", "")
    document_name = quote(filing["primary_document"], safe=".-_")
    source_url = (
        f"{ARCHIVES_BASE_URL}/{cik}/{accession_directory}/{document_name}"
    )
    response = _get(session, source_url)

    ticker_dir = SEC_CACHE_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{ticker}-{filing['form']}-{filing['filing_date']}-"
        f"{filing['accession']}.html"
    )
    destination = ticker_dir / filename
    destination.write_bytes(response.content)
    return {
        **filing,
        "filename": filename,
        "source_url": source_url,
    }


def _documents_from_entries(
    ticker: str,
    entries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    documents = []
    for entry in entries:
        path = SEC_CACHE_DIR / ticker / entry["filename"]
        if not path.is_file():
            continue
        loaded = load_documents(path)
        for document in loaded:
            document.update({
                "doc_id": f"SEC:{ticker}:{entry['accession']}",
                "source_doc_id": entry["accession"],
                "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_url": entry["source_url"],
                "filing_date": entry["filing_date"],
                "form": entry["form"],
                "section": f"{entry['form']} filed {entry['filing_date']}",
            })
            documents.append(document)
    return documents


def sync_sec_filings(
    ticker: str,
    *,
    already_indexed: bool,
    encoder=None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Cache and index the latest annual/interim SEC filings for a ticker."""
    normalized_ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,11}", normalized_ticker):
        raise SECFilingError("Enter a valid ticker symbol.")
    user_agent = _user_agent()
    manifest_path = _manifest_path(normalized_ticker)
    manifest = _read_manifest(manifest_path)
    entries = manifest.get("filings", []) if manifest else []
    cached_files_exist = bool(entries) and all(
        (SEC_CACHE_DIR / normalized_ticker / entry.get("filename", "")).is_file()
        for entry in entries
    )

    if not force_refresh and _manifest_is_fresh(manifest) and cached_files_exist:
        if already_indexed:
            return {
                "company_name": manifest.get("company_name", normalized_ticker),
                "filings": entries,
                "indexed_chunks": 0,
                "from_cache": True,
            }
        documents = _documents_from_entries(normalized_ticker, entries)
        return _index_documents(
            normalized_ticker,
            manifest.get("company_name", normalized_ticker),
            entries,
            documents,
            encoder,
            from_cache=True,
        )

    company = _resolve_company(normalized_ticker, user_agent)
    cik = int(company["cik_str"])
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html",
    }
    with requests.Session() as session:
        session.headers.update(headers)
        submissions = _get(
            session,
            SUBMISSIONS_URL.format(cik=f"{cik:010d}"),
        ).json()
        latest = _recent_filings(submissions)
        if not latest:
            raise SECFilingError(
                f"No recent 10-K, 10-Q, 20-F, 40-F, or 6-K filing was found for {normalized_ticker}."
            )

        old_by_accession = {entry.get("accession"): entry for entry in entries}
        refreshed_entries = []
        new_entries = []
        for filing in latest:
            old_entry = old_by_accession.get(filing["accession"])
            old_path = (
                SEC_CACHE_DIR / normalized_ticker / old_entry.get("filename", "")
                if old_entry
                else None
            )
            if old_entry and old_path and old_path.is_file():
                entry = {**old_entry, **filing}
            else:
                entry = _download_filing(session, cik, normalized_ticker, filing)
                new_entries.append(entry)
            refreshed_entries.append(entry)

    company_name = str(company.get("title") or normalized_ticker)
    new_manifest = {
        "ticker": normalized_ticker,
        "cik": cik,
        "company_name": company_name,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filings": refreshed_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, new_manifest)

    if already_indexed and not new_entries:
        return {
            "company_name": company_name,
            "filings": refreshed_entries,
            "indexed_chunks": 0,
            "from_cache": False,
        }

    entries_to_index = refreshed_entries if not already_indexed else new_entries
    documents = _documents_from_entries(normalized_ticker, entries_to_index)
    return _index_documents(
        normalized_ticker,
        company_name,
        entries_to_index,
        documents,
        encoder,
        from_cache=False,
    )


def _index_documents(
    ticker: str,
    company_name: str,
    entries: list[dict[str, str]],
    documents: list[dict[str, Any]],
    encoder,
    *,
    from_cache: bool,
) -> dict[str, Any]:
    from src.llm_pipeline.vector_store import VectorStoreManager

    if not documents:
        raise SECFilingError(f"SEC returned no readable filing text for {ticker}.")
    chunks = chunk_documents(documents, chunk_size=900, chunk_overlap=120)
    with VectorStoreManager(encoder=encoder) as vector_store:
        indexed_chunks = vector_store.index_chunks(chunks)
    logger.info(
        "Indexed %d SEC filing chunks for %s from %d filings.",
        indexed_chunks,
        ticker,
        len(entries),
    )
    return {
        "company_name": company_name,
        "filings": entries,
        "indexed_chunks": indexed_chunks,
        "from_cache": from_cache,
    }
