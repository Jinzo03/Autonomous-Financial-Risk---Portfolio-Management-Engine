"""Stage 1 local dashboard for quantitative risk and supporting evidence."""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
import streamlit as st
from dotenv import load_dotenv

from src.agentic_pipeline.graph import create_multi_agent_graph
from src.llm_pipeline.config import COLLECTION_NAME, create_qdrant_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)
MLOPS_SERVICE_URL = os.getenv(
    "MLOPS_SERVICE_URL", "http://127.0.0.1:8000/predict_risk"
)
_mlops_url = urlsplit(MLOPS_SERVICE_URL)
_mlops_parent = _mlops_url.path.rsplit("/", 1)[0]
MLOPS_READY_URL = urlunsplit(
    (_mlops_url.scheme, _mlops_url.netloc, f"{_mlops_parent}/ready", "", "")
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_UI_URL = (
    MLFLOW_TRACKING_URI.rstrip("/")
    if MLFLOW_TRACKING_URI.startswith(("http://", "https://"))
    else None
)
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,11}$")

st.set_page_config(
    page_title="Financial Risk Assessment",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_compiled_graph():
    return create_multi_agent_graph()


@st.cache_data(ttl=15, show_spinner=False)
def check_services() -> dict[str, bool]:
    """Check that the prediction API and configured Qdrant collection respond."""
    try:
        response = requests.get(MLOPS_READY_URL, timeout=2)
        mlops_ok = response.status_code == 200
    except requests.RequestException:
        mlops_ok = False

    qdrant_client = None
    try:
        qdrant_client = create_qdrant_client()
        qdrant_ok = qdrant_client.collection_exists(COLLECTION_NAME)
    except Exception:
        logger.exception("Qdrant health check failed")
        qdrant_ok = False
    finally:
        if qdrant_client is not None:
            try:
                qdrant_client.close()
            except Exception:
                logger.exception("Failed to close Qdrant health-check client")

    if MLFLOW_UI_URL:
        try:
            response = requests.get(MLFLOW_UI_URL, timeout=2, allow_redirects=False)
            mlflow_ok = response.status_code in (200, 301, 302, 307, 308)
        except requests.RequestException:
            mlflow_ok = False
    else:
        mlflow_ok = False

    return {"mlops": mlops_ok, "qdrant": qdrant_ok, "mlflow": mlflow_ok}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


with st.sidebar:
    st.header("Service status")
    health = check_services()

    (st.success if health["mlops"] else st.error)(
        "Prediction API: online" if health["mlops"] else "Prediction API: unavailable"
    )
    (st.success if health["qdrant"] else st.error)(
        f"Filing index ({COLLECTION_NAME}): ready"
        if health["qdrant"]
        else "Filing index: unavailable"
    )
    if MLFLOW_UI_URL:
        if health["mlflow"]:
            st.markdown(f"[Open MLflow]({MLFLOW_UI_URL})")
        else:
            st.error("MLflow tracking server: unavailable")
    else:
        st.info("MLflow UI link unavailable: tracking URI is not HTTP.")

    st.caption("News search runs during an assessment. Search sentiment is not scored.")
    if st.button("Refresh service status", use_container_width=True):
        check_services.clear()
        st.rerun()


st.title("Financial Risk Assessment")
st.caption(
    "Stage 1 · Quantitative model output with filing and live-news evidence. "
    "This page reports decisions; it does not place trades."
)

with st.form("assessment_form", clear_on_submit=False):
    input_col, button_col = st.columns([3, 1])
    with input_col:
        ticker_input = st.text_input(
            "Ticker symbol",
            value="AAPL",
            max_chars=12,
            placeholder="AAPL, NVDA, NEXA, BRK.B",
        ).strip().upper()
    with button_col:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Run assessment", use_container_width=True)

if submitted:
    st.session_state.pop("active_snapshot", None)
    st.session_state.pop("active_next", None)
    if not TICKER_PATTERN.fullmatch(ticker_input):
        st.error("Enter a ticker using letters, numbers, dots, or hyphens (up to 12 characters).")
    else:
        thread_id = f"dashboard_{ticker_input}_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "ticker": ticker_input,
            "messages": [],
            "quant_report": None,
            "audit_report": None,
            "sentiment_report": None,
            "composite_risk_score": None,
            "risk_level": None,
            "human_approval_required": False,
            "human_approved": None,
            "rejection_reason": None,
            "final_verdict": None,
        }

        try:
            with st.spinner(f"Assessing {ticker_input}…"):
                graph = get_compiled_graph()
                graph.invoke(initial_state, config=config)
                snapshot = graph.get_state(config)
            st.session_state["active_snapshot"] = snapshot.values
            st.session_state["active_next"] = snapshot.next
            st.session_state["active_thread_id"] = thread_id
        except Exception as exc:
            logger.exception("Dashboard assessment failed for %s", ticker_input)
            st.error(f"Assessment failed: {exc}")


state = st.session_state.get("active_snapshot")
next_nodes = st.session_state.get("active_next", ())

if state:
    quant_report = state.get("quant_report") or {}
    audit_report = state.get("audit_report") or {}
    news_report = state.get("sentiment_report") or {}
    risk_level = state.get("risk_level") or "UNKNOWN"
    verdict = state.get("final_verdict") or {}

    if risk_level == "UNKNOWN":
        st.error(
            "Assessment is blocked because a valid quantitative model score is unavailable. "
            f"{quant_report.get('error') or 'Check the Prediction API and model registry.'}"
        )
    elif "human_governance" in next_nodes:
        st.warning(
            f"{risk_level} risk requires human review. This Stage 1 dashboard does not "
            "provide approval controls; no execution has occurred."
        )

    quant_score = quant_report.get("risk_probability")
    audit_docs = audit_report.get("documents") or []
    news_items = news_report.get("news") or []
    kpi_risk, kpi_model, kpi_filings, kpi_news = st.columns(4)
    kpi_risk.metric("Overall risk level", risk_level)
    kpi_model.metric(
        "Model risk probability",
        f"{quant_score:.1%}"
        if quant_report.get("status") == "available" and _is_number(quant_score)
        else "Unavailable",
    )
    kpi_filings.metric("Filing matches", len(audit_docs))
    kpi_news.metric("News results", len(news_items))

    score = state.get("composite_risk_score")
    if _is_number(score):
        st.caption(
            f"Assessment score: **{score:.4f}** · Based on the quantitative probability "
            "and a sentiment score only if one is supplied. Filing matches and current "
            "news results are supporting evidence, not scored risk inputs."
        )

    if quant_report.get("status") == "available":
        with st.expander("Model output and Feast features"):
            st.write(f"Model category: **{quant_report.get('risk_category', 'Unknown')}**")
            st.json(quant_report.get("features_used") or {})

    st.divider()
    filing_col, news_col = st.columns(2)

    with filing_col:
        st.subheader("Filing evidence")
        if audit_report.get("available") is False:
            st.error(f"Filing retrieval failed: {audit_report.get('error', 'Unknown error')}")
        elif not audit_docs:
            st.info(f"No indexed filing chunks matched `{state.get('ticker', 'ticker')}`.")
        else:
            for document in audit_docs:
                source = document.get("source_path") or document.get("doc_id") or "Unknown source"
                section = document.get("section") or "Unspecified section"
                page = document.get("page_number")
                location = f"Page {page}" if page is not None else section
                similarity = document.get("score")
                suffix = Path(str(source)).suffix.lstrip(".").upper() or "DOCUMENT"
                with st.container(border=True):
                    st.write(f"**{source}** · {location} · {suffix}")
                    if _is_number(similarity):
                        st.caption(
                            f"Retrieval similarity: {similarity:.3f} "
                            "(relevance signal, not model confidence)."
                        )
                    st.write(document.get("text") or "No text available for this chunk.")

    with news_col:
        st.subheader("Recent news")
        st.caption(
            f"Sentiment: **{news_report.get('sentiment_label', 'UNASSESSED')}**. "
            "Headlines are not sentiment scores."
        )
        if news_report.get("error"):
            st.error(f"News search failed: {news_report['error']}")
        elif not news_items:
            st.info(f"No recent news results found for {state.get('ticker', 'ticker')}.")
        else:
            for article in news_items:
                title = article.get("headline") or "Untitled result"
                url = article.get("url")
                with st.container(border=True):
                    st.write(f"**{title}**")
                    metadata = " · ".join(
                        str(value)
                        for value in (article.get("source"), article.get("published_at"))
                        if value
                    )
                    if metadata:
                        st.caption(metadata)
                    if article.get("snippet"):
                        st.write(article["snippet"])
                    if isinstance(url, str) and urlsplit(url).scheme in {"http", "https"}:
                        st.link_button("Open source", url)

    if verdict.get("final_status"):
        st.caption(f"Workflow status: `{verdict['final_status']}`")
