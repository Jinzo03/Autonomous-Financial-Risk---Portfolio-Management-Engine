"""Local dashboard for risk assessment, evidence review, and human decisions."""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from src.agentic_pipeline.graph import create_multi_agent_graph
from src.llm_pipeline.config import COLLECTION_NAME, create_qdrant_client
from src.ui.monitoring_store import get_assessment_history, save_assessment


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
MLFLOW_HEALTH_URL = (
    MLFLOW_TRACKING_URI.rstrip("/")
    if MLFLOW_TRACKING_URI.startswith(("http://", "https://"))
    else None
)
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL")
if MLFLOW_UI_URL is None:
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


@st.cache_resource
def get_rag_engine():
    from src.llm_pipeline.rag_engine import FinancialRAGEngine

    return FinancialRAGEngine()


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

    if MLFLOW_HEALTH_URL:
        try:
            response = requests.get(MLFLOW_HEALTH_URL, timeout=2, allow_redirects=False)
            mlflow_ok = response.status_code in (200, 301, 302, 307, 308)
        except requests.RequestException:
            mlflow_ok = False
    else:
        mlflow_ok = False

    return {"mlops": mlops_ok, "qdrant": qdrant_ok, "mlflow": mlflow_ok}


@st.cache_data(ttl=60, show_spinner=False)
def get_indexed_tickers() -> tuple[str, ...]:
    """Read distinct ticker metadata from the configured filing collection."""
    client = create_qdrant_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            return ()

        tickers: set[str] = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=["ticker"],
                with_vectors=False,
            )
            tickers.update(
                str(point.payload["ticker"]).strip().upper()
                for point in points
                if point.payload and point.payload.get("ticker")
            )
            if offset is None:
                break
        return tuple(sorted(tickers))
    finally:
        client.close()


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
        get_indexed_tickers.clear()
        st.rerun()

    page = st.radio(
        "Dashboard",
        ["Assessment", "Ask about a stock", "Monitoring"],
        key="dashboard_page",
    )


if page == "Monitoring":
    st.title("Financial Risk Monitoring")
    st.caption(
        "Assessment and human-review history. These trends are not model accuracy or "
        "drift measurements because no ground-truth outcomes are recorded."
    )
    if st.button("Refresh monitoring data", use_container_width=False):
        st.rerun()

    try:
        history = get_assessment_history()
    except Exception:
        logger.exception("Could not load dashboard monitoring history")
        history = []
        st.error("Monitoring history could not be loaded. Check local storage permissions.")

    if not history:
        st.info("No assessments have been recorded yet. Run an assessment to start the history.")
    else:
        records = pd.DataFrame(history)
        records["created_at"] = pd.to_datetime(records["created_at"], utc=True, errors="coerce")
        latest_time = records["created_at"].max()
        pending_count = int((records["review_decision"] == "PENDING").sum())
        approved_count = int((records["review_decision"] == "APPROVED").sum())
        rejected_count = int((records["review_decision"] == "REJECTED").sum())
        valid_scores = records["composite_risk_score"].dropna()

        total_col, pending_col, approved_col, rejected_col, mean_col = st.columns(5)
        total_col.metric("Assessments", len(records))
        pending_col.metric("Pending review", pending_count)
        approved_col.metric("Approved", approved_count)
        rejected_col.metric("Rejected", rejected_count)
        mean_col.metric(
            "Mean composite score",
            f"{valid_scores.mean():.3f}" if not valid_scores.empty else "Unavailable",
        )
        if pd.notna(latest_time):
            st.caption(f"Most recent assessment: {latest_time.strftime('%Y-%m-%d %H:%M UTC')}")

        scored = records.dropna(subset=["composite_risk_score"]).sort_values("created_at")
        if not scored.empty:
            st.markdown("#### Composite risk score over time")
            chart_data = scored.set_index("created_at")[["composite_risk_score"]]
            st.line_chart(chart_data, y="composite_risk_score")

        st.markdown("#### Assessment and review history")
        ticker_options = ["All tickers", *sorted(records["ticker"].dropna().unique())]
        selected_ticker = st.selectbox("Filter by ticker", ticker_options)
        visible_records = records
        if selected_ticker != "All tickers":
            visible_records = visible_records[visible_records["ticker"] == selected_ticker]
        display_columns = {
            "created_at": "Created (UTC)",
            "ticker": "Ticker",
            "risk_probability": "Model probability",
            "composite_risk_score": "Composite score",
            "risk_level": "Risk level",
            "final_status": "Workflow status",
            "review_decision": "Review decision",
            "review_reason": "Review note",
        }
        table = visible_records[list(display_columns)].rename(columns=display_columns).copy()
        table["Created (UTC)"] = table["Created (UTC)"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(table, use_container_width=True, hide_index=True)
    st.stop()

if page == "Ask about a stock":
    st.title("Ask about a stock")
    st.caption(
        "Ask questions about indexed company filings. Answers are grounded in those "
        "filings and include citations; this chat does not provide live prices or news."
    )

    active_snapshot = st.session_state.get("active_snapshot") or {}
    try:
        indexed_tickers = get_indexed_tickers()
    except Exception:
        logger.exception("Could not read indexed filing tickers")
        st.error("Could not read the filing index. Check Qdrant service status.")
        st.stop()

    if not indexed_tickers:
        st.error("No filing chunks are indexed yet. Add filings and run the indexer first.")
        st.code("docker compose run --rm index-filings", language="powershell")
        st.stop()

    active_ticker = str(active_snapshot.get("ticker", "")).strip().upper()
    if active_ticker and active_ticker not in indexed_tickers:
        st.info(
            f"The current assessment is for {active_ticker}, but the filing index has "
            f"no documents for it. Indexed tickers: {', '.join(indexed_tickers)}."
        )

    ticker_col, refresh_col, clear_col = st.columns([3, 1, 1])
    with ticker_col:
        default_index = indexed_tickers.index(active_ticker) if active_ticker in indexed_tickers else 0
        qa_ticker_input = st.selectbox(
            "Ticker with indexed filings",
            indexed_tickers,
            index=default_index,
        )
    with refresh_col:
        st.write("")
        refresh_index = st.button("Refresh tickers", use_container_width=True)
    with clear_col:
        st.write("")
        clear_chat = st.button("Clear chat", use_container_width=True)

    if refresh_index:
        get_indexed_tickers.clear()
        st.rerun()

    if st.session_state.get("qa_active_ticker") != qa_ticker_input:
        st.session_state["qa_active_ticker"] = qa_ticker_input
        st.session_state["qa_messages"] = []
    if clear_chat:
        st.session_state["qa_messages"] = []

    if active_snapshot.get("ticker") == qa_ticker_input:
        assessment = active_snapshot
        model_probability = (assessment.get("quant_report") or {}).get("risk_probability")
        result_col, score_col, status_col = st.columns(3)
        result_col.metric("Current session assessment", assessment.get("risk_level") or "UNKNOWN")
        score_col.metric(
            "Model risk probability",
            f"{model_probability:.1%}"
            if _is_number(model_probability)
            else "Unavailable",
        )
        status_col.metric(
            "Workflow status",
            (assessment.get("final_verdict") or {}).get("final_status")
            or ("Pending review" if "human_governance" in st.session_state.get("active_next", ()) else "In progress"),
        )
        st.caption("Assessment metrics are shown separately and are not filing citations.")

    for message in st.session_state.get("qa_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("model"):
                    st.caption(f"Generated with {message['model']}")
                sources = message.get("sources") or []
                if sources:
                    with st.expander("Filing sources"):
                        for source in sources:
                            source_name = (
                                source.get("source_path")
                                or source.get("doc_id")
                                or "Indexed filing"
                            )
                            location = source.get("page_number") or source.get("section")
                            st.write(
                                f"**{source.get('citation', '')} {source_name}**"
                                + (f" · {location}" if location else "")
                            )
                            if _is_number(source.get("score")):
                                st.caption(
                                    f"Retrieval similarity: {source['score']:.3f} "
                                    "(relevance signal, not confidence)."
                                )

    question = st.chat_input("Ask a question about this company’s filings")
    if question:
        question = question.strip()
        if not question:
            st.warning("Enter a question first.")
        elif len(question) > 2000:
            st.warning("Keep questions under 2,000 characters.")
        else:
            conversation_history = list(st.session_state["qa_messages"])
            st.session_state["qa_messages"].append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)
            try:
                with st.spinner("Searching filings and drafting a cited answer…"):
                    answer_result = get_rag_engine().generate_answer(
                        query=question,
                        ticker=qa_ticker_input,
                        top_k=4,
                        conversation_history=conversation_history,
                    )
                answer_message = {
                    "role": "assistant",
                    "content": answer_result["answer"],
                    "sources": answer_result.get("sources", []),
                    "model": answer_result.get("model"),
                }
            except Exception:
                logger.exception("Stock Q&A failed for %s", qa_ticker_input)
                answer_message = {
                    "role": "assistant",
                    "content": (
                        "I couldn’t answer that right now. Check that the filing index "
                        "is available and GEMINI_API_KEY is configured."
                    ),
                    "sources": [],
                }
            st.session_state["qa_messages"].append(answer_message)
            with st.chat_message("assistant"):
                st.markdown(answer_message["content"])
                if answer_message.get("model"):
                    st.caption(f"Generated with {answer_message['model']}")
                if answer_message["sources"]:
                    with st.expander("Filing sources"):
                        for source in answer_message["sources"]:
                            source_name = (
                                source.get("source_path")
                                or source.get("doc_id")
                                or "Indexed filing"
                            )
                            location = source.get("page_number") or source.get("section")
                            st.write(
                                f"**{source.get('citation', '')} {source_name}**"
                                + (f" · {location}" if location else "")
                            )
                            if _is_number(source.get("score")):
                                st.caption(
                                    f"Retrieval similarity: {source['score']:.3f} "
                                    "(relevance signal, not confidence)."
                                )
    st.stop()

st.title("Financial Risk Assessment")
st.caption(
    "Quantitative model output with filing and live-news evidence, followed by human review when required. "
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
    st.session_state.pop("active_thread_id", None)
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
            try:
                save_assessment(thread_id, snapshot.values, snapshot.next)
            except Exception:
                logger.exception("Could not save monitoring record for %s", thread_id)
                st.warning("Assessment completed, but its monitoring record was not saved.")
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
            f"{risk_level} risk requires human review. The workflow is paused; "
            "no execution has occurred."
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

    if "human_governance" in next_nodes:
        st.divider()
        st.subheader("Human review")
        st.write(
            f"Review the evidence above for **{state.get('ticker', 'this ticker')}**. "
            f"The current assessment is **{risk_level}** with score "
            f"**{state.get('composite_risk_score'):.4f}**."
            if _is_number(state.get("composite_risk_score"))
            else f"Review the evidence above for **{state.get('ticker', 'this ticker')}**."
        )
        with st.form("human_review_form", clear_on_submit=False):
            review_reason = st.text_area(
                "Review note (required when rejecting)",
                placeholder="Summarize why this assessment is approved or rejected.",
            ).strip()
            approve_col, reject_col = st.columns(2)
            with approve_col:
                approve_clicked = st.form_submit_button(
                    "Approve for execution",
                    type="primary",
                    use_container_width=True,
                )
            with reject_col:
                reject_clicked = st.form_submit_button(
                    "Reject assessment",
                    use_container_width=True,
                )

        if approve_clicked or reject_clicked:
            approved = approve_clicked
            if reject_clicked and not review_reason:
                st.error("Add a brief reason before rejecting this assessment.")
            else:
                thread_id = st.session_state.get("active_thread_id")
                if not thread_id:
                    st.error("This review session has expired. Run the assessment again.")
                else:
                    config = {"configurable": {"thread_id": thread_id}}
                    decision = {
                        "approved": approved,
                        "reason": review_reason or None,
                    }
                    try:
                        with st.spinner("Recording review decision…"):
                            graph = get_compiled_graph()
                            graph.invoke(Command(resume=decision), config=config)
                            snapshot = graph.get_state(config)
                        st.session_state["active_snapshot"] = snapshot.values
                        st.session_state["active_next"] = snapshot.next
                        try:
                            save_assessment(thread_id, snapshot.values, snapshot.next)
                        except Exception:
                            logger.exception("Could not update review history for %s", thread_id)
                            st.warning("The decision was recorded, but monitoring history was not updated.")
                        st.success(
                            "Assessment approved for execution. No trade was placed."
                            if approved
                            else "Assessment rejected. No trade was placed."
                        )
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Failed to resume human review for %s", thread_id)
                        st.error(f"Could not record the review decision: {exc}")

    if verdict.get("final_status"):
        st.caption(f"Workflow status: `{verdict['final_status']}`")
