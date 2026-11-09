import logging
import os
import requests
from typing import Dict, Any, List
from langchain_core.tools import tool
from src.llm_pipeline.rag_engine import FinancialRAGEngine

logger = logging.getLogger(__name__)
MLOPS_SERVICE_URL = "http://127.0.0.1:8000/predict_risk"
WEB_SEARCH_MAX_RESULTS = max(1, min(10, int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))))
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "us-en")
WEB_SEARCH_TIMELIMIT = os.getenv("WEB_SEARCH_TIMELIMIT", "w")


# Tool 1: Quantitative MLOps Microservice Tool
@tool
def get_quantitative_risk_tool(ticker: str) -> Dict[str, Any]:
    """Queries the Phase 2 MLOps FastAPI microservice to retrieve real-time quantitative XGBoost risk scores and Feast features.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'NVDA', 'NEXA').
    """
    try:
        response = requests.post(
            MLOPS_SERVICE_URL,
            json={"ticker": ticker.upper()},
            timeout=5.0
        )
        if response.status_code == 200:
            return response.json()
        logger.error(f"MLOps service status {response.status_code}: {response.text}")
        return {"error": f"HTTP {response.status_code}", "risk_category": "UNKNOWN", "risk_probability": 0.5}
    except Exception as e:
        logger.exception("Failed to connect to MLOps endpoint.")
        return {"error": str(e), "risk_category": "UNKNOWN", "risk_probability": 0.5}


# Tool 2: Qualitative SEC Audit RAG Tool
@tool
def query_sec_rag_tool(ticker: str, query: str = "supply chain regulatory litigation risk") -> List[Dict[str, Any]]:
    """Queries the Phase 3 Qdrant vector index to extract qualitative risk disclosures and regulatory warnings from SEC filings.
    
    Args:
        ticker: Stock ticker symbol.
        query: Specific semantic risk search terms.
    """
    try:
        with FinancialRAGEngine() as rag:
            return rag.retrieve_context(query=query, ticker=ticker, top_k=2)
    except Exception as e:
        logger.exception("Failed to query Qdrant RAG index.")
        return []


# Tool 3: Market Sentiment & Web Events Tool
@tool
def search_market_sentiment_tool(ticker: str) -> Dict[str, Any]:
    """Searches live web news for recent company-specific events.
    
    Args:
        ticker: Stock ticker symbol.
    """
    ticker_upper = ticker.strip().upper()
    if not ticker_upper:
        return {"error": "ticker must not be empty", "news": []}

    try:
        from ddgs import DDGS

        results = DDGS(timeout=10).news(
            f'"{ticker_upper}" stock company',
            region=WEB_SEARCH_REGION,
            safesearch="moderate",
            timelimit=WEB_SEARCH_TIMELIMIT,
            max_results=WEB_SEARCH_MAX_RESULTS,
        )
        news = [
            {
                "headline": item.get("title"),
                "snippet": item.get("body"),
                "url": item.get("url"),
                "source": item.get("source"),
                "published_at": item.get("date"),
            }
            for item in results
        ]
        return {
            "ticker": ticker_upper,
            "news": news,
            "result_count": len(news),
            "sentiment_label": "UNASSESSED",
            "sentiment_score": None,
        }
    except Exception as exc:
        logger.exception("Live news search failed for %s", ticker_upper)
        return {
            "ticker": ticker_upper,
            "news": [],
            "result_count": 0,
            "sentiment_label": "UNAVAILABLE",
            "sentiment_score": None,
            "error": str(exc),
        }
