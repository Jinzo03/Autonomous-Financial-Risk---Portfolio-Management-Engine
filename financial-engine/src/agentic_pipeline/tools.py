import logging
import requests
from typing import Dict, Any, List
from langchain_core.tools import tool
from src.llm_pipeline.rag_engine import FinancialRAGEngine

logger = logging.getLogger(__name__)
MLOPS_SERVICE_URL = "http://127.0.0.1:8000/predict_risk"


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
    """Runs live market news and sentiment checks for recent macroeconomic or company-specific news events.
    
    Args:
        ticker: Stock ticker symbol.
    """
    # Simulates live market search API integration (e.g., Tavily, DuckDuckGo, or NewsAPI)
    mock_news_database = {
        "AAPL": {"headline": "Antitrust scrutiny intensifies over App Store revenue share.", "sentiment": "BEARISH", "sentiment_score": -0.45},
        "NVDA": {"headline": "Next-gen AI chip demand surges despite international export limits.", "sentiment": "BULLISH", "sentiment_score": 0.65},
        "NEXA": {"headline": "Patent infringement suit filed in Delaware District Court.", "sentiment": "VERY_BEARISH", "sentiment_score": -0.80}
    }
    
    ticker_upper = ticker.upper()
    news = mock_news_database.get(
        ticker_upper, 
        {"headline": "No major market events reported in the last 24 hours.", "sentiment": "NEUTRAL", "sentiment_score": 0.0}
    )
    
    return {
        "ticker": ticker_upper,
        "recent_headline": news["headline"],
        "sentiment_label": news["sentiment"],
        "sentiment_score": news["sentiment_score"]
    }