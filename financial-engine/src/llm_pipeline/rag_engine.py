from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from src.llm_pipeline.vector_store import COLLECTION_NAME

class FinancialRAGEngine:
    def __init__(self, storage_path: str = "data/qdrant_db"):
        self.client = QdrantClient(path=storage_path)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve_context(
        self, 
        query: str, 
        ticker: Optional[str] = None, 
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """Retrieves contextually relevant text chunks matching query and metadata filters."""
        query_vector = self.encoder.encode(query).tolist()
        
        query_filter = None
        if ticker:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="ticker",
                        match=MatchValue(value=ticker.upper())
                    )
                ]
            )

        search_results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        retrieved_docs = []
        for res in search_results:
            retrieved_docs.append({
                "score": round(res.score, 4),
                "ticker": res.payload.get("ticker"),
                "doc_id": res.payload.get("doc_id"),
                "section": res.payload.get("section"),
                "text": res.payload.get("text")
            })

        return retrieved_docs

if __name__ == "__main__":
    rag = FinancialRAGEngine()
    results = rag.retrieve_context(
        query="What supply chain or component risks exist?",
        ticker="AAPL",
        top_k=2
    )
    print("RAG Retrieval Output:")
    for r in results:
        print(f"\n[Score: {r['score']}] {r['ticker']} ({r['doc_id']}):")
        print(f"Text: {r['text']}")