**Autonomous Financial Risk & Portfolio Management Engine**

## Gemini RAG answers

The LLM pipeline reads Gemini settings from `financial-engine/.env`. Add your
Google AI Studio key as `GEMINI_API_KEY`; `financial-engine/.env.example`
contains the available settings. The real `.env` file is ignored by Git.

Install the project dependencies from `financial-engine`, then index documents
and run the retrieval engine:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m src.llm_pipeline.vector_store .\data\filings
.\venv\Scripts\python.exe -m src.llm_pipeline.rag_engine
```

`FinancialRAGEngine.generate_answer()` returns a grounded answer and a list of
source citations. Set `GEMINI_MODEL`, `GEMINI_TEMPERATURE`, or
`GEMINI_MAX_OUTPUT_TOKENS` in `.env` to change generation settings.
The default model fallback order is Gemini 3.8 Flash, 3.6 Flash, then 2.5 Flash;
set `GEMINI_FALLBACK_MODELS` to customize the fallback sequence.
