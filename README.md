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
The Primary model is Gemini 4 Hydrogen. The default model fallback order is Gemini 3.8 Flash, 3.6 Flash, then 2.5 Flash;
set `GEMINI_FALLBACK_MODELS` to customize the fallback sequence.

## Risk assessment dashboard

From `financial-engine`, start the local dashboard with:

```powershell
.\venv\Scripts\python.exe -m streamlit run src/ui/dashboard.py
```

The dashboard displays the quantitative model output, retrieved filing chunks,
and recent news results. It reports whether the prediction API, Qdrant index,
and MLflow UI are reachable. News sentiment is explicitly unscored. The local
Streamlit file watcher is disabled to avoid noisy optional `torchvision`
imports from Transformers; restart the dashboard manually after editing code.
When the risk level requires review, the dashboard pauses and offers explicit
approve/reject controls. Rejecting requires a note. Approval records clearance
for execution only; the workflow does not place trades.

The **Ask about a stock** view lets users ask ticker-filtered questions about
indexed filings before or after an assessment. Answers include filing citations
and can use recent chat turns to resolve follow-up questions. This view does not
provide live stock prices or current news, and assessment metrics are displayed
separately from filing evidence.

For a ticker that is not indexed, the dashboard can fetch its latest annual and
interim SEC filings, cache them in `financial-engine/data/sec_filings`, and add
their chunks to Qdrant automatically. Set `SEC_USER_AGENT` in `.env` once to an
application name and a real contact email, as requested by the SEC for automated
access. SEC filings are public and this lookup does not require an API key.

Use the dashboard's **Monitoring** view to review recent assessments, risk-score
trends, pending reviews, and reviewer decisions. Local history is stored in
`financial-engine/data/dashboard_monitoring.sqlite3` and survives dashboard
restarts. It contains assessment summaries and review notes; an in-progress
human-review checkpoint itself is still held in memory and cannot be resumed
after a dashboard process restart. The monitoring view does not measure model
accuracy or drift because verified outcome labels are not yet collected.

## Run the full stack with Docker

From `financial-engine`, start the dashboard, prediction API, MLflow, and Qdrant
with one command:

```powershell
docker compose up --build
```

Container builds use `financial-engine/requirements-container.txt` instead of
installing the full local development set. Spark/Delta extras are omitted from
the service image, and embeddings use the CPU-only PyTorch wheel; Docker caches
pip downloads for later rebuilds.

Open the dashboard at <http://localhost:8501>. MLflow is at
<http://localhost:5000>, the prediction API docs are at
<http://localhost:8000/docs>, and Qdrant is at <http://localhost:6333/dashboard>.
The startup process indexes the PDFs in `financial-engine/data/filings` into
Qdrant. Stop the stack with Ctrl+C; `docker compose down` also stops it while
preserving Qdrant and downloaded embedding-model data.

Docker Desktop must be running. The compose setup reuses the local
`mlflow.db`, `mlartifacts`, Feast online-store files, and feature parquet data.
If the champion model artifacts are missing, train and promote a model once
from `financial-engine` with:

```powershell
docker compose --profile training run --build --rm train-model
```

Then start the regular stack with `docker compose up --build`.

Project still in progress
