# Retrieval Evaluation Metrics

When evaluating a retrieval system (such as in RAG pipelines), the following core metrics are used to measure performance, relevance, and speed:

| Metric | Core Question | Focus |
| :--- | :--- | :--- |
| **Recall@k** | Did we retrieve the passages needed to answer? | Measures the proportion of relevant passages successfully captured within the top $k$ results. |
| **MRR or nDCG@k** | How high did the relevant passages rank? | Evaluates ranking quality, giving higher weight and position penalties to relevant items appearing further down the list. |
| **Precision@k / Context Precision** | How much of what we retrieved was useful? | Measures the proportion of retrieved passages that are genuinely relevant and actionable versus noise. |
| **Latency** | How long retrieval takes. | Measures system performance, tracking the time elapsed from query submission to result delivery. |

---

## Detailed Breakdown

### 1. Recall@k
* **Question:** Did we retrieve the passages needed to answer?
* **Description:** Ensures the system isn't missing critical context required to answer the user's prompt.

### 2. MRR or nDCG@k
* **Question:** How high did the relevant passages rank?
* **Description:** Evaluates positional relevance. *MRR* (Mean Reciprocal Rank) looks at the rank of the first relevant item, while *nDCG* (Normalized Discounted Cumulative Gain) accounts for graded relevance across multiple positions.

### 3. Precision@k / Context Precision
* **Question:** How much of what we retrieved was useful?
* **Description:** Keeps token usage efficient and prevents LLM distraction by ensuring retrieved chunks contain minimal noise or irrelevant text.

### 4. Latency
* **Question:** How long retrieval takes.
* **Description:** Critical for real-time user experiences to ensure the pipeline responds within acceptable time thresholds.