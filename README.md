# ArXiv AI Research Tracker

A research assistant that fetches the latest AI papers from arXiv, indexes them into a searchable vector store, and lets you explore them through a conversational multi-agent interface — from the **terminal** or over **HTTP**.

| Layer | Technology |
|---|---|
| Vector store & retrieval | LanceDB · OpenAI embeddings · dense + BM25 hybrid retrieval (RRF merge) · cross-encoder reranking |
| Agent framework | LangGraph (supervisor + Self-RAG) · LangChain tools |
| API | FastAPI · Pydantic · SSE streaming |
| LLM support | OpenAI · Anthropic Claude · Ollama (local) · vLLM (GPU serving) |
| Observability | Prometheus `/metrics` · Grafana dashboard · structured JSON logs |
| Evaluation | DeepEval · LangSmith tracing |
| Storage | SQLite (cache, memory, metadata) · NDJSON · diskcache |

---

## Features

### Multi-agent supervisor

| Capability | Example | What happens |
| --- | --- | --- |
| **Ingest** | *"Fetch recent NLP papers"* | Pulls new papers from arXiv, embeds, and indexes them. Incremental — only fetches what's new. |
| **Library** | *"List saved papers"* | Shows your saved collection with arXiv IDs, citation counts, and one-sentence TLDRs. |
| **Search & Q&A** | *"Find papers on diffusion models"* | Local vector search first (oversampled, reranked); falls back to live arXiv if nothing is indexed — results are saved and indexed automatically. |
| **Summarize** | *"Summarize recent robotics work"*, *"Summarize #2504.08123v2"* | Batch or single-paper summarization. |
| **Compare** | *"Compare #2301.12345 and #2504.08123"* | Side-by-side comparison: motivation, approach, limitations, and a verdict. |
| **Themes** | *"Tag papers"* | Groups your collection into named research themes. |
| **Reading aids** | *"Diagram #…"*, *"Get figures from #…"* | Mermaid methodology diagram from the abstract, or real figures extracted from the paper. |
| **Digest** | *"Daily digest"*, *"Digest last 14 days"* | Newsletter-style summary grouped by research area. |
| **Export** | *"Export saved --bibtex"*, *"Trends last 14 days"* | Deterministic BibTeX/CSV export or category trend analysis. |
| **Tags & notes** | *"Save tag #… diffusion"* | Attach personal tags and notes to papers — influences future retrieval ranking. |
| **Chaining** | *"Fetch new ML papers then find the best ones on LLMs"* | Multiple steps run in sequence within a single turn. |

### Self-RAG

Multi-step verification loop: retrieves chunks via `hybrid_search()` → grades relevance → drafts a grounded answer → checks for hallucinations (rewrites and retries if grounding fails) → returns a **Sources** block (arXiv IDs + titles).

### Retrieval and reranking

Dense vector search (OpenAI embeddings) and BM25 run in parallel, merged with **Reciprocal Rank Fusion** (RRF), then reranked with a **cross-encoder** for a final precision pass. LangChain's built-in hybrid path is bypassed — it's incompatible with LanceDB ≥0.30. Saved tags influence ranking when they overlap the query.

### Model serving and caching

Set `AGENT_LLM` in `.env` to switch between `openai`, `claude`, `ollama`, or `vllm`. Summarizer prefers local Ollama; falls back to `gpt-4o-mini`. Three-layer caching: LLM responses (SQLite), embeddings (disk, 30-day TTL), citation edges (SQLite). Streaming supported across all major tools via SSE.

### Observability

Prometheus metrics at `/metrics` · Grafana dashboard (auto-provisioned) · structured JSON logs · rate limiting (20 req/min per IP). Conversation state checkpointed to SQLite — sessions survive restarts.

### Security

All user input passes through `guardrails/sanitizer.py`: prompt-injection patterns, jailbreak triggers, role overrides, and exfiltration-style content (20+ rules, Unicode normalization). Queries over 500 characters are rejected; the API returns HTTP 400 for invalid input.

---

## Evaluation

Scored with [DeepEval](https://github.com/confident-ai/deepeval). Answer model: `gpt-5.4`. Judge: `claude-haiku-4-5` (cross-provider — avoids same-model inflation). Raw scores: [`evaluation/eval_metrics_snapshot.json`](evaluation/eval_metrics_snapshot.json).

| Suite | Faithfulness | Answer relevancy | n |
| --- | ---: | ---: | ---: |
| RAG | **0.992** ± 0.024 | **0.870** ± 0.241 | 10 |
| Adversarial RAG | **0.952** ± 0.082 | **0.292** ± 0.505 | 3 |

Adversarial relevancy is intentionally low — the model stays grounded and refuses off-topic queries rather than fabricate.

```bash
uv run python -m evaluation.run_eval --suite rag
uv run python -m evaluation.run_eval --suite adversarial --write-metrics evaluation/eval_metrics_snapshot.json
uv run pytest evaluation/   # deterministic tests, no API key needed
```

Configure judge in `.env`: `EVAL_JUDGE=openai|claude|prometheus`, `EVAL_JUDGE_MODEL=<model-name>`.

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py          # CLI entry point
├── api.py           # FastAPI app — /health, /chat, /chat/stream, /metrics
├── agents/          # LangGraph supervisor + Self-RAG runner + tools
├── ingestion/       # arXiv fetching, incremental sync, Semantic Scholar enrichment
├── databases/       # LanceDB stores, hybrid search, caching, export utils
├── guardrails/      # Prompt injection sanitizer
├── evaluation/      # DeepEval metrics, test datasets, pytest suites, snapshot
├── grafana/         # Pre-built dashboard + Prometheus scrape config
└── pyproject.toml
```

---

## Setup

```bash
git clone https://github.com/ravichrn/arxiv-ai-research-tracker.git
cd arxiv-ai-research-tracker
uv sync
cp .env.example .env
```

Requires Python 3.13. See `.env.example` for all configuration options.

---

## Usage

### CLI

```bash
uv run python main.py
```

### HTTP API

Same supervisor behind FastAPI — Pydantic-validated JSON, OpenAPI docs at `/docs`, optional LangSmith tracing.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Liveness check |
| `/models` | GET | Active LLM backend and model |
| `/metrics` | GET | Prometheus metrics |
| `/chat` | POST | One supervisor turn — `{"query":"...","thread_id":"..."}` |
| `/chat/stream` | POST | SSE streaming version of `/chat` |

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
docker compose up --build                           # app only
docker compose --profile local-llm up --build      # with local Ollama
docker compose --profile vllm up --build           # with vLLM GPU serving
docker compose --profile monitoring up             # with Prometheus + Grafana
```

Make targets: `make lint` · `make test` · `make ci-test` · `make api`.

---

## Sample terminal session

An illustrative transcript is in [`docs/sample_terminal_session.txt`](docs/sample_terminal_session.txt). The supervisor prints routing logs (`[Supervisor] intent=...`), then streams model tokens or prints a final block when the turn completes.
