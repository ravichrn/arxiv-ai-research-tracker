# ArXiv AI Research Tracker

A research assistant that fetches the latest AI papers from arXiv, indexes them into a searchable vector store, and lets you explore them through a conversational multi-agent interface — from the **terminal** or over **HTTP**.

| Layer | Technology |
|---|---|
| Vector store & retrieval | LanceDB · OpenAI embeddings · dense retrieval (k*3 oversample) · cross-encoder reranking · FTS index built for future BM25 |
| Agent framework | LangGraph (supervisor + Self-RAG) · LangChain tools |
| API | FastAPI · Pydantic · SSE streaming |
| Data sources | arXiv API · Semantic Scholar batch API |
| LLM support | OpenAI · Anthropic Claude · Ollama (local) · vLLM (GPU serving) |
| Model serving | vLLM OpenAI-compatible API · pluggable via `AGENT_LLM` env var |
| Observability | Prometheus `/metrics` · Grafana dashboard · structured JSON logs |
| Evaluation | DeepEval · LangSmith tracing |
| Storage | SQLite (cache, memory, metadata) · NDJSON · diskcache |
| Tooling | uv · Ruff · pytest · Docker Compose |

---

## Features

### Interfaces

| Surface | What you get |
| --- | --- |
| **CLI** | Interactive supervisor — type plain English, see routing logs, and get streaming output where supported. |
| **HTTP API** | The same supervisor behind FastAPI: JSON chat, SSE streaming, `/health`, and OpenAPI docs at `/docs`. |

Both surfaces share the same tool calls, conversation memory, the same `hybrid_search()` retrieval helper (dense + rerank today), and guardrails.

### Multi-agent supervisor

Describe what you want in natural language. The supervisor routes your message to the right agent and can chain multiple steps in one turn.

| Capability | Example | What happens |
| --- | --- | --- |
| **Ingest** | *”Fetch recent NLP papers”* | Pulls new papers from arXiv, embeds, and indexes them. Incremental — only fetches what's new. |
| **Library** | *”List saved papers”* | Shows your saved collection with arXiv IDs, citation counts, and one-sentence TLDRs. |
| **Search & Q&A** | *”Find papers on diffusion models”*, *”Find JEPA paper by Yann LeCun”* | Local vector search first (oversampled, reranked); falls back to live arXiv search if nothing is indexed — results are automatically saved and indexed for future queries. Each result shows its `#arxiv_id` for immediate follow-up commands. |
| **Summarize** | *”Summarize recent robotics work”*, *”Summarize #2504.08123v2”* | Batch or single-paper summarization. |
| **Compare** | *”Compare #2301.12345 and #2504.08123”* | Side-by-side comparison: motivation, approach, limitations, and a verdict. |
| **Themes** | *”Tag papers”* | Groups your collection into named research themes. |
| **Reading aids** | *”Diagram #…”*, *”Get figures from #…”* | Mermaid methodology diagram from the abstract, or real figures extracted from the paper. |
| **Digest** | *”Daily digest”*, *”Digest last 14 days”* | Newsletter-style summary grouped by research area, with optional email delivery. |
| **Export** | *”Export saved --bibtex”*, *”Trends last 14 days”* | Deterministic BibTeX/CSV export, source attribution for the last answer, or category trend analysis. |
| **Tags & notes** | *”Save tag #… diffusion”*, *”Show tags #…”* | Attach personal tags and notes to papers; influences future retrieval ranking. |
| **Chaining** | *”Fetch new ML papers then find the best ones on LLMs”* | Multiple steps run in sequence within a single turn. |

### Self-RAG (retrieval-augmented Q&A)

Answers open-ended research questions over your local corpus using a multi-step verification loop:

1. Retrieves candidate chunks via `hybrid_search()` (dense vectors + cross-encoder rerank).
2. Grades each chunk for relevance — weak results are dropped before generation.
3. Drafts an answer grounded only on the kept context.
4. Checks the answer for hallucinations; rewrites the query and retries if grounding fails.
5. Returns a **Sources** block (arXiv IDs + titles) for verification.

### Retrieval and reranking

- **Dense vector search** (OpenAI embeddings) oversampled (roughly **3× k** chunks before dedupe), then **deduplication per paper** (by URL).
- **Cross-encoder** reranking on the shortlist for higher precision.
- An **FTS index** is maintained on the text column for future BM25/hybrid wiring; LangChain’s current `query_type="hybrid"` path is incompatible with recent `lancedb` releases, so BM25 is not mixed into the score today (see `hybrid_search` docstring in `databases/stores.py`).
- Optional arXiv category filter; saved tags influence ranking when they overlap the query.

### Ingestion pipeline

- **Incremental** — per-topic timestamps ensure only newly published papers are fetched on each run; no duplicates.
- **Semantic Scholar enrichment** — a single batch request at fetch time adds a one-sentence TLDR, citation count, and fields of study.
- **Canonical IDs** — papers are assigned stable arXiv IDs (e.g. `2504.08123v2`) usable across the CLI and API.

### Model serving and caching

- **Pluggable backends** — set `AGENT_LLM` in `.env` to switch between `openai`, `claude`, `ollama`, or `vllm`. vLLM runs any HuggingFace model locally via GPU with an OpenAI-compatible API.
- **Summarizer** — prefers **Ollama** (`llama3.2` by default) when reachable; otherwise OpenAI **gpt-4o-mini**.
- **LLM response cache** (SQLite) and **embedding disk cache** (30-day TTL) reduce repeat cost.
- **Prompt caching** — Anthropic beta header added automatically when `AGENT_LLM=claude`.
- **Streaming** — `summarize`, `clarify`, `compare`, `tag`, `digest`, `diagram`, and `figures` stream tokens in the CLI; the API exposes the same supervisor via **SSE** on `/chat/stream`.

### Observability

- **Prometheus metrics** at `GET /metrics` — request count, p50/p95 latency, and error rate per endpoint.
- **Grafana dashboard** (`grafana/dashboard.json`) — pre-built panels auto-provisioned on `docker compose --profile monitoring up`. Panels target `prometheus-fastapi-instrumentator` defaults (`http_requests_total` with grouped `status` labels like `2xx` / `5xx`, and `http_request_duration_seconds`).
- **Structured JSON logs** — all log output is machine-parseable (compatible with Datadog, CloudWatch, etc.).
- **Rate limiting** — `/chat` and `/chat/stream` are capped at 20 requests/minute per IP (HTTP 429 on excess).

### Conversation memory

Conversation state is checkpointed to SQLite so sessions survive restarts. Use `new session` or `reset` in the CLI to start a fresh thread.

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py                    # Entry point — calls launch_supervisor()
├── api.py                     # FastAPI: /health, /models, /metrics, /chat, /chat/stream (SSE)
├── agents/
│   ├── supervisor.py          # Multi-agent supervisor: routes all intents, supports chaining
│   ├── runner.py              # Self-RAG LangGraph agent (grade_docs → agent → hallucination_check)
│   └── tools.py               # search_papers, search_saved_papers, add/delete saved
├── ingestion/
│   └── arxiv_fetcher.py       # arXiv fetching, incremental sync, S2 enrichment, figure extraction
├── databases/
│   ├── stores.py              # LanceDB stores, hybrid search, LLM singletons, caching
│   ├── export_utils.py        # Deterministic BibTeX/CSV export (no LLM)
│   ├── interest_rerank.py     # Interest-aware reranking via saved tags
│   ├── saved_metadata.py      # SQLite side-table for user tags/notes
│   ├── trends_utils.py        # Category trend analysis over time windows
│   ├── papers_raw.jsonl       # NDJSON cache of all fetched paper metadata
│   └── last_run.txt           # Per-topic fetch timestamps (JSON)
├── guardrails/
│   └── sanitizer.py           # Prompt injection prevention
├── evaluation/
│   ├── datasets.py            # Test cases (summarizer + RAG)
│   ├── eval_metrics_snapshot.json  # Pinned mean scores for README (regenerate locally)
│   ├── run_eval.py            # Standalone eval runner
│   ├── test_summarizer.py     # pytest — hallucination + summarization metrics
│   ├── test_rag.py            # pytest — faithfulness + relevancy metrics
│   ├── test_api.py            # pytest — FastAPI routes (stubbed supervisor)
│   ├── test_guardrails.py     # pytest — sanitizer edge cases
│   └── test_feature_helpers.py  # pytest — deterministic export/trends/sqlite helpers
├── grafana/
│   ├── dashboard.json         # Pre-built Grafana dashboard (request rate, latency, errors)
│   ├── prometheus.yml         # Prometheus scrape config targeting the app
│   └── provisioning/          # Auto-provisioned Grafana datasource + dashboard
├── docs/
│   └── sample_terminal_session.txt  # Illustrative CLI transcript
├── prompts/
│   └── summarize.txt          # Summarization prompt template
└── pyproject.toml             # uv-managed dependencies
```

---

## Setup

```bash
git clone https://github.com/ravichrn/arxiv-ai-research-tracker.git
cd arxiv-ai-research-tracker
uv sync
```

Use Python 3.13 for local development (the current dependency set is not stable on Python 3.14).

Create a `.env` file using `.env.example` as a reference.

---

## Usage

### CLI (interactive supervisor)

```bash
uv run python main.py
```

### HTTP API (FastAPI)

`api.py` defines a **[FastAPI](https://fastapi.tiangolo.com/)** application (`app`) served with **Uvicorn**. It wraps the same supervisor as the CLI: **Pydantic**-validated JSON bodies, OpenAPI schema, and optional **LangSmith** traces for requests that hit the LLM.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Liveness — returns `{"status":"ok","backend":"...","model":"..."}`. |
| `/models` | GET | Active LLM backend and model names. |
| `/metrics` | GET | Prometheus metrics (request count, latency, error rate). |
| `/chat` | POST | One full supervisor turn; JSON body `{"query":"...","thread_id":"..."}`. |
| `/chat/stream` | POST | Same as `/chat` but streams **Server-Sent Events** (`text/event-stream`); ends with `event: done`. |

After starting the server, open **`http://localhost:8000/docs`** for interactive **Swagger UI** (try requests from the browser) or **`http://localhost:8000/redoc`** for ReDoc.

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

API prerequisites:
- Set `OPENAI_API_KEY` in `.env`, or
- Run with local Ollama (`docker compose --profile local-llm up --build`) and ensure Ollama is reachable.
- Without either provider, `/chat` and `/chat/stream` will return provider connection errors.

One-line chat request:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query":"find papers on diffusion models","thread_id":"demo"}'
```

One-line streaming chat request (SSE):

```bash
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d '{"query":"summarize recent cs.AI papers","thread_id":"demo"}'
```

Run with Docker Compose:

```bash
docker compose up --build
```

Run with optional local Ollama sidecar:

```bash
docker compose --profile local-llm up --build
```

Run with vLLM GPU model serving:

```bash
AGENT_LLM=vllm docker compose --profile vllm up --build
```

Run with Prometheus + Grafana monitoring (dashboard at `http://localhost:3000`, Prometheus UI at `http://localhost:9090`):

```bash
docker compose --profile monitoring up
```

Grafana’s default login is **`admin` / `admin`** unless you set **`GF_SECURITY_ADMIN_PASSWORD`** in `.env` (recommended outside local sandboxes).

Make targets:

```bash
make lint
make test
make api
make ci-test
```

Type `exit` or `quit` to stop. Use `ollama pull llama3.2` for local summarization.

---

## Sample terminal session

The supervisor prints routing logs, then either streams model tokens (for long-form
nodes) or prints a final block when the turn completes. A longer illustrative
transcript lives in [`docs/sample_terminal_session.txt`](docs/sample_terminal_session.txt).

```text
You: fetch recent cs.CL papers
  [Supervisor] intent=fetch, chain=[], topics=['cs.CL']
Fetched and indexed 12 new paper(s) for: cs.CL. You can now search them.

You: find papers on retrieval-augmented generation
  [Supervisor] intent=rag, chain=[], topics=[]
Agent: ... grounded answer with inline citations ...

You: summarize recent cs.CL papers
Agent: - Paper A: ...   (tokens stream here in the real CLI)

You: exit
```

---

## Evaluation

### Reported benchmark scores (pinned snapshot)

These means come from a single local run of `evaluation.run_eval` (DeepEval
metrics). Raw numbers and metadata are committed in
[`evaluation/eval_metrics_snapshot.json`](evaluation/eval_metrics_snapshot.json).
Re-run and replace that file when you change models, judges, or corpus.

Faithfulness and relevancy use the **judge** LLM, not deterministic code. If the
judge is the **same** model as the one that wrote the answers (`OPENAI_MODEL`),
scores tend to look unrealistically high; this repo defaults the OpenAI judge to
**`gpt-4o`** when `EVAL_JUDGE=openai` so a typical **`gpt-5.4`** agent is graded
cross-model. Use `EVAL_JUDGE=prometheus` or `claude` for stronger separation.

| Suite | Faithfulness (mean) | Answer relevancy (mean) | Cases scored | Judge |
| --- | ---: | ---: | ---: | --- |
| RAG (`RAG_CASES`) | **0.921** | **0.712** | 10 | `openai/gpt-4o` |
| Adversarial RAG (`ADVERSARIAL_RAG_CASES`) | **1.000** | **0.362** | 3 | `openai/gpt-4o` |

Notes:
- Scores were captured with `gpt-4o` judging answers from a different agent model to avoid same-model inflation. If your `.env` uses the same model for both agent and judge, scores will skew high — use `EVAL_JUDGE=prometheus` or `claude` for separation.
- Adversarial answer relevancy is expected to be low when retrieval is intentionally off-topic; faithfulness should remain high if the model refuses to fabricate.
- The summarizer hallucination metric requires papers to be indexed first — run `main.py` and fetch at least one topic before evaluating.

```bash
uv run python -m evaluation.run_eval                   # all suites
uv run python -m evaluation.run_eval --suite rag       # RAG only
uv run python -m evaluation.run_eval --suite adversarial

# Pin a cross-model judge and write updated scores to the snapshot file:
EVAL_JUDGE=openai EVAL_JUDGE_MODEL=gpt-4o \
  uv run python -m evaluation.run_eval --suite all --write-metrics evaluation/eval_metrics_snapshot.json

uv run pytest evaluation/                              # pytest suite
```

After a successful run, the runner prints an `EVAL_SUMMARY` block you can use to refresh the table above. `evaluation/test_api.py` stubs the supervisor and validates endpoint behavior without requiring a live model or API key.

**Judge model** — set `EVAL_JUDGE` in `.env`:

| `EVAL_JUDGE` | Judge | Cost |
|---|---|---|
| `prometheus` | Prometheus 2 via Ollama | Free (local) |
| `claude` | `claude-opus-4-6` | ~$6–10/run |
| `openai` | `gpt-4o` (default judge; differs from typical agent model) | ~$1–3/run |

---

## Tracing with LangSmith

Add to `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
```
All LLM calls and agent steps appear in the LangSmith dashboard automatically.

---

## Security

All user input and retrieved content passes through `guardrails/sanitizer.py`: prompt-injection patterns, jailbreak triggers, role overrides, and exfiltration-style content (20+ rules, Unicode normalization). Queries over 500 characters are rejected; the API returns HTTP 400 for invalid input. API keys are never hardcoded — loaded from `.env` only.

---

## API Keys

| Provider | URL |
|----------|-----|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) |
| Ollama (local) | [ollama.com](https://ollama.com) — no key needed |

---

## Future Enhancements

- [ ] Graph RAG — knowledge graph over entities and relationships across papers
- [ ] Web UI (Streamlit/Gradio or SPA) on top of the existing FastAPI `/chat` and `/chat/stream` endpoints
- [ ] Full PDF analysis — fetch and analyze the complete paper, not just the abstract
- [ ] Slack digest delivery (alongside existing email delivery)
