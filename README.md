# arxiv-ai-research-tracker

A research assistant that fetches the latest AI papers from arXiv, indexes them into a searchable vector store, and lets you explore them through a conversational multi-agent interface — from the **terminal** or over **HTTP**.

**Stack & keywords:** *arXiv ingestion · Semantic Scholar batch enrichment · LanceDB + hybrid retrieval (dense embeddings + BM25) · cross-encoder reranking · interest-aware rerank from saved tags · LangGraph **Self-RAG** (tool retrieval → **grade** irrelevant chunks → generate → **hallucination check** → capped **query rewrite**) · **LangChain** tools · **LangGraph** supervisor (NL routing + command chains) · **FastAPI** REST + **SSE** streaming · **Pydantic** request models · **DeepEval** faithfulness / relevancy (OpenAI / Claude / Prometheus judges) · **LangSmith** tracing (optional) · prompt **guardrails** (regex + normalization) · SQLite LLM cache, embedding disk cache, conversation checkpoints · **Docker Compose** (optional Ollama profile) · **pytest** + **Ruff** in CI*

---

## Features

### Two surfaces, one brain

| Surface | How to run | What you get |
| --- | --- | --- |
| **CLI** | `uv run python main.py` | Interactive supervisor: type plain English (or chained commands), see routing logs, streaming tokens where supported. |
| **HTTP API** | `uv run uvicorn api:app --host 0.0.0.0 --port 8000` | Same supervisor behind **FastAPI** (`api.py`): JSON chat, SSE streaming, `/health`, OpenAPI docs at `/docs`. |

Both paths share tool calls, **thread-scoped memory** (`thread_id` in the API; session in the CLI), hybrid search, and guardrails.

### Multi-agent supervisor (orchestration)

The supervisor **classifies each user message** into an intent and may run a **chain** of steps (e.g. fetch then search). You do not pick node names manually — describe what you want in natural language.

| Task | Example phrasing | Execution |
| --- | --- | --- |
| **Ingest** | *"Fetch recent NLP papers"* | arXiv fetch → embed → LanceDB; incremental per-topic sync via `last_run.txt`. |
| **Library** | *"List saved papers"* | Reads saved store; shows arXiv IDs, citations, TLDRs where enriched. |
| **Search & Q&A** | *"Find papers on diffusion models"* | Hybrid search + rerank → **Self-RAG** answer grounded on abstracts (see below). |
| **Synthesis** | *"Summarize recent robotics work"*, *"Summarize #2504.08123v2"* | Retrieves candidates or one paper, batch / single summarization. |
| **Compare** | *"Compare #2301.12345 and #2504.08123"* | Structured comparison (motivation, method, limits, verdict); can fall back to search if IDs omitted. |
| **Collection themes** | *"Tag papers"* | Theme clusters using Semantic Scholar fields when coverage is high; else LLM clustering on titles. |
| **Reading aids** | *"Diagram #…"*, *"Get figures from #…"* | Mermaid from abstract **or** ar5iv HTML figures → PDF image fallback to `databases/paper_figures/`. |
| **Digest** | *"Daily digest"*, *"Digest last 14 days"* | Category-grouped newsletter; optional SMTP email from `.env`. |
| **No-LLM helpers** | *"Export saved --bibtex"*, *"… then explain"*, *"Trends last 14 days"* | Deterministic BibTeX/CSV, “sources used” for last RAG turn, category deltas over time windows. |
| **Personalization** | *"Save tag #… diffusion"*, *"Show tags #…"* | SQLite-backed tags/notes; boosts retrieval when query words overlap saved tags. |
| **Chaining** | *"Fetch new ML papers then find the best ones on LLMs"* | Same turn runs multiple intents in order. |

### Self-RAG (retrieval-augmented Q&A)

Used for open-ended **research questions** over your local corpus (not just keyword lookup). Implemented as a **LangGraph** graph in `agents/runner.py` (see module docstring for the full diagram): the agent calls **search tools**, **grades** retrieved chunks and drops weak ones before leaning on them, then drafts an answer, runs a **hallucination check** against context, and can **rewrite the query** (bounded retries) if grounding fails. Replies include a **Sources** block (arXiv IDs + titles) for verification.

### Hybrid search and reranking

- **Dense** vectors (OpenAI embeddings) plus **BM25** full-text over LanceDB; results **deduplicated per paper**.
- **Cross-encoder** (`ms-marco-MiniLM-L-6-v2`) reranks the shortlist for sharper precision.
- Optional **category filter** (arXiv category) on search; **interest-aware** nudge when saved tags align with query terms.

### Data pipeline: fetch → enrich → index

- **Incremental ingestion** — per-topic timestamps in `databases/last_run.txt` avoid re-fetching the whole catalog.
- **Semantic Scholar** — one batch request at fetch time adds TLDR, citation count, and fields-of-study (best-effort, non-blocking).
- **Canonical IDs** — papers keep stable arXiv ids (e.g. `2504.08123v2`) usable across CLI and API.

### Caching, models, and streaming

- **LLM response cache** (SQLite) and **embedding disk cache** (30-day TTL) reduce repeat cost.
- **Summarizer** — prefers **Ollama** (`llama3.2` by default) when reachable; otherwise OpenAI **gpt-4o-mini**.
- **Agent / RAG answers** — `AGENT_LLM=openai` (default) uses `OPENAI_MODEL` from `.env` (e.g. **gpt-5.4**); `AGENT_LLM=claude` uses Anthropic with prompt-caching headers.
- **Streaming** — `summarize`, `clarify`, `compare`, `tag`, `digest`, `diagram`, and `figures` stream tokens in the CLI; the API exposes the same supervisor via **SSE** on `/chat/stream`.

### Conversation memory

LangGraph **checkpoints** to SQLite when `langgraph-checkpoint-sqlite` is available so `thread_id` / sessions survive restarts. Use `new session` or `reset` in the CLI for a fresh thread.

### Guardrails

User and retrieved text pass through `guardrails/sanitizer.py`: prompt-injection and jailbreak-style patterns, role overrides, and exfiltration-like content (**20+** rules, Unicode normalization). The API returns **400** when input is rejected.

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py                    # Entry point — calls launch_supervisor()
├── api.py                     # FastAPI app: /health, /chat, /chat/stream (SSE)
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
| `/health` | GET | Liveness — returns `{"status":"ok"}`. |
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

Convenience commands (optional):

```bash
make lint
make test
make api
make ci-test
```

```
[Supervisor Ready] Fetch, search, summarize, compare, tag, digest, or diagram papers.
Examples:
  - 'fetch recent robotics papers'
  - 'list saved papers'
  - 'find papers on diffusion models'
  - 'summarize #2504.08123v2'
  - 'compare #2301.12345 and #2504.08123'
  - 'tag papers'
  - 'daily digest'
  - 'diagram #2504.08123'
  - 'get figures from #2504.08123'
  - 'export saved --bibtex'
  - 'save tag #2504.08123 diffusion'
  - 'trends last 14 days'
  - 'find papers on LLMs then explain'
  - 'fetch NLP papers then find the best on transformers'

You:
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
- Pinned means match [`evaluation/eval_metrics_snapshot.json`](evaluation/eval_metrics_snapshot.json). Captured with **`gpt-4o`** judging answers from **`gpt-5.4`** (`OPENAI_MODEL`). If `.env` sets `EVAL_JUDGE_MODEL` equal to `OPENAI_MODEL`, scores skew high — override on the CLI (see below) or use `EVAL_JUDGE=prometheus|claude`. Re-run with `--write-metrics` after changing judges or corpus.
- Hallucination on the summarizer suite was not measured in this snapshot because
  the LanceDB table had no rows for the random summarizer sample path at run time
  (run `main.py` / fetch first to populate).
- Adversarial relevancy is often **expected to be low** when retrieval is intentionally
  off-topic; faithfulness should stay high if the model refuses to invent facts.

```bash
uv run python -m evaluation.run_eval                        # all suites
uv run python -m evaluation.run_eval --suite rag            # RAG only
uv run python -m evaluation.run_eval --suite adversarial    # adversarial only
# After a successful run, capture aggregates for README / CI artifacts (pin gpt-4o judge if .env uses gpt-5.4 for both):
EVAL_JUDGE=openai EVAL_JUDGE_MODEL=gpt-4o uv run python -m evaluation.run_eval --suite all --write-metrics evaluation/eval_metrics_snapshot.json
uv run pytest evaluation/                                   # pytest suite
```

At the end of a successful `run_eval` run, the runner prints an `EVAL_SUMMARY`
block you can paste into release notes or refresh the table above.

`evaluation/test_api.py` is a fast API-layer unit suite that stubs heavy supervisor/runtime dependencies; it validates endpoint behavior and error handling without requiring live model/provider connectivity.

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

User input and retrieved content passes through `guardrails/sanitizer.py`. Queries over 500 characters are rejected. API keys are never hardcoded — loaded from `.env` only.

---

## Get API Keys

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
