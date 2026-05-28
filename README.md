# ArXiv AI Research Tracker

A supervisor–worker multi-agent research platform built with LangGraph and FastAPI. A routing supervisor LLM-classifies intent and dispatches to specialized nodes (ingest, summarize, compare, tag, diagram, lineage), invoking a compiled Self-RAG sub-agent — retrieve → grade docs → rewrite query → hallucination check — over a LanceDB vectorstore with hybrid dense+BM25 retrieval (RRF merge) and cross-encoder reranking. Pluggable LLM backends (OpenAI, Claude, Ollama, vLLM), three-layer caching, Prometheus observability, and prompt-injection guardrails. Adversarial DeepEval suite (cross-provider LLM-as-judge) scored faithfulness=0.99 across 26 adversarial probes — model grounds to retrieved context and refuses rather than hallucinating on off-topic queries.

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

Three-layer guardrail stack protecting both the input and output surfaces:

**Input (user queries)**
- **Prompt-Guard-86M** (primary) — Meta's DeBERTa-based classifier trained on injection/jailbreak examples. Catches semantic paraphrases and obfuscated variants that regex misses. Returns a probability score; queries above 0.85 are rejected. Configurable via `PROMPT_GUARD_THRESHOLD` env var.
- **Regex fallback** — 20+ pattern rules (role overrides, jailbreak keywords, delimiter smuggling, exfiltration phrases) used when the model is unavailable. Unicode NFC-normalized before matching.
- Length cap: queries over 500 characters are rejected with HTTP 400.

**Retrieved content (titles, abstracts injected into prompts)**
- Regex sanitizer replaces matched injection patterns with `[blocked]` before the text reaches the LLM. Field length capped at 2000 chars.

**Output (LLM responses)**
- **`ArxivCitationValidator`** — extracts any arxiv IDs cited in the response and checks them against the set of IDs from retrieved documents. IDs not found in the retrieval context are flagged as likely hallucinated; a disclaimer is appended.
- **`ToxicLanguageValidator`** — `unitary/toxic-bert` (110M params) scores the response; flagged responses above threshold 0.80 are rejected. Falls back to pass-through if the model is unavailable.

All validators follow a `PassResult` / `FailResult` contract (mirroring the Guardrails AI validator interface) and are tested offline via monkeypatched fixtures — no GPU or API key needed to run the test suite.

---

## Evaluation

Scored with [DeepEval](https://github.com/confident-ai/deepeval). Answer model: `gpt-5.4`. Judge: `claude-haiku-4-5` (cross-provider — avoids same-model inflation). Raw scores: [`evaluation/eval_metrics_snapshot.json`](evaluation/eval_metrics_snapshot.json).

Eval DB: ~185 indexed papers. RAG cases use 15 paper-specific queries targeting named papers (BERT-as-a-Judge, RecaLLM, VisionFoundry, VISOR, etc.) — each query has exactly one target paper, so the eval measures whether retrieval surfaces and ranks the right one. Adversarial cases cover 26 probes across two axes: completely non-CS domains (medicine, law, cosmology, climate science, music theory, archaeology) and within-CS subfields not in the corpus (cryptography, distributed systems, networking, OS, compilers, formal verification) — the latter are harder since the model has strong parametric knowledge there.

| Suite | Metric | Score | n | Notes |
| --- | --- | ---: | ---: | --- |
| RAG | Contextual precision | **1.000** ± 0.000 | 12 | target paper ranks above off-topic chunks every time; 3/15 unscored (judge returned invalid JSON, not a retrieval miss) |
| RAG | Answer relevancy | **0.985** ± 0.057 | 12 | grounded answer quality; min 0.78 |
| Adversarial RAG | Faithfulness | **0.988** ± 0.042 | 26 | stays grounded to off-topic context; 26 probes across two axes — non-CS domains (medicine, law, climate, physics) and within-CS subfields the LLM knows well (cryptography, distributed systems, compilers) |
| Adversarial RAG | Answer relevancy | **0.560** ± 0.406 | 26 | higher than prior run because gpt-5.4 occasionally gives partially useful grounded answers; faithfulness (not relevancy) is the primary signal here |
| No-context baseline | Answer relevancy | 0.995 ± 0.012 | 5 | gpt-5.4 from parametric knowledge; retrieval value strongest on niche/recent papers |

**Retrieval fix (before → after).** The BM25/FTS index is built on the chunk text only, and originally only the abstract was embedded — so paper *titles* were unsearchable. A query like *"What does BERT-as-a-Judge propose"* couldn't match the paper by title and `ContextualRelevancyMetric` scored **0.11** (the right paper rarely ranked into the top-k). Prepending the title to each indexed chunk (so both the dense embedding and BM25 see it) lifted retrieval to **1.00 contextual precision** — the target paper now ranks first. Contextual *precision* (does the relevant node rank above irrelevant ones?) is the correct metric for single-target queries; contextual *relevancy* has a ~1/k ceiling when only one of k retrieved papers matches.

The 3 skipped RAG cases are judge-side failures (`claude-haiku` occasionally returns invalid JSON for the precision metric), not retrieval failures — they are skipped rather than scored as 0.

```bash
uv run python -m evaluation.run_eval --suite all --samples 5 --write-metrics evaluation/eval_metrics_snapshot.json
uv run python -m evaluation.run_eval --suite rag
uv run python -m evaluation.run_eval --suite adversarial
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
├── guardrails/      # Input sanitizer + Prompt-Guard-86M classifier + output validators
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
