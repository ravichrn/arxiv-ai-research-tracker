# arxiv-ai-research-tracker

A research assistant that fetches the latest AI papers from arXiv, indexes them into a searchable vector store, and lets you explore them through a conversational multi-agent interface — all from the terminal.

---

## Features

### Multi-Agent Supervisor
Natural language routing to the right sub-agent automatically:

- *"Fetch recent NLP papers"* → fetches from arXiv, embeds, and stores
- *"List saved papers"* → shows all papers with arXiv IDs, citation counts, and TLDRs
- *"Find papers on diffusion models"* → hybrid search over local database
- *"Summarize recent robotics work"* → retrieves and batch-summarizes papers
- *"Summarize #2504.08123v2"* → summarizes a specific paper by arXiv ID
- *"Compare #2301.12345 and #2504.08123"* → structured side-by-side comparison
- *"Tag papers"* → clusters all papers into named research themes
- *"Daily digest"* or *"Digest last 14 days"* → newsletter-style digest with optional email delivery
- *"Diagram #2504.08123"* → Mermaid flowchart of the paper's methodology
- *"Get figures from #2504.08123"* → extracts real images/figures from the paper
- *"Export saved --bibtex"* → deterministic BibTeX/CSV export of saved papers (no LLM)
- *"Find papers on LLMs then explain"* → search + show which sources were retrieved
- *"Save tag #2504.08123 diffusion"* / *"Show tags #2504.08123"* → store and view user tags/notes per paper
- *"Trends last 14 days"* → rising arXiv categories over two adjacent time windows (no LLM)
- *"Fetch new ML papers then find the best ones on LLMs"* → chains fetch + search in one command

### Self-RAG Agent
The Q&A sub-agent actively verifies retrieval quality before responding:

1. **Document grading** — each retrieved chunk is scored for relevance; irrelevant results are filtered
2. **Query rewriting** — if retrieval quality is poor, the agent reformulates the query and retries (up to 2×)
3. **Hallucination checking** — the generated answer is verified against retrieved context; ungrounded answers trigger a retry with a disclaimer
4. **Source citations** — answers include a Sources block listing arXiv IDs and titles of retrieved papers

### Hybrid Search + Cross-Encoder Reranking
Combines dense vector similarity (OpenAI embeddings) with BM25 full-text search. Results are deduplicated by paper and reranked by a cross-encoder for higher precision. Pass `category_filter` to scope retrieval to a specific arXiv category.

### Semantic Scholar Enrichment
Papers are automatically enriched at fetch time via a single batch request to the Semantic Scholar API — no per-paper calls. Adds `s2_tldr` (one-sentence summary), `s2_citations` (citation count), and `s2_fields` (fields of study). Used by `list`, `summarize`, and `tag` commands. Best-effort and non-blocking.

### Auto-Tagging and Clustering
```
tag papers
```
Groups all fetched papers into named research themes. Uses S2 fields of study directly when available (≥80% coverage); falls back to LLM batch clustering over titles otherwise.

### Research Digest
```
daily digest
digest last 14 days
```
Newsletter-style executive summary grouped by arXiv category. If SMTP credentials are set in `.env`, the digest is also emailed automatically.

### Figure Extraction
```
get figures from #2504.08123
show images from #2301.12345
```
Tries arXiv HTML (ar5iv) first for figure URLs and captions, then falls back to PDF image extraction (saved to `databases/paper_figures/<arxiv_id>/`). Falls back to a Mermaid diagram if no figures are found. Distinct from `diagram`, which always generates Mermaid from the abstract.

### Mermaid Diagrams
```
diagram #2504.08123
```
Generates a `flowchart TD` Mermaid diagram of the paper's methodology from its abstract. Paste into any Mermaid renderer (GitHub, [mermaid.live](https://mermaid.live), VS Code).

### Paper Comparison
```
compare #2301.12345 and #2504.08123
```
Structured comparison across: problem & motivation, approach, results, limitations, and a verdict on when to prefer each. Falls back to top hybrid search results if no IDs are given.

### Export Metadata
Export paper metadata deterministically (no LLM calls):
```
export saved --bibtex
export saved --csv
export #2301.12345v2 --bibtex
export #2301.12345v2 --csv
```

### Explain Sources
Explain why the last RAG answer used certain sources (no LLM calls). Best used as a chain:
```
find papers on diffusion models then explain
```

### Saved Tags & Notes
Store user-controlled tags/notes for a paper, and reuse them for interest-aware ranking:
```
save tag #2301.12345v2 transformers
show tags #2301.12345v2
note #2301.12345v2 Important to read later
show note #2301.12345v2
```

### Trend Analysis
Compute rising categories over time windows (no LLM calls):
```
trends last 14 days
trends last 30 days
```

### Incremental Ingestion
A per-topic timestamp registry ensures only newly published papers are fetched on subsequent runs — no duplicates. Papers are assigned canonical arXiv IDs (e.g. `2504.08123v2`) usable across commands.

### Caching
- SQLite cache for LLM responses (shared across runs)
- Disk-backed embedding cache with 30-day TTL
- Anthropic prompt caching header when using Claude

### Multi-LLM Support
- **Summarizer** — Ollama (llama3.2) if available, falls back to OpenAI gpt-4o-mini
- **Agent** — OpenAI gpt-4o by default; set `AGENT_LLM=claude` for Anthropic Claude

### Streaming Output
`summarize`, `clarify`, `compare`, `tag`, `digest`, `diagram`, and `figures` stream tokens to the terminal as the model generates. `rag`, `ingestion`, and `list` print after completion.

### Persistent Conversation Memory
State is checkpointed to SQLite across restarts. Type `new session` or `reset` to start a fresh thread.

### Guardrails
All user input and retrieved content passes through a sanitization layer — detects prompt injection, jailbreak patterns, role overrides, and data exfiltration using 20+ regex rules with Unicode normalization.

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py                    # Entry point — calls launch_supervisor()
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
│   ├── run_eval.py            # Standalone eval runner
│   ├── test_summarizer.py     # pytest — hallucination + summarization metrics
│   └── test_rag.py            # pytest — faithfulness + relevancy metrics
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

Create a `.env` file using `.env.example` as a reference.

---

## Usage

```bash
uv run python main.py
```

Run the HTTP API:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

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

## Evaluation

```bash
uv run python -m evaluation.run_eval                        # all suites
uv run python -m evaluation.run_eval --suite rag            # RAG only
uv run python -m evaluation.run_eval --suite adversarial    # adversarial only
uv run pytest evaluation/                                   # pytest suite
```

**Judge model** — set `EVAL_JUDGE` in `.env`:

| `EVAL_JUDGE` | Judge | Cost |
|---|---|---|
| `prometheus` | Prometheus 2 via Ollama | Free (local) |
| `claude` | `claude-opus-4-6` | ~$6–10/run |
| `openai` | `gpt-4o` (default) | ~$1–2/run |

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
- [ ] Web interface with Streamlit/Gradio
- [ ] Full PDF analysis — fetch and analyze the complete paper, not just the abstract
- [ ] Slack digest delivery (alongside existing email delivery)
