# arxiv-ai-research-tracker

A research assistant that fetches the latest AI papers from arXiv, indexes them into a searchable vector store, and lets you explore them through a conversational multi-agent interface — all from the terminal.

---

## Features

### Multi-Agent Supervisor
The entry point is a natural language supervisor that understands your intent and routes it to the right sub-agent automatically. You no longer pick topics from a menu — just describe what you want:

- *"Fetch recent NLP papers"* → fetches from arXiv, embeds, and stores — all in one step
- *"List saved papers"* → shows all fetched papers with their arXiv IDs and titles
- *"Find papers on diffusion models"* → searches the local database
- *"Summarize recent robotics work"* → retrieves and batch-summarizes papers
- *"Summarize #2504.08123v2"* → summarizes that specific paper directly from its abstract
- *"Fetch new ML papers then find the best ones on LLMs"* → chains fetch + search in one command

The supervisor uses an LLM to extract intent, resolve topic aliases (e.g. "robotics" → cs.RO), and build an ordered execution chain for multi-step requests. It also resolves `#<arxiv_id>` references so you can target a specific paper by its canonical arXiv ID.

### Self-RAG Agent
The Q&A sub-agent doesn't just retrieve and respond — it actively verifies the quality of what it returns:

1. **Document grading** — each retrieved chunk is scored for relevance before the answer is generated; irrelevant results are filtered out
2. **Query rewriting** — if retrieval quality is poor, the agent automatically reformulates the query and retries (up to 2 times)
3. **Hallucination checking** — the generated answer is checked against the retrieved context; if it's not grounded, the agent rewrites and retries before delivering a response with a disclaimer

### Hybrid Search + Cross-Encoder Reranking
Retrieval combines dense vector similarity (OpenAI embeddings) with BM25 full-text search using LanceDB's hybrid query mode. Results are fused and deduplicated at the paper level, so you get the best chunk per paper rather than repeated chunks from the same abstract.

After deduplication, a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks the candidates by scoring each (query, document) pair jointly — far more accurate than bi-encoder similarity for final selection. The model is ~23 MB, runs CPU-only, and is loaded lazily on first use. Pass `rerank=False` to `hybrid_search()` to skip reranking when order doesn't matter (e.g. batch summarization).

An optional `category_filter` parameter (e.g. `"cs.RO"`) can scope retrieval to a specific arXiv category — used by the evaluation suite to avoid cross-topic dilution when scoring retrieval precision for topic-specific queries.

### Source Citations
RAG answers automatically include a **Sources** block listing the arxiv IDs and titles of every paper the answer drew from, e.g.:

```
Sources:
  [2301.12345v2] Attention Is All You Need
  [2504.08123v1] Scaling Laws for Neural Language Models
```

Citations are parsed from the `retrieval_context` state by `_format_citations()` in `agents/supervisor.py` and appended only when the Self-RAG agent actually retrieved and graded relevant documents.

### Fast Chunking
Paper abstracts are split using `RecursiveCharacterTextSplitter` with a 2000-character limit. arXiv abstracts are typically 150–300 words, so almost every abstract is stored as a single chunk — no unnecessary splitting. A character-based splitter is used deliberately: the previous semantic chunker embedded every sentence per abstract to find split boundaries, adding ~150–300 uncached API calls per 20-paper fetch with no retrieval benefit for short texts.

### Incremental Ingestion
The tracker remembers what it has already fetched using a per-topic timestamp registry. Each arXiv category gets its own timestamp, and a single "all" key supersedes individual ones when all topics are fetched together. On subsequent runs, only newly published papers since the last fetch are retrieved — no redundant API calls or duplicate entries.

Each paper is assigned its canonical arXiv ID (e.g. `2504.08123v2`) at fetch time. This ID is stored in `databases/papers_raw.jsonl` and as `arxiv_id` metadata in LanceDB, making it possible to reference, retrieve, or relate individual papers by their stable identifier across sessions.

### Resilient Fetching
Rate limits are handled automatically via exponential backoff (tenacity) on both arXiv requests and OpenAI embedding calls. Between topics, the fetcher pauses 3 seconds to stay within arXiv's recommended rate. If a topic fails after retries, it's skipped gracefully and the rest continue.

### Three-Layer Caching
LLM responses, embeddings, and Anthropic prompt prefixes are all cached to minimize redundant API calls:
- SQLite cache for LLM responses (shared across runs)
- Disk-backed embedding cache with a 30-day TTL using diskcache
- Anthropic prompt caching header for Claude-based deployments

### Multi-LLM Support
- **Summarizer** — uses a local Ollama model (llama3.2) if available, falls back to OpenAI gpt-4o-mini
- **Agent** — OpenAI gpt-4o by default; switch to Anthropic Claude by setting `AGENT_LLM=claude`

### Streaming Output
The supervisor streams LLM tokens directly to the terminal for long-running nodes. `summarize` and `clarify` node responses appear word-by-word as the model generates them — no waiting for the full response. For `rag`, `ingestion`, and `list` nodes, the result is printed after the agent completes. This uses LangGraph's `stream_mode=["values", "messages"]` API.

### Persistent Conversation Memory
Conversation state is checkpointed to `databases/agent_memory.db` via LangGraph's `SqliteSaver`. Each session continues from where the previous one left off — the agent remembers prior queries, fetched topics, and context across program restarts. Type `new session` or `reset` at any prompt to start a fresh conversation thread.

### Guardrails
All user input and retrieved content passes through a sanitization layer before reaching the LLM. It detects and blocks prompt injection attempts, jailbreak patterns, role overrides, and data exfiltration requests using compiled regex patterns with Unicode normalization.

### Evaluation Suite
DeepEval-based evaluation with four suites — runnable standalone or via pytest:

- **RAG eval** (10 cases) — faithfulness, answer relevancy, and retrieval relevancy across cs.AI, cs.LG, cs.CL, and cs.RO, each category-scoped to avoid cross-topic dilution
- **Adversarial eval** (3 cases) — queries whose topic is intentionally absent from the retrieved category; the LLM must stay grounded and not confabulate from prior knowledge
- **No-context baseline** — same RAG queries answered without retrieval, to quantify what the pipeline adds over raw LLM knowledge
- **Summarizer eval** (6 cases) — hallucination and coverage checks on known paper abstracts

**Tiered judge model** — set `EVAL_JUDGE` in `.env`:

| `EVAL_JUDGE` | Judge | Cost | When to use |
|---|---|---|---|
| `prometheus` | Prometheus 2 via Ollama | Free (local) | Daily iteration — fine-tuned for faithfulness/hallucination |
| `claude` | `claude-opus-4-6` via Anthropic | ~$6–10/run | Final portfolio eval — cross-provider, most credible |
| `openai` | `EVAL_JUDGE_MODEL` (default `gpt-4o`) | ~$1–2/run | Fallback when no Anthropic key |

Auto-selects `claude` if `ANTHROPIC_API_KEY` is set, otherwise `openai`. Prometheus 2 requires a one-time `ollama pull vicgalle/prometheus-7b-v2.0` (~4.4 GB).

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py                    # Entry point — calls launch_supervisor()
├── agents/
│   ├── supervisor.py          # Multi-agent supervisor: routes fetch/list/rag/summarize, supports chaining
│   ├── runner.py              # Self-RAG LangGraph agent (grade_docs → agent → hallucination_check)
│   └── tools.py               # search_papers, search_saved_papers, add/delete saved
├── ingestion/
│   └── arxiv_fetcher.py       # arXiv fetching, incremental sync, character chunking, embed + store
├── databases/
│   ├── stores.py              # LanceDB stores, hybrid search, LLM singletons, caching
│   ├── papers_raw.jsonl       # NDJSON cache of all fetched paper metadata (arxiv_id, title, abstract, …)
│   └── last_run.txt           # Per-topic fetch timestamps (JSON)
├── guardrails/
│   └── sanitizer.py           # Prompt injection prevention (20+ patterns)
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

### Environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

---

## Usage

```bash
uv run python main.py
```

The supervisor starts immediately and accepts natural language:

```
[Supervisor Ready] Ask me to fetch papers, search, summarize, or chain tasks.
Examples:
  - 'fetch recent robotics papers'          (fetch + embed in one step)
  - 'list saved papers'                     (show all papers with arXiv IDs)
  - 'find papers on diffusion models'
  - 'summarize #2504.08123v2'               (summarize a specific paper by arXiv ID)
  - 'fetch NLP papers then find the best on transformers'

You:
```

Type `exit` or `quit` to stop.

### Using a local Ollama model for summarization

```bash
ollama pull llama3.2
uv run python main.py
```

Ollama is auto-detected at `http://localhost:11434`. If unavailable, the summarizer falls back to `gpt-4o-mini`.

---

## Evaluation

### Standalone runner

```bash
uv run python -m evaluation.run_eval                        # all suites
uv run python -m evaluation.run_eval --suite rag            # RAG only
uv run python -m evaluation.run_eval --suite adversarial    # adversarial only
uv run python -m evaluation.run_eval --suite baseline       # no-context baseline
uv run python -m evaluation.run_eval --samples 5            # custom summarizer sample size
```

### pytest suite

```bash
uv run pytest evaluation/                        # all eval tests
uv run pytest evaluation/test_summarizer.py      # summarizer only
uv run pytest evaluation/test_rag.py             # RAG only
```

> **Note:** DeepEval metrics make LLM calls. Set `EVAL_JUDGE=prometheus` in `.env` for free local judging via Ollama (requires `ollama pull vicgalle/prometheus-7b-v2.0`). Set `EVAL_JUDGE=claude` with `ANTHROPIC_API_KEY` for final cross-provider evaluation. See `EVAL_JUDGE` in `.env.example` for all options.

---

## Tracing with LangSmith

1. Create an account at [smith.langchain.com](https://smith.langchain.com)
2. Generate an API key from Settings → API Keys
3. Add to `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   ```
   The project name is pre-set to `arxiv-ai-research-tracker` in `main.py` via `os.environ.setdefault`.
4. Run `main.py` — all LLM calls, tool invocations, and agent steps appear in the LangSmith dashboard automatically.

---

## Security

All user input and retrieved content passes through `guardrails/sanitizer.py` before reaching the LLM:

- Detects and blocks prompt injection, role overrides, jailbreak keywords, and data exfiltration patterns using 20+ compiled regex rules with Unicode normalization
- Queries over 500 characters are rejected outright
- Retrieved fields are truncated and matched content is replaced with `[blocked]`
- LanceDB deletes use parameterized-style escaping to prevent SQL injection
- API keys are never hardcoded — loaded from `.env` only

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
- [ ] Auto-tag papers by research area ("LLM", "Vision", "RL")
- [ ] Compare and critique two or more related papers
- [ ] Email/Slack digest of new papers
