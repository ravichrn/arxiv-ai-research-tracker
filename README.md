# arxiv-ai-research-tracker

A research assistant that fetches the latest AI research papers from arXiv, enables follow-up Q&A via an intelligent agent, and helps you curate a personal collection of papers for future reference.

---

## Features

- **Topic-selective ingestion** — choose which arXiv categories to fetch (cs.AI, cs.LG, cs.CL, cs.RO) at startup; avoids unnecessary API calls
- **Incremental fetching** — tracks last-run timestamp in `databases/last_run.txt`; only fetches papers published since the previous run (default fallback: 14 days)
- **Multi-topic deduplication** — URL-based set lookup prevents the same paper being stored twice across overlapping topic feeds
- **arXiv rate-limit resilience** — custom `arxiv.Client` with 3 s delay between pages, 5 retries, 5 s pause between topics, and graceful per-topic 429 handling (skips topic, continues rest)
- **Multi-LLM routing**
  - Summarizer: Ollama/Llama (local, zero cost) → falls back to OpenAI `gpt-4o-mini`
  - Agent: OpenAI `gpt-4o` (default) or Anthropic Claude (set `AGENT_LLM=claude`)
- **LangGraph agent** — explicit `StateGraph` with inspectable agent → tools → agent loop; 10-turn conversation memory with automatic history trimming
- **Four agent tools** — `search_papers`, `search_saved_papers`, `add_paper_to_saved`, `delete_paper_from_saved`
- **LanceDB vector store** — embedded, columnar, SQL-style filters; replaces ChromaDB; single shared connection for both `papers` and `saved` tables
- **Three-layer caching**
  - SQLite LLM response cache (via `langchain_community.cache.SQLiteCache`)
  - Disk-backed embedding cache with 30-day TTL (via `diskcache`)
  - Anthropic prompt caching header (`anthropic-beta: prompt-caching-2024-07-31`)
- **Lazy loading** — all heavy packages (LanceDB, OpenAI, Ollama, embeddings) deferred to first use via `_LazyProxy`; startup time ~2 s, memory ~65 MB
- **Parallel ingestion** — documents built concurrently with `ThreadPoolExecutor(max_workers=4)`
- **LangSmith tracing** — zero-code auto-instrumentation via environment variables
- **DeepEval evaluation suite** — hallucination, faithfulness, and answer relevancy metrics with standalone runner and pytest integration

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py                    # Entry point — topic selection, ingestion, agent launch
├── agents/
│   ├── runner.py              # LangGraph StateGraph agent loop
│   └── tools.py               # search_papers, search_saved_papers, add/delete saved
├── ingestion/
│   └── arxiv_fetcher.py       # arXiv fetching, incremental sync, parallel doc building
├── databases/
│   └── stores.py              # LanceDB stores, LLM singletons, caching, lazy proxies
├── guardrails/
│   └── sanitizer.py           # Prompt injection prevention (20+ patterns)
├── evaluation/
│   ├── datasets.py            # Hardcoded test cases (summarizer + RAG)
│   ├── run_eval.py            # Standalone eval runner
│   ├── test_summarizer.py     # pytest — hallucination + summarization metrics
│   └── test_rag.py            # pytest — faithfulness + contextual relevancy metrics
├── prompts/
│   └── summarize.txt          # On-demand summarization prompt template
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

Create a `.env` file in the project root using `.env.example` file

---

## Execution

### Main app

```bash
uv run python main.py
```

At startup you will be prompted to select topics:

```
Available topics:
  [1] cs.AI — Artificial Intelligence
  [2] cs.LG — Machine Learning
  [3] cs.CL — Computation & Language (NLP / LLMs)
  [4] cs.RO — Robotics
  [5] All of the above

Enter topic numbers (e.g. 1 3) or press Enter for all:
```

After ingestion the agent starts. Type your question or `exit` to quit.

### Using a local Ollama model for summarization

```bash
ollama pull llama3.2
uv run python main.py
```

Ollama is auto-detected at `http://localhost:11434`. If unavailable, the summarizer falls back to `gpt-4o-mini`.

---

## Evaluation

### Standalone runner (no pytest, scores printed to terminal)

```bash
uv run python -m evaluation.run_eval             # 3 random papers + 3 RAG queries
uv run python -m evaluation.run_eval --samples 5 # custom paper sample size
```

### pytest suite

```bash
uv run pytest evaluation/                        # all eval tests
uv run pytest evaluation/test_summarizer.py      # summarizer only
uv run pytest evaluation/test_rag.py             # RAG only
```

Metrics tested:

| Test | Metric | Threshold |
|------|--------|-----------|
| Summarizer | HallucinationMetric | ≤ 0.4 |
| Summarizer | SummarizationMetric | ≥ 0.4 |
| RAG | FaithfulnessMetric | ≥ 0.7 |
| RAG | AnswerRelevancyMetric | ≥ 0.7 |
| RAG | ContextualRelevancyMetric | ≥ 0.5 |

> **Note:** DeepEval metrics make LLM calls. Ensure `OPENAI_API_KEY` is set before running evals.

---

## Tracing with LangSmith

1. Create an account at [smith.langchain.com](https://smith.langchain.com)
2. Generate an API key from Settings → API Keys
3. Add to `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   LANGCHAIN_PROJECT=arxiv-tracker
   ```
4. Run `main.py` — all LLM calls, tool invocations, and agent steps appear automatically in the LangSmith dashboard with no code changes required.

---

## Security

### Prompt injection prevention (`guardrails/sanitizer.py`)

All user input and retrieved content passes through a guardrail layer before reaching the LLM:

- **20+ compiled regex patterns** (case-insensitive, Unicode-aware) covering:
  - Role / identity override (`ignore previous instructions`, `act as`, `pretend you are`, …)
  - Instruction injection (`new instructions:`, `override prompt`, …)
  - Structural / delimiter injection (`<system>`, `[INST]`, `|im_start|`, ChatML tags, …)
  - Jailbreak keywords (`DAN`, `jailbroken`, `developer mode`, `god mode`, …)
  - Data exfiltration attempts (`reveal your system prompt`, `print your instructions`, …)
- **NFC Unicode normalization** before matching to defeat lookalike-character bypass attempts
- **Hard length caps** — queries over 500 chars rejected; retrieved fields truncated at 2 000 chars
- **Matched content replaced** with `[blocked]` in retrieved text; user queries raise `InputRejected`

### Additional security measures

- **No summary stored in vector DB** — abstracts only; summaries generated on demand, reducing stored LLM output surface
- **SQL injection prevention** — LanceDB delete uses single-quote escaping (`title.replace("'", "''")`)
- **Secrets via `.env`** — no API keys hardcoded; `.env` not committed

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

- [ ] Web interface with Streamlit/Gradio
- [ ] Auto-tag papers ("LLM", "Vision", "RL")
- [ ] Compare and critique two or more related papers
- [ ] Email/Slack digest of new papers
- [ ] Citation graph exploration
