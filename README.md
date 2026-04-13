# arxiv-ai-research-tracker

A research assistant that fetches the latest AI papers from arXiv, indexes them into a searchable vector store, and lets you explore them through a conversational multi-agent interface — all from the terminal.

---

## Features

### Multi-Agent Supervisor
The entry point is a natural language supervisor that understands your intent and routes it to the right sub-agent automatically. You no longer pick topics from a menu — just describe what you want:

- *"Fetch recent NLP papers"* → triggers ingestion for cs.CL
- *"Find papers on diffusion models"* → searches the local database
- *"Summarize recent robotics work"* → retrieves and batch-summarizes papers
- *"Fetch new ML papers then find the best ones on LLMs"* → chains fetch + search in one command

The supervisor uses an LLM to extract intent, resolve topic aliases (e.g. "robotics" → cs.RO), and build an ordered execution chain for multi-step requests.

### Self-RAG Agent
The Q&A sub-agent doesn't just retrieve and respond — it actively verifies the quality of what it returns:

1. **Document grading** — each retrieved chunk is scored for relevance before the answer is generated; irrelevant results are filtered out
2. **Query rewriting** — if retrieval quality is poor, the agent automatically reformulates the query and retries (up to 2 times)
3. **Hallucination checking** — the generated answer is checked against the retrieved context; if it's not grounded, the agent rewrites and retries before delivering a response with a disclaimer

### Hybrid Search
Retrieval combines dense vector similarity (OpenAI embeddings) with BM25 full-text search using LanceDB's hybrid query mode. Results are fused and deduplicated at the paper level, so you get the best chunk per paper rather than repeated chunks from the same abstract.

### Semantic Chunking
Paper abstracts are split at natural semantic boundaries rather than fixed token windows. This means each stored chunk represents a coherent idea, improving retrieval precision for longer abstracts.

### Incremental Ingestion
The tracker remembers what it has already fetched using a per-topic timestamp registry. Each arXiv category gets its own timestamp, and a single "all" key supersedes individual ones when all topics are fetched together. On subsequent runs, only newly published papers since the last fetch are retrieved — no redundant API calls or duplicate entries.

### Resilient Fetching
Rate limits are handled automatically via exponential backoff (tenacity) on both arXiv requests and OpenAI embedding calls. Between topics, the fetcher pauses to stay within arXiv's recommended rate. If a topic fails after retries, it's skipped gracefully and the rest continue.

### Three-Layer Caching
LLM responses, embeddings, and Anthropic prompt prefixes are all cached to minimize redundant API calls:
- SQLite cache for LLM responses (shared across runs)
- Disk-backed embedding cache with a 30-day TTL using diskcache
- Anthropic prompt caching header for Claude-based deployments

### Multi-LLM Support
- **Summarizer** — uses a local Ollama model (llama3.2) if available, falls back to OpenAI gpt-4o-mini
- **Agent** — OpenAI gpt-4o by default; switch to Anthropic Claude by setting `AGENT_LLM=claude`

### Guardrails
All user input and retrieved content passes through a sanitization layer before reaching the LLM. It detects and blocks prompt injection attempts, jailbreak patterns, role overrides, and data exfiltration requests using compiled regex patterns with Unicode normalization.

### Evaluation Suite
DeepEval-based evaluation tests cover summarizer hallucination, answer faithfulness, and retrieval relevancy — runnable standalone or via pytest.

---

## Project Structure

```
arxiv-ai-research-tracker/
├── main.py                    # Entry point — calls launch_supervisor()
├── agents/
│   ├── supervisor.py          # Multi-agent supervisor: routes fetch/rag/summarize, supports chaining
│   ├── runner.py              # Self-RAG LangGraph agent (grade_docs → agent → hallucination_check)
│   └── tools.py               # search_papers, search_saved_papers, add/delete saved
├── ingestion/
│   └── arxiv_fetcher.py       # arXiv fetching, incremental sync, semantic chunking
├── databases/
│   └── stores.py              # LanceDB stores, hybrid search, LLM singletons, caching
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
  - 'fetch recent robotics papers'
  - 'find papers on diffusion models'
  - 'fetch NLP papers then find the best on transformers'
  - 'summarize recent cs.AI papers'

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
uv run python -m evaluation.run_eval             # default sample size
uv run python -m evaluation.run_eval --samples 5 # custom sample size
```

### pytest suite

```bash
uv run pytest evaluation/                        # all eval tests
uv run pytest evaluation/test_summarizer.py      # summarizer only
uv run pytest evaluation/test_rag.py             # RAG only
```

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
