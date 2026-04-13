import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import arxiv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from databases.stores import llm_fast, papers_store

LAST_RUN_FILE = Path(__file__).parent.parent / "databases" / "last_run.txt"
TOPICS = ["cs.AI", "cs.LG", "cs.CL", "cs.RO"]

# arXiv rate limit: 1 request per 3 seconds recommended for bulk access.
_ARXIV_CLIENT = arxiv.Client(
    page_size=20,
    delay_seconds=3.0,
    num_retries=5,
)

# Prompt loaded lazily — only read from disk when summarisation actually runs.
_SUMMARIZE_PROMPT: str | None = None


def _get_prompt() -> str:
    global _SUMMARIZE_PROMPT
    if _SUMMARIZE_PROMPT is None:
        _SUMMARIZE_PROMPT = (Path(__file__).parent.parent / "prompts" / "summarize.txt").read_text()
    return _SUMMARIZE_PROMPT


@retry(
    retry=retry_if_exception_type(arxiv.HTTPError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch_results(search: arxiv.Search) -> list[arxiv.Result]:
    """Fetch arXiv results with exponential backoff on HTTP errors (e.g. 429)."""
    return list(_ARXIV_CLIENT.results(search))


def summarize_text(text: str) -> str:
    prompt = _get_prompt().replace("{abstract}", text)
    response = llm_fast.invoke(prompt)
    return str(response.content).strip()


def _load_last_run() -> dict:
    """Return the last-run registry as a dict. Handles legacy plain-text format."""
    if not LAST_RUN_FILE.exists():
        return {}
    raw = LAST_RUN_FILE.read_text().strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Legacy format: single ISO timestamp — treat as an "all" entry.
    try:
        return {"all": raw}
    except Exception:
        return {}


def _get_since(topic: str) -> datetime:
    """Return the earliest datetime we should fetch for *topic*.

    Precedence (newest wins):
    1. ``"all"`` key — set when every topic was fetched in a single run.
    2. Topic-specific key (e.g. ``"cs.AI"``).
    If neither exists, fall back to 14 days ago.
    """
    registry = _load_last_run()
    fallback = datetime.now(UTC) - timedelta(days=14)

    candidates: list[datetime] = []
    for key in ("all", topic):
        ts = registry.get(key)
        if ts:
            try:
                candidates.append(datetime.fromisoformat(ts))
            except ValueError:
                print(f"Warning: corrupt timestamp for key '{key}' in last_run.txt — ignoring.")

    if not candidates:
        return fallback

    # Use the most recent timestamp — don't re-fetch papers already seen.
    return max(candidates)


def _save_last_run(topics: list[str], fetched_all: bool) -> None:
    """Persist per-topic (or global) timestamps.

    If *fetched_all* is True the ``"all"`` key is written and individual
    topic keys are removed — ``"all"`` supersedes them on the next run.
    Otherwise only the supplied topic keys are updated.
    """
    registry = _load_last_run()
    now = datetime.now(UTC).isoformat()

    if fetched_all:
        # "all" supersedes everything — drop stale per-topic keys.
        registry = {"all": now}
    else:
        # Update only the topics that were fetched this run.
        for topic in topics:
            registry[topic] = now
        # If an "all" key exists it would cause future per-topic fetches to
        # skip everything already covered.  Keep it — it's still valid.

    LAST_RUN_FILE.write_text(json.dumps(registry, indent=2))


def _fetch_existing_urls() -> set[str]:
    """Load all stored paper URLs into a set once — O(1) existence checks thereafter."""
    try:
        table = papers_store.get_table()
        rows = table.search().select(["url"]).to_list()
        return {row["url"] for row in rows if row.get("url")}
    except Exception:
        return set()


def _print_paper(result: arxiv.Result) -> None:
    authors = [a.name for a in result.authors]
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    print(f"\n{'─' * 70}")
    print(f"Title    : {result.title}")
    print(f"Authors  : {author_str}")
    print(f"Topics   : {', '.join(result.categories)}")
    print(f"Published: {result.published.strftime('%Y-%m-%d')}")
    print(f"URL      : {result.entry_id}")


def _get_chunker() -> SemanticChunker:
    """Build a SemanticChunker lazily — requires the embedding model to be available."""
    from langchain_openai import OpenAIEmbeddings

    return SemanticChunker(
        embeddings=OpenAIEmbeddings(),
        breakpoint_threshold_type="percentile",  # split where embedding distance > 95th pct
        breakpoint_threshold_amount=95,
    )


# Module-level singleton — constructed on first _build_docs call.
_chunker: SemanticChunker | None = None


def _build_docs(result: arxiv.Result) -> list[Document]:
    """Split an arXiv abstract into semantic chunks and return one Document per chunk.

    Each chunk inherits the paper's metadata plus chunk-level fields so that
    retrieval results can be deduplicated back to paper level.
    """
    global _chunker
    if _chunker is None:
        _chunker = _get_chunker()

    base_meta = {
        "title": result.title,
        "authors": ", ".join(a.name for a in result.authors),
        "url": result.entry_id,
        "categories": ", ".join(result.categories),
        "published": result.published.isoformat(),
    }

    chunks = _chunker.split_text(result.summary)
    # SemanticChunker may return the full text unsplit for short abstracts — treat as one chunk.
    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    **base_meta,
                    "chunk_id": f"{result.entry_id}#{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
        )
    return docs


def fetch_and_summarize_papers(max_per_topic: int = 20, topics: list[str] | None = None) -> None:
    topics = topics or list(TOPICS)
    fetched_all = set(topics) >= set(TOPICS)
    now_str = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    print(f"Fetching {len(topics)} topic(s)...")

    # Load all existing URLs once — avoids one DB query per candidate paper.
    existing_urls = _fetch_existing_urls()
    seen_ids: set[str] = set()
    new_count = 0

    for i, topic in enumerate(topics):
        if i > 0:
            time.sleep(5)  # pause between topics to avoid rate-limiting

        since = _get_since(topic)
        since_str = since.strftime("%Y%m%d%H%M%S")
        since_label = since.strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n[{topic}] fetching since {since_label}...")

        query = f"cat:{topic} AND submittedDate:[{since_str} TO {now_str}]"
        search = arxiv.Search(
            query=query,
            max_results=max_per_topic,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # Collect candidates (skip already-known papers)
        candidates = []
        try:
            for result in _fetch_results(search):
                if result.entry_id in seen_ids or result.entry_id in existing_urls:
                    continue
                seen_ids.add(result.entry_id)
                candidates.append(result)
        except arxiv.HTTPError as e:
            print(f"\n[{topic}] skipped after retries — arXiv HTTP {e.status}. Try again later.")
            continue

        if not candidates:
            print(f"\n[{topic}] no new papers.")
            continue

        # Build + chunk docs in parallel — each paper is an independent embedding call.
        docs: list[Document] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_build_docs, r): r for r in candidates}
            for future in as_completed(futures):
                result = futures[future]
                try:
                    chunks = future.result()
                    docs.extend(chunks)
                    existing_urls.add(result.entry_id)
                    _print_paper(result)
                    if len(chunks) > 1:
                        print(f"  → {len(chunks)} semantic chunks")
                except Exception as e:
                    print(f"  [error] {result.title[:60]}: {e}")

        if docs:
            papers_store.add_documents(docs)

        new_count += len(docs)
        print(f"\n[{topic}] added {len(docs)} new papers.")

    _save_last_run(topics, fetched_all)
    print(f"\n{'=' * 70}")
    print(f"Total: {new_count} new papers added. Database updated.")
