import arxiv
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
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


def summarize_text(text: str) -> str:
    prompt = _get_prompt().replace("{abstract}", text)
    response = llm_fast.invoke(prompt)
    return str(response.content).strip()


def _get_since() -> datetime:
    if LAST_RUN_FILE.exists():
        try:
            return datetime.fromisoformat(LAST_RUN_FILE.read_text().strip())
        except (ValueError, OSError):
            print("Warning: last_run.txt is corrupt — defaulting to 30 days ago.")
    return datetime.now(timezone.utc) - timedelta(days=14)


def _save_last_run() -> None:
    LAST_RUN_FILE.write_text(datetime.now(timezone.utc).isoformat())


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


def _build_doc(result: arxiv.Result) -> Document:
    """Build a Document from an arXiv result. Safe to run in a thread.

    Summary is intentionally excluded — the abstract (page_content) is the
    source of truth. Summaries are generated on demand when the user asks.
    """
    return Document(
        page_content=result.summary,
        metadata={
            "title": result.title,
            "authors": ", ".join(a.name for a in result.authors),
            "url": result.entry_id,
            "categories": ", ".join(result.categories),
            "published": result.published.isoformat(),
        },
    )


def fetch_and_summarize_papers(max_per_topic: int = 20, topics: list[str] | None = None) -> None:
    topics = topics or list(TOPICS)
    since = _get_since()
    since_str = since.strftime("%Y%m%d%H%M%S")
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    print(f"Fetching papers since {since.strftime('%Y-%m-%d %H:%M UTC')} across {len(topics)} topics...")

    # Load all existing URLs once — avoids one DB query per candidate paper.
    existing_urls = _fetch_existing_urls()
    seen_ids: set[str] = set()
    new_count = 0

    for i, topic in enumerate(topics):
        if i > 0:
            time.sleep(5)  # pause between topics to avoid rate-limiting

        query = f"cat:{topic} AND submittedDate:[{since_str} TO {now_str}]"
        search = arxiv.Search(
            query=query,
            max_results=max_per_topic,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # Collect candidates (skip already-known papers)
        candidates = []
        try:
            for result in _ARXIV_CLIENT.results(search):
                if result.entry_id in seen_ids or result.entry_id in existing_urls:
                    continue
                seen_ids.add(result.entry_id)
                candidates.append(result)
        except arxiv.HTTPError as e:
            print(f"\n[{topic}] skipped — arXiv rate limit (HTTP {e.status}). Try again later.")
            continue

        if not candidates:
            print(f"\n[{topic}] no new papers.")
            continue

        # Summarize candidates in parallel — each is an independent LLM call.
        docs: list[Document] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_build_doc, r): r for r in candidates}
            for future in as_completed(futures):
                result = futures[future]
                try:
                    doc = future.result()
                    docs.append(doc)
                    existing_urls.add(result.entry_id)
                    _print_paper(result)
                except Exception as e:
                    print(f"  [error] {result.title[:60]}: {e}")

        if docs:
            papers_store.add_documents(docs)

        new_count += len(docs)
        print(f"\n[{topic}] added {len(docs)} new papers.")

    _save_last_run()
    print(f"\n{'=' * 70}")
    print(f"Total: {new_count} new papers added. Database updated.")
