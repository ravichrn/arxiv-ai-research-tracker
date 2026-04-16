import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import arxiv
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def _arxiv_id(result: arxiv.Result) -> str:
    """Return the short arXiv ID, e.g. '2301.12345v2'."""
    return result.get_short_id()


def get_paper_by_title(title: str) -> dict | None:
    """Look up a paper from papers_raw.jsonl by exact title (case-insensitive)."""
    title_lower = title.lower().strip()
    for paper in _load_papers_cache():
        if paper.get("title", "").lower().strip() == title_lower:
            return paper
    return None


def list_papers() -> str:
    """Return a formatted list of all saved papers with arXiv IDs and titles.

    Includes citation count and S2 TLDR when available from Semantic Scholar enrichment.
    """
    papers = _load_papers_cache()
    if not papers:
        return "No papers saved yet. Run a fetch first."
    lines = [f"Saved papers ({len(papers)} total):\n"]
    for p in papers:
        arxiv_id = p.get("arxiv_id") or p.get("url", "")
        lines.append(f"  [{arxiv_id}]  {p.get('title', '(no title)')}")
        citations = p.get("s2_citations")
        tldr = p.get("s2_tldr")
        if citations is not None or tldr:
            meta_parts = []
            if citations is not None:
                meta_parts.append(f"Citations: {citations:,}")
            if tldr:
                meta_parts.append(f"TL;DR: {tldr}")
            lines.append(f"    {' | '.join(meta_parts)}")
    lines.append("\nReference any paper with #<arxiv_id>, e.g. 'summarize #2504.08123v1'")
    return "\n".join(lines)


def get_paper_by_arxiv_id(arxiv_id: str) -> dict | None:
    """Look up a paper from papers_raw.jsonl by its arXiv ID (e.g. '2301.12345v2').

    Matches on both versioned ('2301.12345v2') and base ('2301.12345') forms.
    Returns the raw paper dict, or None if not found.
    """
    base_id = arxiv_id.split("v")[0]
    for paper in _load_papers_cache():
        url = paper.get("arxiv_id", "") or paper.get("url", "")
        if arxiv_id in url or base_id in url:
            return paper
    return None


def _print_paper(result: arxiv.Result) -> None:
    authors = [a.name for a in result.authors]
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    print(f"\n{'─' * 70}")
    print(f"[{_arxiv_id(result)}]")
    print(f"  Title    : {result.title}")
    print(f"  Authors  : {author_str}")
    print(f"  Topics   : {', '.join(result.categories)}")
    print(f"  Published: {result.published.strftime('%Y-%m-%d')}")


# arXiv abstracts are 150-300 words; chunk_size=2000 means almost all are a single chunk.
# RecursiveCharacterTextSplitter does no embedding — purely character-based, near-instant.
_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)


_RAW_PAPERS_FILE = Path(__file__).parent.parent / "databases" / "papers_raw.jsonl"
_DB_DIR = Path(__file__).parent.parent / "databases"

# Module-level cache for papers_raw.jsonl — avoids re-reading the file on every
# get_paper_by_title / get_paper_by_arxiv_id / list_papers call within a session.
# Invalidated after every write in fetch_papers so new papers are visible immediately.
_papers_cache: list[dict] | None = None


def _load_papers_cache() -> list[dict]:
    global _papers_cache
    if _papers_cache is not None:
        return _papers_cache
    if not _RAW_PAPERS_FILE.exists():
        _papers_cache = []
        return _papers_cache
    papers = []
    for line in _RAW_PAPERS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    _papers_cache = papers
    return _papers_cache


def _invalidate_papers_cache() -> None:
    global _papers_cache
    _papers_cache = None


def _load_raw_urls() -> set[str]:
    """Return the set of URLs already saved in papers_raw.jsonl."""
    return {p["url"] for p in _load_papers_cache() if p.get("url")}


def fetch_papers(max_per_topic: int = 20, topics: list[str] | None = None) -> int:
    """Fetch paper metadata from arXiv, print results, and save to papers_raw.jsonl.

    No embedding or LanceDB writes — returns immediately after arXiv API calls.
    Returns the number of new papers saved.
    """
    topics = topics or list(TOPICS)
    fetched_all = set(topics) >= set(TOPICS)
    now_str = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    print(f"Fetching {len(topics)} topic(s)...")

    # Skip papers already saved (raw file) or already indexed (LanceDB).
    raw_urls = _load_raw_urls()
    existing_urls = _fetch_existing_urls()
    known_urls = raw_urls | existing_urls
    seen_ids: set[str] = set()
    new_count = 0
    new_papers: list[dict] = []

    # Track the file offset before appending so we can rewrite enriched data later.
    append_offset = _RAW_PAPERS_FILE.stat().st_size if _RAW_PAPERS_FILE.exists() else 0

    with _RAW_PAPERS_FILE.open("a") as fh:
        for i, topic in enumerate(topics):
            if i > 0:
                time.sleep(3)  # arXiv recommended minimum between bulk requests

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

            topic_count = 0
            try:
                for result in _fetch_results(search):
                    if result.entry_id in seen_ids or result.entry_id in known_urls:
                        continue
                    seen_ids.add(result.entry_id)
                    known_urls.add(result.entry_id)
                    paper = {
                        "arxiv_id": _arxiv_id(result),
                        "url": result.entry_id,
                        "pdf_url": str(result.pdf_url),
                        "title": result.title,
                        "authors": ", ".join(a.name for a in result.authors),
                        "abstract": result.summary,
                        "categories": ", ".join(result.categories),
                        "published": result.published.isoformat(),
                    }
                    _print_paper(result)
                    fh.write(json.dumps(paper) + "\n")
                    new_papers.append(paper)
                    topic_count += 1
            except arxiv.HTTPError as e:
                print(f"\n[{topic}] skipped after retries — arXiv HTTP {e.status}.")
                continue

            if topic_count == 0:
                print(f"\n[{topic}] no new papers.")
            else:
                print(f"\n[{topic}] {topic_count} new papers saved.")
            new_count += topic_count

    # Enrich new papers with Semantic Scholar metadata, then rewrite their JSONL lines.
    if new_papers:
        new_papers = enrich_with_s2(new_papers)
        # Truncate back to the pre-append offset and re-append enriched records.
        with _RAW_PAPERS_FILE.open("r+b") as fh:
            fh.truncate(append_offset)
        with _RAW_PAPERS_FILE.open("a") as fh:
            for paper in new_papers:
                fh.write(json.dumps(paper) + "\n")

    _invalidate_papers_cache()
    _save_last_run(topics, fetched_all)
    print(f"\n{'=' * 70}")
    print(f"Total: {new_count} new papers fetched.")
    if new_count > 0:
        print("Tip: Reference any paper above by its arXiv ID, e.g. 'summarize #2301.12345v2'")
        _embed_and_store(new_papers)
    return new_count


def get_recent_papers(days: int = 7) -> list[dict]:
    """Return papers published within the last *days* days, newest first.

    Reads from ``papers_raw.jsonl`` — no network calls. Uses the ``published``
    field (ISO 8601, stored during fetch) to filter by date.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    results: list[dict] = []
    for paper in _load_papers_cache():
        pub_raw = paper.get("published", "")
        if not pub_raw:
            continue
        try:
            pub_dt = datetime.fromisoformat(pub_raw)
            # Ensure timezone-aware for comparison
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=UTC)
            if pub_dt >= cutoff:
                results.append(paper)
        except ValueError:
            pass
    results.sort(key=lambda p: p.get("published", ""), reverse=True)
    return results


def fetch_paper_content(arxiv_id: str, pdf_url: str) -> dict | None:
    """Extract figures from a paper. Tries arXiv HTML first, falls back to PDF.

    Returns:
        {"figures": [...], "sections": [...], "source": "html"|"pdf"} or None if both fail.
        HTML figures: [{"url": str, "caption": str}]
        PDF figures:  [{"path": str, "page": int}]
    """
    # --- HTML path (ar5iv) ---
    try:
        html_url = f"https://html.arxiv.org/abs/{arxiv_id}"
        resp = requests.get(html_url, timeout=15)
        if resp.status_code == 200:
            from bs4 import BeautifulSoup  # lazy import

            soup = BeautifulSoup(resp.text, "html.parser")
            figures = []
            for fig in soup.find_all("figure"):
                img = fig.find("img")
                if not img or not img.get("src"):
                    continue
                img_url = urljoin(html_url, img["src"])
                caption_tag = fig.find("figcaption")
                caption = caption_tag.get_text(strip=True) if caption_tag else ""
                figures.append({"url": img_url, "caption": caption})
            sections = [
                s.get("id", s.get_text(strip=True))
                for s in soup.find_all("section")
                if s.get("id") or s.get_text(strip=True)
            ]
            # Only return HTML result if we actually found figures
            if figures:
                return {"figures": figures, "sections": sections[:20], "source": "html"}
    except Exception as e:
        print(f"[figures] HTML fetch failed: {e}")

    # --- PDF fallback ---
    try:
        pdf_resp = requests.get(pdf_url, timeout=30)
        if pdf_resp.status_code == 200:
            import fitz  # lazy import (pymupdf)

            doc = fitz.open(stream=pdf_resp.content, filetype="pdf")
            out_dir = _DB_DIR / "paper_figures" / arxiv_id
            out_dir.mkdir(parents=True, exist_ok=True)
            figures = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)
                    img_bytes = img_data["image"]
                    ext = img_data.get("ext", "png")
                    out_path = out_dir / f"fig_p{page_num}_n{img_index}.{ext}"
                    out_path.write_bytes(img_bytes)
                    figures.append({"path": str(out_path), "page": page_num})
            if figures:
                return {"figures": figures, "sections": [], "source": "pdf"}
    except Exception as e:
        print(f"[figures] PDF extraction failed: {e}")

    return None


def enrich_with_s2(papers: list[dict]) -> list[dict]:
    """Batch-enrich papers with Semantic Scholar metadata (TLDR, citation count, fields).

    Sends all arXiv IDs in one POST request to the S2 Graph API.
    Adds s2_tldr, s2_citations, s2_fields to each paper dict in-place.
    Best-effort: returns papers unchanged if S2 is unreachable.
    """
    if not papers:
        return papers

    ids = [f"arXiv:{p['arxiv_id']}" for p in papers if p.get("arxiv_id")]
    if not ids:
        return papers

    try:
        resp = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "tldr,citationCount,fieldsOfStudy"},
            json={"ids": ids},
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"[S2] enrichment skipped: HTTP {resp.status_code}")
            return papers

        results = resp.json()
        # Build lookup: base arXiv ID → S2 record
        s2_by_id: dict[str, dict] = {}
        for record in results:
            if not record:
                continue
            ext_ids = record.get("externalIds") or {}
            arxiv_raw = ext_ids.get("ArXiv") or ""
            if arxiv_raw:
                s2_by_id[arxiv_raw.split("v")[0]] = record

        enriched = 0
        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            base = arxiv_id.split("v")[0]
            record = s2_by_id.get(base)
            if record:
                tldr_obj = record.get("tldr")
                paper["s2_tldr"] = tldr_obj.get("text") if tldr_obj else None
                paper["s2_citations"] = record.get("citationCount")
                fields = record.get("fieldsOfStudy")
                paper["s2_fields"] = fields if fields else None
                enriched += 1
            else:
                paper["s2_tldr"] = None
                paper["s2_citations"] = None
                paper["s2_fields"] = None

        print(f"[S2] enriched {enriched}/{len(papers)} papers.")
    except Exception as e:
        print(f"[S2] enrichment skipped: {e}")
        for paper in papers:
            paper.setdefault("s2_tldr", None)
            paper.setdefault("s2_citations", None)
            paper.setdefault("s2_fields", None)

    return papers


def _embed_and_store(papers: list[dict]) -> int:
    """Embed a list of raw paper dicts and store them in LanceDB. Returns count stored."""
    if not papers:
        return 0
    print(f"Indexing {len(papers)} paper(s) into vector store...")
    docs: list[Document] = []
    for paper in papers:
        chunks = _splitter.split_text(paper["abstract"])
        for i, chunk_text in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "arxiv_id": paper.get("arxiv_id", ""),
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "url": paper["url"],
                        "categories": paper["categories"],
                        "published": paper["published"],
                        "chunk_id": f"{paper['url']}#{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )
    papers_store.add_documents(docs)
    print(f"Indexed {len(papers)} papers ({len(docs)} chunks) into vector store.")
    return len(papers)
