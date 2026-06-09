import logging
import re

from langchain_core.tools import tool

from databases.citation_graph import get_edges, has_edges
from databases.interest_rerank import interest_aware_rerank
from databases.stores import hybrid_search, invalidate_fts_index, papers_store, saved_store
from guardrails.sanitizer import sanitize_retrieved

_log = logging.getLogger(__name__)

# arXiv URLs are strictly alphanumeric + a small set of punctuation — no quotes.
# This whitelist guards against unexpected values reaching the LanceDB delete filter.
_SAFE_URL_RE = re.compile(r"^https?://[a-zA-Z0-9._/:\-]+$")


def _format_paper(
    arxiv_id: str,
    title: str,
    authors: str,
    categories: str,
    published: str,
    url: str,
    abstract: str,
) -> str:
    pin_hint = f"  (use #{arxiv_id} for follow-up commands)" if arxiv_id else ""
    return (
        f"ArXiv ID : #{arxiv_id}{pin_hint}\n"
        f"Title    : {sanitize_retrieved(title)}\n"
        f"Authors  : {sanitize_retrieved(authors)}\n"
        f"Topics   : {categories or 'N/A'}\n"
        f"Published: {published or 'N/A'}\n"
        f"URL      : {sanitize_retrieved(url)}\n"
        f"\nAbstract : {sanitize_retrieved(abstract)}"
    )


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant papers found."
    return "\n\n".join(
        _format_paper(
            doc.metadata.get("arxiv_id", ""),
            str(doc.metadata.get("title", "")),
            str(doc.metadata.get("authors", "")),
            doc.metadata.get("categories", ""),
            doc.metadata.get("published", ""),
            str(doc.metadata.get("url", "")),
            str(doc.page_content),
        )
        for doc in docs
    )


def _format_live_results(papers: list[dict], source_label: str = "arXiv (live)") -> str:
    """Format raw paper dicts from a live arXiv search the same way as _format_docs."""
    if not papers:
        return "No relevant papers found on arXiv."
    header = f"[Results from {source_label} — indexed locally for future searches]\n\n"
    return header + "\n\n".join(
        _format_paper(
            p.get("arxiv_id", ""),
            p.get("title", ""),
            p.get("authors", ""),
            p.get("categories", ""),
            p.get("published", ""),
            p.get("url", ""),
            p.get("abstract", ""),
        )
        for p in papers
    )


def _expand_with_citations(query: str, seed_docs: list) -> list:
    """GraphRAG: expand seed results with 1-hop citation neighbors from the cache.

    Only reads the local SQLite citation cache — no S2 API calls, no extra latency.
    Papers whose lineage has been fetched (via the lineage node) automatically enrich
    subsequent RAG searches with their citation neighbors.

    Caps expansion to 2 novel papers to keep context manageable.
    """
    if not seed_docs:
        return seed_docs

    # Collect base arxiv_ids from top-2 seed papers only.
    seed_ids: list[str] = []
    for doc in seed_docs[:2]:
        arxiv_id = doc.metadata.get("arxiv_id", "")
        if arxiv_id:
            seed_ids.append(arxiv_id.split("v")[0])

    if not seed_ids:
        return seed_docs

    # Gather cited arxiv_ids from cached reference edges.
    seed_url_set = {d.metadata.get("url", "") for d in seed_docs}
    expansion_ids: list[str] = []
    for seed_id in seed_ids:
        if has_edges(seed_id):
            for edge in get_edges(seed_id, "references", limit=5):
                cid = edge.get("cited_arxiv_id", "")
                if cid and cid not in seed_ids:
                    expansion_ids.append(cid)

    if not expansion_ids:
        return seed_docs

    # Look up each cited paper in the local index via its title.
    from ingestion.arxiv_fetcher import get_paper_by_arxiv_id

    novel_docs: list = []
    seen_ids = set(seed_ids)
    for arxiv_id in expansion_ids:
        if arxiv_id in seen_ids:
            continue
        paper = get_paper_by_arxiv_id(arxiv_id)
        if not paper:
            continue
        neighbor = hybrid_search(papers_store, paper["title"], k=1, rerank=False)
        if neighbor:
            neighbor_url = neighbor[0].metadata.get("url", "")
            if neighbor_url and neighbor_url not in seed_url_set:
                novel_docs.append(neighbor[0])
                seed_url_set.add(neighbor_url)
                seen_ids.add(arxiv_id)
        if len(novel_docs) >= 2:
            break

    if novel_docs:
        _log.info(
            "[GraphRAG] expanded %d seed(s) with %d citation neighbor(s)",
            len(seed_ids),
            len(novel_docs),
        )

    return seed_docs + novel_docs


@tool
def search_papers(query: str) -> str:
    """Search AI papers — checks local index first, then falls back to live arXiv search."""
    docs = hybrid_search(papers_store, query, k=3)
    docs = interest_aware_rerank(query, docs)
    docs = _expand_with_citations(query, docs)
    if docs:
        return _format_docs(docs)

    # Nothing in the local index — query arXiv directly, then persist results.
    from ingestion.arxiv_fetcher import save_and_index_papers, search_arxiv_live

    live = search_arxiv_live(query, k=5)
    if live:
        save_and_index_papers(live)
    return _format_live_results(live)


@tool
def search_saved_papers(query: str) -> str:
    """Search your saved AI papers collection."""
    docs = hybrid_search(saved_store, query, k=3)
    docs = interest_aware_rerank(query, docs)
    return _format_docs(docs)


@tool
def add_paper_to_saved(title: str) -> str:
    """Save a paper from the recent papers collection by title."""
    docs = hybrid_search(papers_store, title, k=1)
    if not docs:
        return "Paper not found in current papers."
    url = docs[0].metadata.get("url", "")
    if url and _SAFE_URL_RE.match(url):
        tbl = saved_store.get_table()
        if tbl.count_rows() > 0:
            try:
                matches = tbl.search().where(f"url = '{url}'", prefilter=True).limit(1).to_list()
                if matches:
                    return f"'{docs[0].metadata.get('title')}' is already in saved papers."
            except Exception:
                # Fall back to vector search on LanceDB API incompatibility
                existing = hybrid_search(saved_store, url, k=1)
                if existing and existing[0].metadata.get("url") == url:
                    return f"'{docs[0].metadata.get('title')}' is already in saved papers."
    saved_store.add_documents(docs)
    invalidate_fts_index(saved_store)
    return f"Added '{docs[0].metadata.get('title')}' to saved papers."


@tool
def delete_paper_from_saved(title: str) -> str:
    """Delete a paper from the saved papers collection by title."""
    docs = hybrid_search(saved_store, title, k=1)
    if docs:
        matched_title = docs[0].metadata.get("title", "")
        url = docs[0].metadata.get("url", "")
        if url and _SAFE_URL_RE.match(url):
            # Preferred path: delete by URL. arXiv URLs are safe (no quotes).
            saved_store.get_table().delete(f"url = '{url}'")
        else:
            # Fallback: delete by title with standard SQL single-quote escaping.
            safe_title = matched_title.replace("'", "''")
            saved_store.get_table().delete(f"title = '{safe_title}'")
        invalidate_fts_index(saved_store)
        return f"Deleted '{matched_title}' from saved papers."
    return "Paper not found in saved papers."
