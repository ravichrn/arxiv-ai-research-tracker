import re

from langchain_core.tools import tool

from databases.interest_rerank import interest_aware_rerank
from databases.stores import hybrid_search, invalidate_fts_index, papers_store, saved_store
from guardrails.sanitizer import sanitize_retrieved

# arXiv URLs are strictly alphanumeric + a small set of punctuation — no quotes.
# This whitelist guards against unexpected values reaching the LanceDB delete filter.
_SAFE_URL_RE = re.compile(r"^https?://[a-zA-Z0-9._/:\-]+$")


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant papers found."
    results = []
    for doc in docs:
        meta = doc.metadata
        arxiv_id = meta.get("arxiv_id", "")
        pin_hint = f"  (use #{arxiv_id} for follow-up commands)" if arxiv_id else ""
        results.append(
            f"ArXiv ID : #{arxiv_id}{pin_hint}\n"
            f"Title    : {sanitize_retrieved(str(meta.get('title', '')))}\n"
            f"Authors  : {sanitize_retrieved(str(meta.get('authors', '')))}\n"
            f"Topics   : {meta.get('categories', 'N/A')}\n"
            f"Published: {meta.get('published', 'N/A')}\n"
            f"URL      : {sanitize_retrieved(str(meta.get('url', '')))}\n"
            f"\nAbstract : {sanitize_retrieved(str(doc.page_content))}"
        )
    return "\n\n".join(results)


def _format_live_results(papers: list[dict], source_label: str = "arXiv (live)") -> str:
    """Format raw paper dicts from a live arXiv search the same way as _format_docs."""
    if not papers:
        return "No relevant papers found on arXiv."
    results = []
    for p in papers:
        arxiv_id = p.get("arxiv_id", "")
        pin_hint = f"  (use #{arxiv_id} for follow-up commands)" if arxiv_id else ""
        results.append(
            f"ArXiv ID : #{arxiv_id}{pin_hint}\n"
            f"Title    : {sanitize_retrieved(p.get('title', ''))}\n"
            f"Authors  : {sanitize_retrieved(p.get('authors', ''))}\n"
            f"Topics   : {p.get('categories', 'N/A')}\n"
            f"Published: {p.get('published', 'N/A')}\n"
            f"URL      : {sanitize_retrieved(p.get('url', ''))}\n"
            f"\nAbstract : {sanitize_retrieved(p.get('abstract', ''))}"
        )
    header = f"[Results from {source_label} — indexed locally for future searches]\n\n"
    return header + "\n\n".join(results)


@tool
def search_papers(query: str) -> str:
    """Search AI papers — checks local index first, then falls back to live arXiv search."""
    docs = hybrid_search(papers_store, query, k=3)
    docs = interest_aware_rerank(query, docs)
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
    if docs:
        saved_store.add_documents(docs)
        invalidate_fts_index(saved_store)
        return f"Added '{docs[0].metadata.get('title')}' to saved papers."
    return "Paper not found in current papers."


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
