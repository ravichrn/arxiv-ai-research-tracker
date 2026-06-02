import re

from langchain_core.tools import tool

from databases.interest_rerank import interest_aware_rerank
from databases.stores import hybrid_search, invalidate_fts_index, papers_store, saved_store
from guardrails.sanitizer import sanitize_retrieved

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
