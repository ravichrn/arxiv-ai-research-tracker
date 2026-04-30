import logging
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from prometheus_client import Counter, Histogram
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_log = logging.getLogger(__name__)

load_dotenv()

_ROOT = Path(__file__).parent.parent
_DB_DIR = _ROOT / "databases"

# ---------------------------------------------------------------------------
# LLM response cache — SQLite, lightweight (<1ms setup), persists across runs.
# Must be initialised before any LLM is constructed.
# ---------------------------------------------------------------------------
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=str(_DB_DIR / "llm_cache.db")))


# ---------------------------------------------------------------------------
# Disk-backed embedding cache — same text never re-embedded across restarts.
# diskcache is pure Python, ~1ms import, supports TTL.
# ---------------------------------------------------------------------------
import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_EMBEDDING_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


class _CachedEmbeddings(Embeddings):
    _cache: "diskcache.Cache | None"  # type: ignore[name-defined]
    _base: Embeddings | None

    def __init__(self):
        self._cache = None
        self._base = None
        self._lock = threading.Lock()

    def _init(self):
        if self._base is None:
            with self._lock:
                if self._base is None:  # re-check after acquiring lock
                    import diskcache
                    from langchain_openai import OpenAIEmbeddings

                    self._base = OpenAIEmbeddings()
                    self._cache = diskcache.Cache(str(_DB_DIR / "embedding_cache"))

    @staticmethod
    def _is_rate_limit(exc: BaseException) -> bool:
        return "rate" in str(exc).lower() or "429" in str(exc)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_many_docs(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents in one call (preserves input order)."""
        assert self._base is not None
        return self._base.embed_documents(texts)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_one_query(self, text: str) -> list[float]:
        assert self._base is not None
        return self._base.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._init()
        assert self._cache is not None
        results: list[list[float] | None] = [None] * len(texts)

        # Cache key per text (so repeated inputs within one call don't re-embed).
        keys = [f"doc:{hashlib.sha256(text.encode()).hexdigest()}" for text in texts]
        missing_texts_by_key: dict[str, str] = {}
        missing_indices_by_key: dict[str, list[int]] = {}

        for i, (text, key) in enumerate(zip(texts, keys, strict=True)):
            if key in self._cache:
                results[i] = list(self._cache[key])
                _embedding_cache_hits.inc()
            else:
                missing_texts_by_key[key] = text
                missing_indices_by_key.setdefault(key, []).append(i)
                _embedding_cache_misses.inc()

        if missing_texts_by_key:
            missing_keys = list(missing_texts_by_key.keys())
            missing_texts = [missing_texts_by_key[k] for k in missing_keys]
            embeddings = self._embed_many_docs(missing_texts)

            for key, emb in zip(missing_keys, embeddings, strict=True):
                self._cache.set(key, emb, expire=_EMBEDDING_TTL_SECONDS)
                for idx in missing_indices_by_key[key]:
                    results[idx] = list(emb)

        # All cache misses are filled above; keep output length stable.
        assert all(r is not None for r in results)
        return [r for r in results]  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        self._init()
        assert self._cache is not None
        key = f"q:{hashlib.sha256(text.encode()).hexdigest()}"
        if key not in self._cache:
            _embedding_cache_misses.inc()
            self._cache.set(key, self._embed_one_query(text), expire=_EMBEDDING_TTL_SECONDS)
        else:
            _embedding_cache_hits.inc()
        return list(self._cache[key])


# ---------------------------------------------------------------------------
# Lazy proxy — defers construction to first attribute access.
# Keeps import time fast; heavy packages only load when actually needed.
# __call__ removed — LLMs are always used via .invoke(), not called directly.
# ---------------------------------------------------------------------------
class _LazyProxy:
    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_obj", None)
        object.__setattr__(self, "_lock", threading.Lock())

    def _get(self):
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                obj = object.__getattribute__(self, "_obj")
                if obj is None:
                    obj = object.__getattribute__(self, "_factory")()
                    object.__setattr__(self, "_obj", obj)
        return obj

    def __getattr__(self, name):
        return getattr(self._get(), name)


# ---------------------------------------------------------------------------
# Vector stores — single lancedb connection shared between both tables.
# Arrow/lance dependencies (~80MB) only load on first tool call or ingestion.
# ---------------------------------------------------------------------------
_cached_embeddings = _CachedEmbeddings()
_db_instance = None
_db_lock = threading.Lock()


def _get_db():
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                import lancedb

                _db_instance = lancedb.connect(str(_DB_DIR / "lancedb"))
    return _db_instance


def _ensure_fts_index(store) -> None:
    """Create (or refresh) a full-text search index on the 'text' column.

    Called once per store after the table is confirmed to have rows — creating
    an FTS index on an empty table is a no-op that silently fails in some
    LanceDB versions, so we guard with a row count check.
    """
    # Avoid repeating index creation attempts once the table is non-empty.
    # (But we only mark "done" after a successful index build.)
    key = id(store)
    if key in _FTS_INDEXED:
        return

    try:
        tbl = store.get_table()
        if tbl.count_rows() <= 0:
            return
        tbl.create_fts_index("text", replace=True, language="English", stem=True)
        _FTS_INDEXED.add(key)
    except Exception as e:
        _log.warning("[FTS] index creation skipped: %s", e)


_FTS_INDEXED: set[int] = set()


def invalidate_fts_index(store) -> None:
    """Mark a store's FTS index as stale so it is rebuilt on the next search.

    Call this after adding new documents to a store so hybrid_search picks
    them up via BM25 on the next query.
    """
    _FTS_INDEXED.discard(id(store))


def _make_papers_store():
    from langchain_community.vectorstores import LanceDB

    return LanceDB(connection=_get_db(), table_name="papers", embedding=_cached_embeddings)


def _make_saved_store():
    from langchain_community.vectorstores import LanceDB

    return LanceDB(connection=_get_db(), table_name="saved", embedding=_cached_embeddings)


papers_store = _LazyProxy(_make_papers_store)
saved_store = _LazyProxy(_make_saved_store)


# ---------------------------------------------------------------------------
# Cross-encoder reranker — lazy singleton, loads on first use.
# Model: ms-marco-MiniLM-L-6-v2 (~23 MB, CPU-only, no API key needed).
# Reranks (query, doc) pairs jointly — more accurate than bi-encoder similarity.
# ---------------------------------------------------------------------------
_reranker = None
_reranker_failed_until: float = 0.0  # epoch seconds; 0 means not failed
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER_RETRY_AFTER = 300  # seconds before retrying after a failure


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)
    return _reranker


def _rerank(query: str, docs: list) -> list:
    """Rerank docs by cross-encoder (query, doc) score descending.

    Falls back to original order on failure, prints a visible warning, and retries
    after _RERANKER_RETRY_AFTER seconds so transient failures (OOM, disk) self-heal.
    """
    global _reranker_failed_until, _reranker
    if not docs:
        return docs
    if _reranker_failed_until and time.monotonic() < _reranker_failed_until:
        return docs
    try:
        reranker = _get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs, strict=True), key=lambda x: x[0], reverse=True)
        _reranker_failed_until = 0.0  # clear backoff on success
        return [doc for _, doc in ranked]
    except Exception as e:
        _reranker = None  # allow re-init on next attempt
        _reranker_failed_until = time.monotonic() + _RERANKER_RETRY_AFTER
        _log.warning("[Reranker] skipped: %s", e)
        print(
            f"[Reranker] WARNING: cross-encoder unavailable ({e});"
            f"retrying in {_RERANKER_RETRY_AFTER}s."
        )
        return docs


def _safe_category_filter(category_filter: str | None) -> str | None:
    """Build a safe LanceDB LIKE filter expression for a category code.

    Keeps only alphanumeric characters, dots, and hyphens — the only characters
    that appear in valid arXiv category codes (e.g. cs.AI, eess.SP, q-bio.NC).
    This explicitly strips LIKE wildcards (% and _) and any other metacharacters.
    """
    if not category_filter:
        return None
    safe = re.sub(r"[^\w.\-]", "", category_filter)
    return f"metadata.categories LIKE '%{safe}%'" if safe else None


_search_latency = Histogram(
    "hybrid_search_duration_seconds",
    "End-to-end latency of hybrid_search() including reranking",
    ["store"],
)
_search_results = Histogram(
    "hybrid_search_result_count",
    "Number of results returned by hybrid_search()",
    ["store", "reranked"],
    buckets=[0, 1, 2, 3, 5, 10],
)
_embedding_cache_hits = Counter(
    "embedding_cache_hits_total",
    "Number of embedding cache hits",
)
_embedding_cache_misses = Counter(
    "embedding_cache_misses_total",
    "Number of embedding cache misses",
)


def _row_to_doc(row: dict) -> Document:
    """Convert a raw LanceDB row dict to a LangChain Document."""
    text = row.get("text", "")
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = dict(meta)
    return Document(page_content=text, metadata=meta)


def _rrf_merge(vec_rows: list[dict], fts_rows: list[dict], k_rrf: int = 60) -> list[dict]:
    """Merge two ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    index: dict[str, dict] = {}

    for rank, row in enumerate(vec_rows):
        rid = str(row.get("id") or row.get("_rowid") or rank)
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k_rrf + rank + 1)
        index[rid] = row

    for rank, row in enumerate(fts_rows):
        rid = str(row.get("id") or row.get("_rowid") or rank)
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k_rrf + rank + 1)
        index[rid] = row

    return [index[rid] for rid in sorted(scores, key=scores.__getitem__, reverse=True)]


def hybrid_search(
    store,
    query: str,
    k: int = 5,
    category_filter: str | None = None,
    rerank: bool = True,
) -> list:
    """Dense + sparse (BM25) hybrid search with RRF merge and cross-encoder reranking.

    Vector search (ANN) and BM25 full-text search are run independently on the raw
    LanceDB table (bypassing LangChain's broken ``query_type="hybrid"`` path, which
    passes a tuple query rejected by LanceDB ≥0.30), then merged via Reciprocal Rank
    Fusion. The cross-encoder reranker provides a final precision pass over candidates.

    Args:
        category_filter: Optional arXiv category code (e.g. "cs.RO") to restrict
                         results to papers whose ``categories`` field contains it.
        rerank: If True (default), rerank deduplicated results with a cross-encoder.
    """
    store_label = getattr(store, "_table_name", getattr(store, "_collection_name", "unknown"))
    with _search_latency.labels(store=store_label).time():
        _ensure_fts_index(store)
        filter_expr = _safe_category_filter(category_filter)

        # Bypass LangChain wrapper — use raw LanceDB table for both search branches.
        tbl = store.get_table()
        embedding = _cached_embeddings.embed_query(query)

        # Vector branch: ANN search with k*3 oversampling.
        vec_q = tbl.search(embedding, query_type="vector").limit(k * 3)
        if filter_expr:
            vec_q = vec_q.where(filter_expr, prefilter=True)
        vec_rows = vec_q.to_list()

        # FTS branch: BM25 search — only if index is confirmed built.
        fts_rows: list[dict] = []
        if id(store) in _FTS_INDEXED:
            try:
                fts_q = tbl.search(query, query_type="fts").limit(k * 3)
                if filter_expr:
                    fts_q = fts_q.where(filter_expr, prefilter=True)
                fts_rows = fts_q.to_list()
            except Exception as e:
                _log.warning("[FTS] search failed, falling back to vector only: %s", e)

        # Merge with RRF; fall back to pure vector order when BM25 unavailable.
        merged_rows = _rrf_merge(vec_rows, fts_rows) if fts_rows else vec_rows
        chunks = [_row_to_doc(row) for row in merged_rows]

        # Deduplicate: keep the first (highest-ranked) chunk per paper URL.
        seen: dict[str, object] = {}
        for chunk in chunks:
            parent = chunk.metadata.get("url", chunk.metadata.get("chunk_id", ""))
            if parent not in seen:
                seen[parent] = chunk
            if len(seen) >= k:
                break

        result = list(seen.values())
        did_rerank = rerank and result
        if did_rerank:
            result = _rerank(query, result)

    _search_results.labels(store=store_label, reranked=str(did_rerank)).observe(len(result))
    return result


# ---------------------------------------------------------------------------
# LLMs — lazy singletons, provider packages only import on first use.
# ---------------------------------------------------------------------------
def _check_ollama() -> bool:
    """Confirm Ollama is running by hitting its /api/tags endpoint and checking HTTP 200."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _make_fast_llm():
    if _check_ollama():
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        _log.info("[LLM] Summarizer → Ollama (%s)", model)
        return ChatOllama(model=model, temperature=0)
    from langchain_openai import ChatOpenAI

    _log.info("[LLM] Summarizer → OpenAI gpt-4o-mini (Ollama unavailable)")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _make_agent_llm():
    choice = os.getenv("AGENT_LLM", "openai").lower()
    if choice == "claude":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        _log.info("[LLM] Agent → Anthropic (%s) with prompt caching", model)
        return ChatAnthropic(  # type: ignore[call-arg]
            model=model,
            temperature=0.3,
            model_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
        )
    from langchain_openai import ChatOpenAI

    if choice == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8080/v1")
        model = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        _log.info("[LLM] Agent → vLLM (%s) at %s", model, base_url)
        return ChatOpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM requires a non-empty but arbitrary value
            model=model,
            temperature=0.3,
        )
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    _log.info("[LLM] Agent → OpenAI (%s)", model)
    return ChatOpenAI(model=model, temperature=0.3)


llm_fast = _LazyProxy(_make_fast_llm)
llm_agent = _LazyProxy(_make_agent_llm)
