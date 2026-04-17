import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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

from langchain_core.embeddings import Embeddings

_EMBEDDING_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


class _CachedEmbeddings(Embeddings):
    _cache: "diskcache.Cache | None"  # type: ignore[name-defined]
    _base: Embeddings | None

    def __init__(self):
        self._cache = None
        self._base = None

    def _init(self):
        if self._base is None:
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
    def _embed_one_doc(self, text: str) -> list[float]:
        assert self._base is not None
        return self._base.embed_documents([text])[0]

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
            else:
                missing_texts_by_key[key] = text
                missing_indices_by_key.setdefault(key, []).append(i)

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
            self._cache.set(key, self._embed_one_query(text), expire=_EMBEDDING_TTL_SECONDS)
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

    def _get(self):
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


def _get_db():
    global _db_instance
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
        print(f"[FTS] index creation skipped: {e}")


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
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)
    return _reranker


def _rerank(query: str, docs: list) -> list:
    """Rerank docs by cross-encoder (query, doc) score descending.

    Falls back to original order if the model fails to load or score.
    """
    if not docs:
        return docs
    try:
        reranker = _get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs, strict=True), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]
    except Exception as e:
        print(f"[Reranker] skipped: {e}")
        return docs


def hybrid_search(
    store,
    query: str,
    k: int = 5,
    category_filter: str | None = None,
    rerank: bool = True,
) -> list:
    """Hybrid vector + full-text search with chunk deduplication and optional reranking.

    Retrieves up to k*3 candidate chunks via hybrid search (dense + BM25),
    deduplicates to at most *k* unique papers (one best chunk each), then
    reranks with a cross-encoder for higher precision.

    Falls back to pure vector search if the FTS index is not yet built.

    Args:
        category_filter: Optional arXiv category code (e.g. "cs.RO") to restrict
                         results to papers whose ``categories`` field contains it.
        rerank: If True (default), rerank deduplicated results with a cross-encoder.
    """
    _ensure_fts_index(store)
    # LangChain's LanceDB integration stores metadata as a Struct column with named
    # sub-fields — access via dot notation, not as a flat column or JSON string.
    filter_expr = f"metadata.categories LIKE '%{category_filter}%'" if category_filter else None
    try:
        chunks = store.similarity_search(query, k=k * 3, query_type="hybrid", filter=filter_expr)
    except Exception:
        # FTS index missing or incompatible version — degrade gracefully.
        chunks = store.similarity_search(query, k=k, filter=filter_expr)

    # Deduplicate: keep the first (highest-ranked) chunk per paper URL.
    seen: dict[str, object] = {}
    for chunk in chunks:
        parent = chunk.metadata.get("url", chunk.metadata.get("chunk_id", ""))
        if parent not in seen:
            seen[parent] = chunk
        if len(seen) >= k:
            break

    result = list(seen.values())
    if rerank and result:
        result = _rerank(query, result)
    return result


# ---------------------------------------------------------------------------
# LLMs — lazy singletons, provider packages only import on first use.
# ---------------------------------------------------------------------------
def _check_ollama() -> bool:
    """Lightweight HTTP check — does not load the model into memory."""
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False


def _make_fast_llm():
    if _check_ollama():
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        print(f"[LLM] Summarizer → Ollama ({model})")
        return ChatOllama(model=model, temperature=0)
    from langchain_openai import ChatOpenAI

    print("[LLM] Summarizer → OpenAI gpt-4o-mini (Ollama unavailable)")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _make_agent_llm():
    choice = os.getenv("AGENT_LLM", "openai").lower()
    if choice == "claude":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        print(f"[LLM] Agent → Anthropic ({model}) with prompt caching")
        return ChatAnthropic(  # type: ignore[call-arg]
            model=model,
            temperature=0.3,
            model_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
        )
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    print(f"[LLM] Agent → OpenAI ({model})")
    return ChatOpenAI(model=model, temperature=0.3)


llm_fast = _LazyProxy(_make_fast_llm)
llm_agent = _LazyProxy(_make_agent_llm)
