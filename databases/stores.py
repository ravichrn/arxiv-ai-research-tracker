import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_DB_DIR = _ROOT / "databases"

# ---------------------------------------------------------------------------
# LLM response cache — SQLite, lightweight (<1ms setup), persists across runs.
# Must be initialised before any LLM is constructed.
# ---------------------------------------------------------------------------
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=str(_DB_DIR / "llm_cache.db")))


# ---------------------------------------------------------------------------
# Disk-backed embedding cache — same text never re-embedded across restarts.
# diskcache is pure Python, ~1ms import, supports TTL.
# ---------------------------------------------------------------------------
from typing import Optional
from langchain_core.embeddings import Embeddings

_EMBEDDING_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


class _CachedEmbeddings(Embeddings):
    _cache: Optional["diskcache.Cache"]  # type: ignore[name-defined]
    _base: Optional[Embeddings]

    def __init__(self):
        self._cache = None
        self._base = None

    def _init(self):
        if self._base is None:
            import diskcache
            from langchain_openai import OpenAIEmbeddings
            self._base = OpenAIEmbeddings()
            self._cache = diskcache.Cache(str(_DB_DIR / "embedding_cache"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._init()
        assert self._cache is not None and self._base is not None
        results = []
        for text in texts:
            key = f"doc:{hash(text)}"
            if key not in self._cache:
                self._cache.set(key, self._base.embed_documents([text])[0], expire=_EMBEDDING_TTL_SECONDS)
            results.append(list(self._cache[key]))
        return results

    def embed_query(self, text: str) -> list[float]:
        self._init()
        assert self._cache is not None and self._base is not None
        key = f"q:{hash(text)}"
        if key not in self._cache:
            self._cache.set(key, self._base.embed_query(text), expire=_EMBEDDING_TTL_SECONDS)
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


def _make_papers_store():
    from langchain_community.vectorstores import LanceDB
    return LanceDB(connection=_get_db(), table_name="papers", embedding=_cached_embeddings)


def _make_saved_store():
    from langchain_community.vectorstores import LanceDB
    return LanceDB(connection=_get_db(), table_name="saved", embedding=_cached_embeddings)


papers_store = _LazyProxy(_make_papers_store)
saved_store  = _LazyProxy(_make_saved_store)


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


llm_fast  = _LazyProxy(_make_fast_llm)
llm_agent = _LazyProxy(_make_agent_llm)
