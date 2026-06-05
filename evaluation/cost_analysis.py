"""
Cost vs. quality analysis: measures the token overhead of the full Self-RAG
pipeline (agent → tools → grade_docs → hallucination_check → rewrite loop)
against single-pass RAG (one retrieval call + one LLM answer call).

The framing: a senior engineer honestly reports "the Self-RAG loop costs Nx
tokens for an X% quality gain" rather than asserting the architecture is
strictly better.

Usage:
    uv run python -m evaluation.cost_analysis
    uv run python -m evaluation.cost_analysis --samples 5
    uv run python -m evaluation.cost_analysis --samples 5 \
        --write-results evaluation/cost_vs_quality.json

"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage

from agents.runner import _SYSTEM_MESSAGE, rag_graph, simple_rag_graph
from databases.stores import hybrid_search, papers_store
from databases.stores import llm_agent as llm
from evaluation.datasets import RAG_CASES


# ---------------------------------------------------------------------------
# Token / call counting callback
# ---------------------------------------------------------------------------
class _CallCounter(BaseCallbackHandler):
    """LangChain callback that counts LLM calls and token usage across a run."""

    def __init__(self):
        self.call_count: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.call_count += 1
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------
@dataclass
class QueryProfile:
    query: str
    variant: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    answer: str = ""


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------
def _initial_rag_state(query: str) -> dict:
    return {
        "messages": [_SYSTEM_MESSAGE, HumanMessage(content=query)],
        "retrieval_context": [],
        "rewrite_count": 0,
        "hallucination_verdict": "",
        "known_arxiv_ids": set(),
    }


def _last_ai_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return str(msg.content)
    return ""


def measure_full_pipeline(query: str) -> QueryProfile:
    """Invoke the full Self-RAG graph and return a token + latency profile."""
    counter = _CallCounter()
    config = {"callbacks": [counter]}
    t0 = time.perf_counter()
    result = rag_graph.invoke(_initial_rag_state(query), config=config)
    latency_ms = (time.perf_counter() - t0) * 1000
    return QueryProfile(
        query=query,
        variant="full_selfrag",
        llm_calls=counter.call_count,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        total_tokens=counter.total_tokens,
        latency_ms=latency_ms,
        answer=_last_ai_answer(result.get("messages", [])),
    )


def measure_simple_pipeline(query: str) -> QueryProfile:
    """Invoke the single-pass RAG graph (no grading/hallucination check) and profile it."""
    counter = _CallCounter()
    config = {"callbacks": [counter]}
    t0 = time.perf_counter()
    result = simple_rag_graph.invoke(_initial_rag_state(query), config=config)
    latency_ms = (time.perf_counter() - t0) * 1000
    return QueryProfile(
        query=query,
        variant="simple_pass",
        llm_calls=counter.call_count,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        total_tokens=counter.total_tokens,
        latency_ms=latency_ms,
        answer=_last_ai_answer(result.get("messages", [])),
    )


def measure_single_llm_pass(query: str) -> QueryProfile:
    """Minimal baseline: one hybrid_search call + one direct LLM call, no graph overhead."""
    counter = _CallCounter()
    config = {"callbacks": [counter]}
    t0 = time.perf_counter()
    docs = hybrid_search(papers_store, query, k=5)
    context_block = "\n\n".join(
        f"Title: {d.metadata.get('title', '')}\n{d.page_content}" for d in docs
    )
    response = llm.invoke(
        f"Using only the following research paper abstracts, answer the question.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}",
        config=config,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    answer = str(response.content).strip()
    return QueryProfile(
        query=query,
        variant="single_llm_call",
        llm_calls=counter.call_count,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        total_tokens=counter.total_tokens,
        latency_ms=latency_ms,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# Aggregate stats helpers
# ---------------------------------------------------------------------------
def _mean(vals: list[float | int]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _profiles_to_stats(profiles: list[QueryProfile]) -> dict:
    return {
        "variant": profiles[0].variant if profiles else "",
        "n": len(profiles),
        "llm_calls_mean": _mean([p.llm_calls for p in profiles]),
        "prompt_tokens_mean": _mean([p.prompt_tokens for p in profiles]),
        "completion_tokens_mean": _mean([p.completion_tokens for p in profiles]),
        "total_tokens_mean": _mean([p.total_tokens for p in profiles]),
        "latency_ms_mean": _mean([p.latency_ms for p in profiles]),
    }


def _overhead_ratio(full_stat: dict, baseline_stat: dict, key: str) -> float | str:
    base = baseline_stat.get(key, 0)
    if base == 0:
        return "n/a"
    return round(full_stat.get(key, 0) / base, 2)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_cost_analysis(n_samples: int = 5) -> dict:
    print(f"\n{'=' * 65}")
    print(f"COST VS. QUALITY ANALYSIS  (n={n_samples} queries)")
    print("  Comparing: full Self-RAG | simple graph | single LLM call")
    print(f"{'=' * 65}")

    try:
        probe = papers_store.get_table().search().limit(1).to_list()
    except Exception:
        probe = []
    if not probe:
        print("  DB is empty — run main.py first.")
        return {}

    cases = RAG_CASES[:n_samples]

    # Disable the LangChain LLM response cache for the duration of this analysis.
    # Without this, full_selfrag warms the cache on the first pass and subsequent
    # variants (simple_graph, single_llm_call) hit cached responses — on_llm_end
    # never fires for cache hits, so _CallCounter sees 0 tokens and sub-200ms latency.
    _saved_cache = get_llm_cache()
    set_llm_cache(None)
    print("  [cache disabled for accurate timing]", flush=True)

    full_profiles: list[QueryProfile] = []
    simple_profiles: list[QueryProfile] = []
    single_profiles: list[QueryProfile] = []

    try:
        for i, case in enumerate(cases, 1):
            print(f"\n  [{i}/{len(cases)}] {case.label}", flush=True)

            fp = measure_full_pipeline(case.query)
            full_profiles.append(fp)
            print(
                f"    full_selfrag     calls={fp.llm_calls}  tokens={fp.total_tokens:,}"
                f"  latency={fp.latency_ms:.0f}ms",
                flush=True,
            )

            sp = measure_simple_pipeline(case.query)
            simple_profiles.append(sp)
            print(
                f"    simple_graph     calls={sp.llm_calls}  tokens={sp.total_tokens:,}"
                f"  latency={sp.latency_ms:.0f}ms",
                flush=True,
            )

            bp = measure_single_llm_pass(case.query)
            single_profiles.append(bp)
            print(
                f"    single_llm_call  calls={bp.llm_calls}  tokens={bp.total_tokens:,}"
                f"  latency={bp.latency_ms:.0f}ms",
                flush=True,
            )
    finally:
        set_llm_cache(_saved_cache)

    full_stats = _profiles_to_stats(full_profiles)
    simple_stats = _profiles_to_stats(simple_profiles)
    single_stats = _profiles_to_stats(single_profiles)

    # Print summary table
    print(f"\n{'=' * 65}")
    print(f"{'Pipeline':<20} {'LLM calls':>10} {'Tokens':>10} {'Latency':>10}")
    print("-" * 55)
    for stats in [full_stats, simple_stats, single_stats]:
        print(
            f"  {stats['variant']:<18} {stats['llm_calls_mean']:>10.1f}"
            f" {stats['total_tokens_mean']:>10,.0f}"
            f" {stats['latency_ms_mean']:>8,.0f}ms"
        )
    print("-" * 55)
    print(
        f"  {'full vs single':18}"
        f" {_overhead_ratio(full_stats, single_stats, 'llm_calls_mean'):>10}"
        f" {_overhead_ratio(full_stats, single_stats, 'total_tokens_mean'):>10}"
        f" {_overhead_ratio(full_stats, single_stats, 'latency_ms_mean'):>9}x"
    )
    print(
        f"  {'simple vs single':18}"
        f" {_overhead_ratio(simple_stats, single_stats, 'llm_calls_mean'):>10}"
        f" {_overhead_ratio(simple_stats, single_stats, 'total_tokens_mean'):>10}"
        f" {_overhead_ratio(simple_stats, single_stats, 'latency_ms_mean'):>9}x"
    )
    print(f"{'=' * 65}")

    return {
        "full_selfrag": full_stats,
        "simple_graph": simple_stats,
        "single_llm_call": single_stats,
        "overhead_full_vs_single": {
            "llm_calls": _overhead_ratio(full_stats, single_stats, "llm_calls_mean"),
            "total_tokens": _overhead_ratio(full_stats, single_stats, "total_tokens_mean"),
            "latency_ms": _overhead_ratio(full_stats, single_stats, "latency_ms_mean"),
        },
        "overhead_simple_vs_single": {
            "llm_calls": _overhead_ratio(simple_stats, single_stats, "llm_calls_mean"),
            "total_tokens": _overhead_ratio(simple_stats, single_stats, "total_tokens_mean"),
            "latency_ms": _overhead_ratio(simple_stats, single_stats, "latency_ms_mean"),
        },
        "n_samples": n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Cost vs. quality analysis for the RAG pipeline.")
    parser.add_argument("--samples", type=int, default=5, help="Number of RAG queries to profile")
    parser.add_argument(
        "--write-results",
        type=str,
        default="evaluation/cost_vs_quality.json",
        metavar="PATH",
        help="Write JSON results to PATH (default: evaluation/cost_vs_quality.json)",
    )
    args = parser.parse_args()

    results = run_cost_analysis(n_samples=args.samples)

    if results:
        out = Path(args.write_results)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults written to {out}", flush=True)


if __name__ == "__main__":
    main()
