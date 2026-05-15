"""
Standalone evaluation runner — samples papers already in the DB and scores
the summarizer + RAG pipeline without needing pytest.

Usage:
    uv run python -m evaluation.run_eval
    uv run python -m evaluation.run_eval --samples 5
"""

import argparse
import json
import random
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

from databases.stores import hybrid_search, papers_store
from databases.stores import llm_agent as llm
from evaluation.datasets import ADVERSARIAL_RAG_CASES, RAG_CASES
from evaluation.judges import describe_answer_model, describe_eval_judge, make_judge
from ingestion.arxiv_fetcher import summarize_text

_JUDGE = make_judge()

if describe_answer_model() == describe_eval_judge():
    print(
        "\n[WARNING] Eval answers and DeepEval judge use the same model "
        f"({describe_eval_judge()}). Faithfulness and relevancy can be optimistically biased. "
        "Prefer EVAL_JUDGE_MODEL different from OPENAI_MODEL, or EVAL_JUDGE=prometheus|claude.\n",
        flush=True,
    )


def _stats(scores: list[float]) -> dict:
    """Return mean, std, min, max for a score list."""
    if not scores:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def _score(metric, test_case: LLMTestCase) -> tuple[float | None, bool]:
    try:
        metric.measure(test_case)
        return metric.score, metric.success
    except Exception as e:
        print(f"  [ERROR] metric failed: {e}", flush=True)
        return None, False


def _retrieve(query: str, k: int = 5, category: str | None = None) -> list:
    """Hybrid search with category fallback: retry unfiltered if scoped search returns nothing."""
    docs = hybrid_search(papers_store, query, k=k, category_filter=category)
    if not docs and category:
        docs = hybrid_search(papers_store, query, k=k)
    return docs


def _answer(query: str, context: list[str]) -> str:
    context_block = "\n\n".join(context)
    response = llm.invoke(
        f"Using only the following research paper abstracts, answer the question.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}"
    )
    return str(response.content).strip()


def run_summarizer_eval(n_samples: int) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"SUMMARIZER EVAL  (n={n_samples} random DB papers)")
    print(f"{'=' * 60}")

    try:
        rows = papers_store.get_table().search().select(["text", "title"]).to_list()
    except Exception:
        rows = []
    if not rows:
        print("  DB is empty — run main.py first.")
        return None

    sample = random.sample(rows, min(n_samples, len(rows)))
    hallucination_metric = HallucinationMetric(threshold=0.5, model=_JUDGE, async_mode=False)

    scores = []
    total = len(sample)
    for i, row in enumerate(sample, 1):
        abstract = row.get("text", "")
        summary = summarize_text(abstract)
        test_case = LLMTestCase(input=abstract, actual_output=summary, context=[abstract])
        score, _ = _score(hallucination_metric, test_case)
        if score is not None:
            scores.append(score)
        s_str = f"{score:.2f}" if score is not None else "ERR"
        print(f"  [{i}/{total}] hallucination score={s_str}", flush=True)

    s = _stats(scores)
    print(
        f"\n  Hallucination: mean={s['mean']:.3f}  std={s['std']:.3f}"
        f"  min={s['min']:.2f}  max={s['max']:.2f}  (lower = less hallucination)"
    )
    return {
        "hallucination_mean": s["mean"],
        "hallucination_std": s["std"],
        "hallucination_min": s["min"],
        "hallucination_max": s["max"],
        "n": len(scores),
    }


def run_rag_eval() -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"RAG EVAL  ({len(RAG_CASES)} test queries)")
    print(f"{'=' * 60}")

    try:
        probe = papers_store.get_table().search().limit(1).to_list()
    except Exception:
        probe = []
    if not probe:
        print("  DB is empty — run main.py first.")
        return None

    faithfulness = FaithfulnessMetric(threshold=0.7, model=_JUDGE, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=_JUDGE, async_mode=False)

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    skipped = 0

    total = len(RAG_CASES)
    for i, case in enumerate(RAG_CASES, 1):
        retrieved = _retrieve(case.query, category=case.category)
        if not retrieved:
            print(f"  [{i}/{total}] SKIP — no results", flush=True)
            skipped += 1
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        answer = _answer(case.query, context)

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, _ = _score(faithfulness, tc)
        r_score, _ = _score(relevancy, tc)
        if f_score is not None:
            faithfulness_scores.append(f_score)
        if r_score is not None:
            relevancy_scores.append(r_score)

        f_str = f"{f_score:.2f}" if f_score is not None else "ERR"
        r_str = f"{r_score:.2f}" if r_score is not None else "ERR"
        print(f"  [{i}/{total}] faithfulness={f_str}  relevancy={r_str}", flush=True)

    f_stats = _stats(faithfulness_scores)
    r_stats = _stats(relevancy_scores)
    return {
        "faithfulness_mean": f_stats["mean"],
        "faithfulness_std": f_stats["std"],
        "faithfulness_min": f_stats["min"],
        "faithfulness_max": f_stats["max"],
        "relevancy_mean": r_stats["mean"],
        "relevancy_std": r_stats["std"],
        "relevancy_min": r_stats["min"],
        "relevancy_max": r_stats["max"],
        "n_scored": len(faithfulness_scores),
        "n_skipped": skipped,
    }


def run_adversarial_eval() -> dict | None:
    """Adversarial cases: queries whose topic is absent from the retrieved category.

    The LLM is given context that does NOT contain the answer. A faithful system
    must say so (grounded refusal) rather than confabulating from prior knowledge.
    Expected outcome: faithfulness stays high (answer is grounded), relevancy is
    intentionally low (the corpus can't answer the question).
    """
    print(f"\n{'=' * 60}")
    print(f"ADVERSARIAL RAG EVAL  ({len(ADVERSARIAL_RAG_CASES)} cases)")
    print("  Purpose: context is off-topic — LLM must stay grounded, not hallucinate.")
    print(f"{'=' * 60}")

    try:
        probe = papers_store.get_table().search().limit(1).to_list()
    except Exception:
        probe = []
    if not probe:
        print("  DB is empty — run main.py first.")
        return None

    faithfulness = FaithfulnessMetric(threshold=0.7, model=_JUDGE, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=_JUDGE, async_mode=False)

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    skipped = 0

    total = len(ADVERSARIAL_RAG_CASES)
    for i, case in enumerate(ADVERSARIAL_RAG_CASES, 1):
        retrieved = _retrieve(case.query, category=case.category)
        if not retrieved:
            print(f"  [{i}/{total}] SKIP — no results", flush=True)
            skipped += 1
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        answer = _answer(case.query, context)

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, _ = _score(faithfulness, tc)
        r_score, _ = _score(relevancy, tc)
        if f_score is not None:
            faithfulness_scores.append(f_score)
        if r_score is not None:
            relevancy_scores.append(r_score)

        f_str = f"{f_score:.2f}" if f_score is not None else "ERR"
        r_str = f"{r_score:.2f}" if r_score is not None else "ERR"
        print(
            f"  [{i}/{total}] faithfulness={f_str}  relevancy={r_str}  (low relevancy expected)",
            flush=True,
        )

    f_stats = _stats(faithfulness_scores)
    r_stats = _stats(relevancy_scores)
    return {
        "faithfulness_mean": f_stats["mean"],
        "faithfulness_std": f_stats["std"],
        "faithfulness_min": f_stats["min"],
        "faithfulness_max": f_stats["max"],
        "relevancy_mean": r_stats["mean"],
        "relevancy_std": r_stats["std"],
        "relevancy_min": r_stats["min"],
        "relevancy_max": r_stats["max"],
        "n_scored": len(faithfulness_scores),
        "n_skipped": skipped,
    }


def run_no_context_baseline() -> dict | None:
    """Baseline: answer the same RAG queries with NO retrieved context.

    Measures raw LLM hallucination without grounding. Comparing these scores to
    the RAG eval scores quantifies the value the retrieval pipeline adds.
    Expected outcome: faithfulness is undefined/low (no context to be faithful to),
    so we measure only relevancy — how well the LLM answers from prior knowledge alone.
    """
    print(f"\n{'=' * 60}")
    print("NO-CONTEXT BASELINE  (first 5 RAG queries, no retrieval)")
    print("  Purpose: quantify what retrieval adds over raw LLM knowledge.")
    print(f"{'=' * 60}")

    relevancy = AnswerRelevancyMetric(threshold=0.7, model=_JUDGE, async_mode=False)

    baseline_cases = RAG_CASES[:5]
    total = len(baseline_cases)
    relevancy_scores: list[float] = []
    for i, case in enumerate(baseline_cases, 1):
        response = llm.invoke(case.query)
        answer = str(response.content).strip()

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=[])
        r_score, _ = _score(relevancy, tc)
        if r_score is not None:
            relevancy_scores.append(r_score)

        r_str = f"{r_score:.2f}" if r_score is not None else "ERR"
        print(f"  [{i}/{total}] relevancy={r_str}  (no context)", flush=True)

    r_stats = _stats(relevancy_scores)
    return {
        "relevancy_mean": r_stats["mean"],
        "relevancy_std": r_stats["std"],
        "relevancy_min": r_stats["min"],
        "relevancy_max": r_stats["max"],
        "n": len(relevancy_scores),
    }


def _print_eval_summary(
    judge_label: str,
    summarizer: dict | None,
    rag: dict | None,
    adversarial: dict | None,
    baseline: dict | None,
) -> dict:
    """Print a copy-paste friendly block and return a JSON-serializable summary dict."""
    answer_label = describe_answer_model()
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"
    payload: dict = {
        "judge": judge_label,
        "answer_model": answer_label,
        "git_sha": git_sha,
        "timestamp": datetime.now(UTC).isoformat(),
        "suites": {},
    }
    lines = [
        "",
        "=" * 60,
        "EVAL_SUMMARY (copy into README or release notes)",
        "=" * 60,
        f"Judge (configured): {judge_label}",
        f"Answer model (RAG / baseline): {answer_label}",
    ]

    if summarizer:
        payload["suites"]["summarizer"] = summarizer
        hm = summarizer["hallucination_mean"]
        n = summarizer["n"]
        lines.append(f"Summarizer hallucination (mean, lower=better): {hm:.3f}  (n={n})")
    if rag:
        payload["suites"]["rag"] = rag
        if rag["faithfulness_mean"] is not None:
            lines.append(
                f"RAG faithfulness:  mean={rag['faithfulness_mean']:.3f}  std={rag['faithfulness_std']:.3f}"  # noqa: E501
                f"  min={rag['faithfulness_min']:.2f}  max={rag['faithfulness_max']:.2f}"
                f"  (n={rag['n_scored']}, skipped={rag['n_skipped']})"
            )
            lines.append(
                f"RAG relevancy:     mean={rag['relevancy_mean']:.3f}  std={rag['relevancy_std']:.3f}"  # noqa: E501
                f"  min={rag['relevancy_min']:.2f}  max={rag['relevancy_max']:.2f}"
            )
        else:
            lines.append("RAG: no scored cases (DB empty or all skipped).")
    if adversarial:
        payload["suites"]["adversarial"] = adversarial
        if adversarial["faithfulness_mean"] is not None:
            lines.append(
                f"Adversarial faithfulness:  mean={adversarial['faithfulness_mean']:.3f}"
                f"  std={adversarial['faithfulness_std']:.3f}"
                f"  (n={adversarial['n_scored']}, skipped={adversarial['n_skipped']})"
            )
            lines.append(
                f"Adversarial relevancy:     mean={adversarial['relevancy_mean']:.3f}"
                f"  std={adversarial['relevancy_std']:.3f}  (low expected)"
            )
    if baseline:
        payload["suites"]["baseline_no_context"] = baseline
        if baseline["relevancy_mean"] is not None:
            lines.append(
                f"No-context relevancy:  mean={baseline['relevancy_mean']:.3f}"
                f"  std={baseline['relevancy_std']:.3f}  (first 5 RAG queries, no retrieval)"
            )

    lines.append("=" * 60)
    print("\n".join(lines), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=int, default=3, help="Papers to sample for summarizer eval"
    )
    parser.add_argument(
        "--suite",
        choices=["all", "rag", "summarizer", "adversarial", "baseline"],
        default="all",
        help="Which eval suite to run (default: all)",
    )
    parser.add_argument(
        "--write-metrics",
        type=str,
        default="",
        metavar="PATH",
        help="Write JSON summary (same structure as EVAL_SUMMARY) to PATH",
    )
    args = parser.parse_args()

    summarizer_stats = rag_stats = adversarial_stats = baseline_stats = None

    if args.suite in ("all", "summarizer"):
        summarizer_stats = run_summarizer_eval(args.samples)
    if args.suite in ("all", "rag"):
        rag_stats = run_rag_eval()
    if args.suite in ("all", "adversarial"):
        adversarial_stats = run_adversarial_eval()
    if args.suite in ("all", "baseline"):
        baseline_stats = run_no_context_baseline()

    summary = _print_eval_summary(
        describe_eval_judge(),
        summarizer_stats,
        rag_stats,
        adversarial_stats,
        baseline_stats,
    )

    if args.write_metrics:
        out = Path(args.write_metrics)
        existing: dict = {}
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                pass
        # Merge: top-level metadata from current run, suites merged per-key
        merged = {**existing, **{k: v for k, v in summary.items() if k != "suites"}}
        merged["suites"] = {**existing.get("suites", {}), **summary["suites"]}
        out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"\nUpdated metrics JSON at {out}", flush=True)

    print(f"\n{'=' * 60}\nEval complete.\n")


if __name__ == "__main__":
    main()
