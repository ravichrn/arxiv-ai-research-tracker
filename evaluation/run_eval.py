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


def _score(metric, test_case: LLMTestCase) -> tuple[float, bool]:
    metric.measure(test_case)
    return metric.score, metric.success


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
    print(f"\n{'='*60}")
    print(f"SUMMARIZER EVAL  (n={n_samples} random DB papers)")
    print(f"{'='*60}")

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
    for row in sample:
        abstract = row.get("text", "")
        title = row.get("title", "unknown")
        summary = summarize_text(abstract)

        test_case = LLMTestCase(
            input=abstract,
            actual_output=summary,
            context=[abstract],
        )
        score, passed = _score(hallucination_metric, test_case)
        scores.append(score)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] score={score:.2f}  {title[:60]}", flush=True)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n  Average hallucination score: {avg:.2f}  (lower = less hallucination)")
    return {"hallucination_mean": avg, "n": len(scores)}


def run_rag_eval() -> dict | None:
    print(f"\n{'='*60}")
    print(f"RAG EVAL  ({len(RAG_CASES)} test queries)")
    print(f"{'='*60}")

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

    for case in RAG_CASES:
        retrieved = _retrieve(case.query, category=case.category)
        if not retrieved:
            print(f"  [SKIP] No results for: {case.query}")
            skipped += 1
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        answer = _answer(case.query, context)

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, f_pass = _score(faithfulness, tc)
        r_score, r_pass = _score(relevancy, tc)
        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        print(f"\n  Query : {case.query}", flush=True)
        print(f"  Faithfulness : {'PASS' if f_pass else 'FAIL'}  score={f_score:.2f}", flush=True)
        print(f"  Relevancy    : {'PASS' if r_pass else 'FAIL'}  score={r_score:.2f}", flush=True)

    f_mean = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    r_mean = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
    return {
        "faithfulness_mean": f_mean,
        "relevancy_mean": r_mean,
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
    print(f"\n{'='*60}")
    print(f"ADVERSARIAL RAG EVAL  ({len(ADVERSARIAL_RAG_CASES)} cases)")
    print("  Purpose: context is off-topic — LLM must stay grounded, not hallucinate.")
    print(f"{'='*60}")

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

    for case in ADVERSARIAL_RAG_CASES:
        # Force category-scoped retrieval — if category has no papers, fall back to unfiltered.
        # Either way, context is intentionally mismatched to the query.
        retrieved = _retrieve(case.query, category=case.category)
        if not retrieved:
            print(f"  [SKIP] No results for: {case.query}")
            skipped += 1
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        answer = _answer(case.query, context)

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, f_pass = _score(faithfulness, tc)
        r_score, r_pass = _score(relevancy, tc)
        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        print(f"\n  Query : {case.query}", flush=True)
        print("  [adversarial — expect low relevancy, high faithfulness]", flush=True)
        print(f"  Faithfulness : {'PASS' if f_pass else 'FAIL'}  score={f_score:.2f}", flush=True)
        status_r = "PASS" if r_pass else "FAIL"
        print(f"  Relevancy    : {status_r}  score={r_score:.2f}  (low expected)", flush=True)

    f_mean = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    r_mean = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
    return {
        "faithfulness_mean": f_mean,
        "relevancy_mean": r_mean,
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
    print(f"\n{'='*60}")
    print("NO-CONTEXT BASELINE  (first 5 RAG queries, no retrieval)")
    print("  Purpose: quantify what retrieval adds over raw LLM knowledge.")
    print(f"{'='*60}")

    relevancy = AnswerRelevancyMetric(threshold=0.7, model=_JUDGE, async_mode=False)

    baseline_cases = RAG_CASES[:5]
    relevancy_scores: list[float] = []
    for case in baseline_cases:
        response = llm.invoke(case.query)
        answer = str(response.content).strip()

        tc = LLMTestCase(
            input=case.query,
            actual_output=answer,
            retrieval_context=[],  # no context — pure LLM
        )

        r_score, r_pass = _score(relevancy, tc)
        relevancy_scores.append(r_score)

        print(f"\n  Query : {case.query}", flush=True)
        status_r = "PASS" if r_pass else "FAIL"
        print(f"  Relevancy (no context) : {status_r}  score={r_score:.2f}", flush=True)

    r_mean = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
    return {
        "relevancy_mean": r_mean,
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
    payload: dict = {"judge": judge_label, "answer_model": answer_label, "suites": {}}
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
                f"RAG faithfulness (mean): {rag['faithfulness_mean']:.3f}  "
                f"(n_scored={rag['n_scored']}, skipped={rag['n_skipped']})"
            )
            lines.append(f"RAG answer relevancy (mean): {rag['relevancy_mean']:.3f}")
        else:
            lines.append("RAG: no scored cases (DB empty or all skipped).")
    if adversarial:
        payload["suites"]["adversarial"] = adversarial
        if adversarial["faithfulness_mean"] is not None:
            lines.append(
                f"Adversarial faithfulness (mean): {adversarial['faithfulness_mean']:.3f}  "
                f"(n_scored={adversarial['n_scored']}, skipped={adversarial['n_skipped']})"
            )
            arm = adversarial["relevancy_mean"]
            lines.append(f"Adversarial answer relevancy (mean): {arm:.3f}")
    if baseline:
        payload["suites"]["baseline_no_context"] = baseline
        if baseline["relevancy_mean"] is not None:
            brm = baseline["relevancy_mean"]
            lines.append(f"No-context relevancy (mean, first 5 RAG queries): {brm:.3f}")

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
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote metrics JSON to {out}", flush=True)

    print(f"\n{'='*60}\nEval complete.\n")


if __name__ == "__main__":
    main()
