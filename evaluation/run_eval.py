"""
Standalone evaluation runner — samples papers already in the DB and scores
the summarizer + RAG pipeline without needing pytest.

Usage:
    uv run python -m evaluation.run_eval
    uv run python -m evaluation.run_eval --samples 5
"""

import argparse
import random

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

from databases.stores import hybrid_search, papers_store
from databases.stores import llm_agent as llm
from evaluation.datasets import ADVERSARIAL_RAG_CASES, RAG_CASES
from evaluation.judges import make_judge
from ingestion.arxiv_fetcher import summarize_text

_JUDGE = make_judge()


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


def run_summarizer_eval(n_samples: int) -> None:
    print(f"\n{'='*60}")
    print(f"SUMMARIZER EVAL  (n={n_samples} random DB papers)")
    print(f"{'='*60}")

    try:
        rows = papers_store.get_table().search().select(["text", "title"]).to_list()
    except Exception:
        rows = []
    if not rows:
        print("  DB is empty — run main.py first.")
        return

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


def run_rag_eval() -> None:
    print(f"\n{'='*60}")
    print(f"RAG EVAL  ({len(RAG_CASES)} test queries)")
    print(f"{'='*60}")

    try:
        probe = papers_store.get_table().search().limit(1).to_list()
    except Exception:
        probe = []
    if not probe:
        print("  DB is empty — run main.py first.")
        return

    faithfulness = FaithfulnessMetric(threshold=0.7, model=_JUDGE, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=_JUDGE, async_mode=False)

    for case in RAG_CASES:
        retrieved = _retrieve(case.query, category=case.category)
        if not retrieved:
            print(f"  [SKIP] No results for: {case.query}")
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        answer = _answer(case.query, context)

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, f_pass = _score(faithfulness, tc)
        r_score, r_pass = _score(relevancy, tc)

        print(f"\n  Query : {case.query}", flush=True)
        print(f"  Faithfulness : {'PASS' if f_pass else 'FAIL'}  score={f_score:.2f}", flush=True)
        print(f"  Relevancy    : {'PASS' if r_pass else 'FAIL'}  score={r_score:.2f}", flush=True)


def run_adversarial_eval() -> None:
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
        return

    faithfulness = FaithfulnessMetric(threshold=0.7, model=_JUDGE, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=_JUDGE, async_mode=False)

    for case in ADVERSARIAL_RAG_CASES:
        # Force category-scoped retrieval — if category has no papers, fall back to unfiltered.
        # Either way, context is intentionally mismatched to the query.
        retrieved = _retrieve(case.query, category=case.category)
        if not retrieved:
            print(f"  [SKIP] No results for: {case.query}")
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        answer = _answer(case.query, context)

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, f_pass = _score(faithfulness, tc)
        r_score, r_pass = _score(relevancy, tc)

        print(f"\n  Query : {case.query}", flush=True)
        print("  [adversarial — expect low relevancy, high faithfulness]", flush=True)
        print(f"  Faithfulness : {'PASS' if f_pass else 'FAIL'}  score={f_score:.2f}", flush=True)
        status_r = "PASS" if r_pass else "FAIL"
        print(f"  Relevancy    : {status_r}  score={r_score:.2f}  (low expected)", flush=True)


def run_no_context_baseline() -> None:
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
    for case in baseline_cases:
        response = llm.invoke(case.query)
        answer = str(response.content).strip()

        tc = LLMTestCase(
            input=case.query,
            actual_output=answer,
            retrieval_context=[],  # no context — pure LLM
        )

        r_score, r_pass = _score(relevancy, tc)

        print(f"\n  Query : {case.query}", flush=True)
        status_r = "PASS" if r_pass else "FAIL"
        print(f"  Relevancy (no context) : {status_r}  score={r_score:.2f}", flush=True)


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
    args = parser.parse_args()

    if args.suite in ("all", "summarizer"):
        run_summarizer_eval(args.samples)
    if args.suite in ("all", "rag"):
        run_rag_eval()
    if args.suite in ("all", "adversarial"):
        run_adversarial_eval()
    if args.suite in ("all", "baseline"):
        run_no_context_baseline()

    print(f"\n{'='*60}\nEval complete.\n")


if __name__ == "__main__":
    main()
