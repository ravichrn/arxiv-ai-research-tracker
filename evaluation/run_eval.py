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
from evaluation.datasets import RAG_CASES
from ingestion.arxiv_fetcher import summarize_text


def _score(metric, test_case: LLMTestCase) -> tuple[float, bool]:
    metric.measure(test_case)
    return metric.score, metric.success


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
    hallucination_metric = HallucinationMetric(threshold=0.5)

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
        print(f"  [{status}] score={score:.2f}  {title[:60]}")

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

    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = AnswerRelevancyMetric(threshold=0.7)

    for case in RAG_CASES:
        retrieved = hybrid_search(papers_store, case.query, k=5)
        if not retrieved:
            print(f"  [SKIP] No results for: {case.query}")
            continue

        context = [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in retrieved]
        context_block = "\n\n".join(context)
        response = llm.invoke(
            f"Using only the following research paper abstracts, answer the question.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {case.query}"
        )
        answer = str(response.content).strip()

        tc = LLMTestCase(input=case.query, actual_output=answer, retrieval_context=context)

        f_score, f_pass = _score(faithfulness, tc)
        r_score, r_pass = _score(relevancy, tc)

        print(f"\n  Query : {case.query}")
        print(f"  Faithfulness : {'PASS' if f_pass else 'FAIL'}  score={f_score:.2f}")
        print(f"  Relevancy    : {'PASS' if r_pass else 'FAIL'}  score={r_score:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=int, default=3, help="Papers to sample for summarizer eval"
    )
    args = parser.parse_args()

    run_summarizer_eval(args.samples)
    run_rag_eval()
    print(f"\n{'='*60}\nEval complete.\n")


if __name__ == "__main__":
    main()
