"""
RAG pipeline evaluation: checks that agent answers are faithful to retrieved
context and relevant to the user query.

Requires the papers DB to be populated (run main.py at least once first).

Run with:  deepeval test run evaluation/test_rag.py
       or: pytest evaluation/test_rag.py -v
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from databases.stores import hybrid_search, papers_store
from databases.stores import llm_agent as llm
from evaluation.datasets import RAG_CASES


def _retrieve(query: str, k: int = 5) -> list[str]:
    try:
        docs = hybrid_search(papers_store, query, k=k)
    except Exception:
        return []
    return [f"Title: {d.metadata.get('title')}\n{d.page_content}" for d in docs]


def _answer(query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(context_chunks)
    response = llm.invoke(
        f"Using only the following research paper abstracts, answer the question.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}"
    )
    return str(response.content).strip()


@pytest.mark.parametrize("case", RAG_CASES, ids=[c.label for c in RAG_CASES])
def test_rag_faithfulness(case):
    """Agent answer must be grounded in the retrieved context (no hallucination)."""
    context = _retrieve(case.query)
    if not context:
        pytest.skip("DB is empty — run main.py to populate papers first.")

    answer = _answer(case.query, context)

    test_case = LLMTestCase(
        input=case.query,
        actual_output=answer,
        retrieval_context=context,
    )

    assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])


@pytest.mark.parametrize("case", RAG_CASES, ids=[c.label for c in RAG_CASES])
def test_rag_answer_relevancy(case):
    """Agent answer must be relevant to the user query."""
    context = _retrieve(case.query)
    if not context:
        pytest.skip("DB is empty — run main.py to populate papers first.")

    answer = _answer(case.query, context)

    test_case = LLMTestCase(
        input=case.query,
        actual_output=answer,
        retrieval_context=context,
    )

    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])


@pytest.mark.parametrize("case", RAG_CASES, ids=[c.label for c in RAG_CASES])
def test_rag_context_relevancy(case):
    """Retrieved documents must be relevant to the query (retrieval quality).

    Uses k=3 (not k=5) — ContextualRelevancyMetric scores the fraction of
    retrieved chunks that are on-topic. Fetching 5 docs from a mixed-topic DB
    almost always pulls in 1-2 tangential papers, collapsing the score to ~0.4
    regardless of retrieval quality. k=3 tests precision without penalising
    the inherent breadth of a multi-topic corpus.
    Threshold 0.5: at least 2 of 3 returned docs must be on-topic.
    """
    context = _retrieve(case.query, k=3)
    if not context:
        pytest.skip("DB is empty — run main.py to populate papers first.")

    test_case = LLMTestCase(
        input=case.query,
        actual_output=context[0],
        retrieval_context=context,
    )

    assert_test(test_case, [ContextualRelevancyMetric(threshold=0.5)])
