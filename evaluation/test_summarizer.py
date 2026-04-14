"""
Summarizer evaluation: checks that generated summaries are faithful to the
source abstract and do not introduce hallucinated claims.

Run with:  deepeval test run evaluation/test_summarizer.py
       or: pytest evaluation/test_summarizer.py -v
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase

from evaluation.datasets import SUMMARIZER_CASES
from evaluation.judges import make_judge
from ingestion.arxiv_fetcher import summarize_text

_JUDGE = make_judge()


@pytest.mark.parametrize("case", SUMMARIZER_CASES, ids=[c.label for c in SUMMARIZER_CASES])
def test_summarizer_no_hallucination(case):
    """Summary must not introduce facts absent from the source abstract."""
    summary = summarize_text(case.abstract)

    test_case = LLMTestCase(
        input=case.abstract,
        actual_output=summary,
        context=[case.abstract],
    )

    assert_test(test_case, [HallucinationMetric(threshold=0.5, model=_JUDGE)])


@pytest.mark.parametrize("case", SUMMARIZER_CASES, ids=[c.label for c in SUMMARIZER_CASES])
def test_summarizer_coverage(case):
    """Summary must cover the key information from the source abstract."""
    summary = summarize_text(case.abstract)

    test_case = LLMTestCase(
        input=case.abstract,
        actual_output=summary,
    )

    assert_test(test_case, [SummarizationMetric(threshold=0.5, model=_JUDGE)])
