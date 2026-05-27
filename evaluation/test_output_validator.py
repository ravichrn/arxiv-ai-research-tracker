"""
Tests for the output validator (Option 1 — ArxivCitationValidator + ToxicLanguageValidator).

All tests run offline — ToxicLanguageValidator falls back to PassResult when
the model is unavailable, so these tests cover the structural citation logic
without requiring a GPU or internet connection.
"""

import pytest

from guardrails.output_validator import (
    ArxivCitationValidator,
    FailResult,
    OutputRejected,
    PassResult,
    validate_output,
)

# ---------------------------------------------------------------------------
# ArxivCitationValidator
# ---------------------------------------------------------------------------


class TestArxivCitationValidator:
    def setup_method(self):
        self.validator = ArxivCitationValidator()
        self.known = {"2312.01234", "2401.56789"}

    def test_passes_response_with_known_ids(self):
        response = "As shown in 2312.01234, the approach improves by 5%."
        result = self.validator.validate(response, {"known_arxiv_ids": self.known})
        assert isinstance(result, PassResult)

    def test_flags_hallucinated_id(self):
        response = "According to 9999.99999, this is state of the art."
        result = self.validator.validate(response, {"known_arxiv_ids": self.known})
        assert isinstance(result, FailResult)
        assert "9999.99999" in result.error_message

    def test_passes_when_no_ids_mentioned(self):
        response = "There are no papers directly addressing this question."
        result = self.validator.validate(response, {"known_arxiv_ids": self.known})
        assert isinstance(result, PassResult)

    def test_passes_with_empty_known_ids(self):
        # No DB context provided — citation check is skipped.
        response = "See 9999.99999 for details."
        result = self.validator.validate(response, {"known_arxiv_ids": set()})
        assert isinstance(result, PassResult)

    def test_handles_versioned_id(self):
        # 2312.01234v2 should match base ID 2312.01234
        response = "See arxiv:2312.01234v2 for the full proof."
        result = self.validator.validate(response, {"known_arxiv_ids": self.known})
        assert isinstance(result, PassResult)

    def test_flags_multiple_hallucinated_ids(self):
        response = "Papers 9999.11111 and 8888.22222 both discuss this."
        result = self.validator.validate(response, {"known_arxiv_ids": self.known})
        assert isinstance(result, FailResult)
        assert "9999.11111" in result.error_message
        assert "8888.22222" in result.error_message

    def test_passes_arxiv_prefixed_known_id(self):
        response = "Described in arXiv:2401.56789."
        result = self.validator.validate(response, {"known_arxiv_ids": self.known})
        assert isinstance(result, PassResult)


# ---------------------------------------------------------------------------
# validate_output (full guard — toxicity falls back to Pass when model absent)
# ---------------------------------------------------------------------------


class TestValidateOutput:
    def test_passes_clean_response_with_known_ids(self):
        response = "The paper 2312.01234 introduces a novel attention mechanism."
        result = validate_output(response, {"known_arxiv_ids": {"2312.01234"}})
        assert result == response

    def test_raises_on_hallucinated_id(self):
        response = "According to 9999.99999, transformers are obsolete."
        with pytest.raises(OutputRejected, match=r"9999\.99999"):
            validate_output(response, {"known_arxiv_ids": {"2312.01234"}})

    def test_passes_with_no_metadata(self):
        response = "No specific papers were retrieved for this query."
        result = validate_output(response)
        assert result == response
