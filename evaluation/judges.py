"""
Eval judge factory — selects and constructs the LLM judge for DeepEval metrics.

EVAL_JUDGE env var (set in .env):
  "prometheus"  → Prometheus 2 via Ollama (free, local, good for iteration)
  "claude"      → claude-opus-4-6 via Anthropic (cross-provider, use for final eval)
  "openai"      → EVAL_JUDGE_MODEL env var string, default gpt-4o (fallback)

If EVAL_JUDGE is unset:
  - Uses "claude" if ANTHROPIC_API_KEY is set
  - Falls back to "openai" otherwise
"""

import os

from deepeval.models.base_model import DeepEvalBaseLLM


class PrometheusOllamaJudge(DeepEvalBaseLLM):
    """Prometheus 2 judge via local Ollama — wraps DeepEvalBaseLLM.

    Prometheus 2 is fine-tuned for reference-based evaluation (faithfulness,
    hallucination, answer relevancy). Running it locally via Ollama is free
    and avoids same-provider bias when the answering model is also local.

    Pull the model once before use:
        ollama pull vicgalle/prometheus-7b-v2.0   # ~4.4 GB
    """

    def __init__(self, model: str = "vicgalle/prometheus-7b-v2.0"):
        self.model = model
        self._client = None

    def get_model_name(self) -> str:
        return f"prometheus-ollama/{self.model}"

    def load_model(self):
        from langchain_ollama import ChatOllama

        self._client = ChatOllama(model=self.model, temperature=0)
        return self._client

    def generate(self, prompt: str) -> str:
        if self._client is None:
            self.load_model()
        response = self._client.invoke(prompt)
        return str(response.content).strip()

    async def a_generate(self, prompt: str) -> str:
        # async_mode=False is used throughout — sync fallback is sufficient
        return self.generate(prompt)


def make_judge():
    """Return the appropriate judge for DeepEval metrics based on EVAL_JUDGE env var.

    Auto-selects if EVAL_JUDGE is not set:
      - "claude" if ANTHROPIC_API_KEY is present
      - "openai" otherwise
    """
    choice = os.getenv("EVAL_JUDGE", "").lower()

    if not choice:
        choice = "claude" if os.getenv("ANTHROPIC_API_KEY") else "openai"

    if choice == "prometheus":
        model = os.getenv("PROMETHEUS_MODEL", "vicgalle/prometheus-7b-v2.0")
        print(f"[Judge] Prometheus 2 via Ollama ({model})")
        return PrometheusOllamaJudge(model=model)

    if choice == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise OSError("EVAL_JUDGE=claude requires ANTHROPIC_API_KEY in .env")
        from deepeval.models import AnthropicModel

        model = os.getenv("EVAL_JUDGE_MODEL_CLAUDE", "claude-opus-4-6")
        print(f"[Judge] {model} via Anthropic (cross-provider)")
        return AnthropicModel(
            model=model,
            cost_per_input_token=0.000015,
            cost_per_output_token=0.000075,
        )

    # Default: OpenAI string — DeepEval creates ChatOpenAI internally
    model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
    print(f"[Judge] {model} via OpenAI")
    return model
