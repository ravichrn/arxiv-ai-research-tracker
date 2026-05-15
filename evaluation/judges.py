"""
Eval judge factory — selects and constructs the LLM judge for DeepEval metrics.

EVAL_JUDGE env var (set in .env):
  "claude"      → Anthropic (cross-provider when answer model is OpenAI)
  "openai"      → OpenAI
  "prometheus"  → Prometheus 2 via local Ollama (free, no API key)

EVAL_JUDGE_MODEL env var — judge model name, applies to whichever provider is selected:
  claude:     default claude-haiku-4-5-20251001
  openai:     default gpt-4o
  prometheus: default vicgalle/prometheus-7b-v2.0

If EVAL_JUDGE is unset: uses "claude" if ANTHROPIC_API_KEY is set, else "openai".
"""

import os

from deepeval.models.base_model import DeepEvalBaseLLM

_DEFAULTS = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o",
    "prometheus": "vicgalle/prometheus-7b-v2.0",
}


def _judge_choice() -> str:
    choice = os.getenv("EVAL_JUDGE", "").lower()
    if not choice:
        choice = "claude" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    return choice


def _judge_model(choice: str) -> str:
    return os.getenv("EVAL_JUDGE_MODEL", _DEFAULTS.get(choice, "gpt-4o"))


class PrometheusOllamaJudge(DeepEvalBaseLLM):
    """Prometheus 2 via local Ollama. Pull once: ollama pull vicgalle/prometheus-7b-v2.0"""

    def __init__(self, model: str = _DEFAULTS["prometheus"]):
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
        return str(self._client.invoke(prompt).content).strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


def describe_eval_judge() -> str:
    choice = _judge_choice()
    model = _judge_model(choice)
    if choice == "prometheus":
        return f"prometheus-ollama/{model}"
    provider = "anthropic" if choice == "claude" else "openai"
    return f"{provider}/{model}"


def describe_answer_model() -> str:
    choice = os.getenv("AGENT_LLM", "openai").lower()
    if choice == "claude":
        return f"anthropic/{os.getenv('ANTHROPIC_MODEL', 'claude-opus-4-6')}"
    return f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4o')}"


def make_judge():
    choice = _judge_choice()
    model = _judge_model(choice)

    if choice == "prometheus":
        print(f"[Judge] Prometheus 2 via Ollama ({model})")
        return PrometheusOllamaJudge(model=model)

    if choice == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise OSError("EVAL_JUDGE=claude requires ANTHROPIC_API_KEY in .env")
        from deepeval.models import AnthropicModel

        print(f"[Judge] {model} via Anthropic (cross-provider)")
        return AnthropicModel(model=model, max_tokens=4096)

    print(f"[Judge] {model} via OpenAI")
    return model
