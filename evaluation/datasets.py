"""
Hardcoded test cases for offline evaluation.
Abstracts are from well-known, publicly available papers so facts are verifiable.
"""

from dataclasses import dataclass, field


@dataclass
class SummarizerCase:
    abstract: str
    label: str = ""


@dataclass
class RAGCase:
    query: str
    expected_keywords: list[str] = field(default_factory=list)
    label: str = ""


SUMMARIZER_CASES: list[SummarizerCase] = [
    SummarizerCase(
        label="transformer_attention",
        abstract=(
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show "
            "these models to be superior in quality while being more parallelizable and requiring significantly less "
            "time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, "
            "improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 "
            "English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU "
            "score of 41.0 after training for 3.5 days on eight GPUs."
        ),
    ),
    SummarizerCase(
        label="bert_pretraining",
        abstract=(
            "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder "
            "Representations from Transformers. Unlike recent language representation models, BERT is designed to "
            "pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left "
            "and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just "
            "one additional output layer to create state-of-the-art models for a wide range of tasks, such as "
            "question answering and language inference, without substantial task-specific architecture modifications."
        ),
    ),
    SummarizerCase(
        label="rlhf_instructgpt",
        abstract=(
            "Making language models bigger does not inherently make them better at following a user's intent. "
            "Large models can generate outputs that are untruthful, toxic, or simply not helpful to the user. "
            "In other words, these models are not aligned with their users. In this paper, we show an avenue for "
            "aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. "
            "Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect "
            "a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using "
            "supervised learning. We then collect a dataset of rankings of model outputs, which we use to further "
            "fine-tune this supervised model using reinforcement learning from human feedback (RLHF)."
        ),
    ),
]

RAG_CASES: list[RAGCase] = [
    RAGCase(
        label="llm_query",
        query="What are the latest papers on large language models?",
        expected_keywords=["language model", "LLM", "transformer", "GPT", "BERT", "training"],
    ),
    RAGCase(
        label="robotics_query",
        query="What recent work exists on robot learning and manipulation?",
        expected_keywords=["robot", "manipulation", "reinforcement learning", "policy", "control"],
    ),
    RAGCase(
        label="rl_query",
        query="What papers focus specifically on reinforcement learning algorithms, reward modeling, or policy optimization?",
        expected_keywords=["reward", "policy", "agent", "environment", "reinforcement"],
    ),
]
