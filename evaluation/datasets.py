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
    category: str | None = None  # arXiv category to scope retrieval (e.g. "cs.RO")


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
    SummarizerCase(
        label="diffusion_ddpm",
        abstract=(
            "We present high quality image synthesis results using diffusion probabilistic models, a class of latent "
            "variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are "
            "obtained by training on a weighted variational bound designed according to a novel connection between "
            "diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models "
            "naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization "
            "of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 "
            "and a state-of-the-art FID score of 3.17."
        ),
    ),
    SummarizerCase(
        label="gpt3_fewshot",
        abstract=(
            "We demonstrate that scaling language models greatly improves task-agnostic, few-shot performance, "
            "sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. "
            "Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, "
            "and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any "
            "gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via "
            "text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, "
            "including translation, question-answering, and cloze tasks."
        ),
    ),
    SummarizerCase(
        label="clip_contrastive",
        abstract=(
            "State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object "
            "categories. This restricted form of supervision limits their generality and usability since additional "
            "labeled data is needed to specify any other visual concept. Learning directly from raw text about "
            "images is a promising alternative which leverages a much broader source of supervision. We demonstrate "
            "that the simple pre-training task of predicting which caption goes with which image is an efficient "
            "and scalable way to learn SOTA image representations from scratch on a dataset of 400 million "
            "collected from the internet."
        ),
    ),
]

ADVERSARIAL_RAG_CASES: list[RAGCase] = [
    # Query is on-topic but retrieved docs are about a *different* sub-field —
    # the LLM should stay grounded and not hallucinate from prior knowledge.
    RAGCase(
        label="adv_quantum_computing",
        query="What are recent breakthroughs in quantum computing algorithms?",
        expected_keywords=["quantum", "qubit", "algorithm"],
        category="cs.AI",  # scopes to AI papers — no quantum computing content expected
    ),
    RAGCase(
        label="adv_biology_genomics",
        query="How are transformer models applied to protein folding and genomics?",
        expected_keywords=["protein", "genome", "AlphaFold"],
        category="cs.RO",  # scopes to robotics papers — no bioinformatics content expected
    ),
    RAGCase(
        label="adv_wrong_category",
        query="Explain the latest advances in large language model training.",
        expected_keywords=["language model", "training", "transformer"],
        category="cs.RO",  # intentionally wrong category — robotics DB, not NLP
    ),
]

RAG_CASES: list[RAGCase] = [
    # cs.CL — NLP / language models
    RAGCase(
        label="llm_scaling",
        query="What are the latest papers on large language models and scaling?",
        expected_keywords=["language model", "LLM", "transformer", "GPT", "scaling", "training"],
        category="cs.CL",
    ),
    RAGCase(
        label="instruction_tuning",
        query="What methods are used for instruction tuning and alignment of language models?",
        expected_keywords=["instruction", "alignment", "fine-tuning", "RLHF", "human feedback"],
        category="cs.CL",
    ),
    RAGCase(
        label="rag_retrieval",
        query="How do retrieval-augmented generation systems work for question answering?",
        expected_keywords=["retrieval", "augmented", "generation", "RAG", "question answering"],
        category="cs.CL",
    ),
    # cs.LG — Machine learning
    RAGCase(
        label="diffusion_models",
        query="What are recent advances in diffusion models for generative tasks?",
        expected_keywords=["diffusion", "generative", "score matching", "denoising", "image"],
        category="cs.LG",
    ),
    RAGCase(
        label="rl_policy_optimization",
        query="What papers focus on reinforcement learning algorithms, reward modeling, or policy optimization?",
        expected_keywords=["reward", "policy", "agent", "environment", "reinforcement"],
        category="cs.LG",
    ),
    RAGCase(
        label="graph_neural_networks",
        query="What recent work exists on graph neural networks and their applications?",
        expected_keywords=["graph", "neural network", "node", "edge", "GNN"],
        category="cs.LG",
    ),
    # cs.AI — AI systems
    RAGCase(
        label="multimodal_learning",
        query="What papers cover multimodal learning combining vision and language?",
        expected_keywords=["multimodal", "vision", "language", "image", "contrastive"],
        category="cs.AI",
    ),
    RAGCase(
        label="reasoning_planning",
        query="What work exists on reasoning and planning in AI agents?",
        expected_keywords=["reasoning", "planning", "agent", "chain-of-thought", "inference"],
        category="cs.AI",
    ),
    # cs.RO — Robotics
    RAGCase(
        label="robotics_manipulation",
        query="What recent work exists on robot learning and manipulation?",
        expected_keywords=["robot", "manipulation", "reinforcement learning", "policy", "control"],
        category="cs.RO",
    ),
    RAGCase(
        label="robot_perception",
        query="How do robots perceive and understand their environment for navigation?",
        expected_keywords=["robot", "perception", "navigation", "sensor", "mapping"],
        category="cs.RO",
    ),
]
