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
    # -----------------------------------------------------------------------
    # Axis 1: Completely non-CS domains — clear domain mismatch.
    # The LLM has strong parametric knowledge here but must stay grounded.
    # -----------------------------------------------------------------------
    RAGCase(
        label="adv_quantum_computing",
        query="What are recent breakthroughs in quantum computing algorithms?",
        expected_keywords=["quantum", "qubit", "algorithm"],
        category="cs.AI",  # AI papers — no quantum computing content expected
    ),
    RAGCase(
        label="adv_biology_genomics",
        query="How are transformer models applied to protein folding and genomics?",
        expected_keywords=["protein", "genome", "AlphaFold"],
        category="cs.RO",  # robotics papers — no bioinformatics content expected
    ),
    RAGCase(
        label="adv_clinical_medicine",
        query="What are the latest FDA-approved drugs for treating Alzheimer's disease?",
        expected_keywords=["drug", "clinical", "FDA", "Alzheimer", "treatment"],
        category="cs.AI",  # AI papers — no clinical medicine content expected
    ),
    RAGCase(
        label="adv_economics",
        query="How does monetary policy affect inflation and interest rates?",
        expected_keywords=["inflation", "interest rate", "monetary", "central bank", "policy"],
        category="cs.LG",  # ML papers — no economics content expected
    ),
    RAGCase(
        label="adv_climate_science",
        query="What are the primary drivers of Arctic ice sheet melting and how are they measured?",
        expected_keywords=["Arctic", "ice sheet", "glacier", "melting", "permafrost"],
        category="cs.CL",  # NLP papers — no climate science content expected
    ),
    RAGCase(
        label="adv_organic_chemistry",
        query="What catalysts are used in asymmetric synthesis of chiral pharmaceutical compounds?",
        expected_keywords=["catalyst", "chiral", "synthesis", "enantiomer", "pharmaceutical"],
        category="cs.RO",  # robotics papers — no organic chemistry content expected
    ),
    RAGCase(
        label="adv_cosmology",
        query="What is the current evidence for dark matter from galaxy rotation curves?",
        expected_keywords=["dark matter", "galaxy", "rotation curve", "WIMP", "cosmology"],
        category="cs.LG",  # ML papers — no astrophysics content expected
    ),
    RAGCase(
        label="adv_public_health",
        query="How did mRNA vaccine technology achieve high efficacy against COVID-19 variants?",
        expected_keywords=["mRNA", "vaccine", "efficacy", "COVID", "immunogenicity"],
        category="cs.AI",  # AI papers — no virology content expected
    ),
    RAGCase(
        label="adv_constitutional_law",
        query="What is the precedent established by Marbury v. Madison for judicial review?",
        expected_keywords=[
            "judicial review",
            "Supreme Court",
            "constitution",
            "precedent",
            "Marbury",
        ],
        category="cs.RO",  # robotics papers — no legal content expected
    ),
    RAGCase(
        label="adv_geotectonics",
        query="How do subduction zones cause megathrust earthquakes and tsunamis?",
        expected_keywords=["subduction", "earthquake", "tsunami", "tectonic", "fault"],
        category="cs.CL",  # NLP papers — no geology content expected
    ),
    RAGCase(
        label="adv_nutrition_science",
        query="What is the role of gut microbiome diversity in metabolic syndrome and obesity?",
        expected_keywords=["microbiome", "gut", "metabolic", "obesity", "bacteria"],
        category="cs.LG",  # ML papers — no nutrition science content expected
    ),
    RAGCase(
        label="adv_music_theory",
        query="How does serialism differ from atonality in twentieth-century classical composition?",
        expected_keywords=["serialism", "atonality", "twelve-tone", "Schoenberg", "composition"],
        category="cs.AI",  # AI papers — no music theory content expected
    ),
    RAGCase(
        label="adv_nuclear_physics",
        query="What are the engineering challenges in maintaining plasma confinement in a tokamak fusion reactor?",
        expected_keywords=["tokamak", "plasma", "fusion", "confinement", "magnetic field"],
        category="cs.CL",  # NLP papers — no nuclear physics content expected
    ),
    RAGCase(
        label="adv_marine_biology",
        query="How do deep-sea hydrothermal vent ecosystems support chemosynthetic food webs?",
        expected_keywords=["hydrothermal", "chemosynthesis", "deep-sea", "ecosystem", "vent"],
        category="cs.RO",  # robotics papers — no marine biology content expected
    ),
    RAGCase(
        label="adv_macroeconomics",
        query="What mechanisms explain stagflation and how did central banks respond in the 1970s?",
        expected_keywords=["stagflation", "oil shock", "central bank", "inflation", "unemployment"],
        category="cs.AI",  # AI papers — no macroeconomics content expected
    ),
    RAGCase(
        label="adv_archaeology",
        query="What dating methods are used to establish the chronology of Bronze Age settlements?",
        expected_keywords=["radiocarbon", "stratigraphy", "Bronze Age", "dating", "excavation"],
        category="cs.LG",  # ML papers — no archaeology content expected
    ),
    # -----------------------------------------------------------------------
    # Axis 2: Within-CS but not AI/ML/NLP/Robotics — harder adversarial cases.
    # The LLM has very strong parametric knowledge in these CS subfields,
    # making grounded refusal more difficult.
    # -----------------------------------------------------------------------
    RAGCase(
        label="adv_wrong_category",
        query="Explain the latest advances in large language model training.",
        expected_keywords=["language model", "training", "transformer"],
        category="cs.RO",  # robotics DB — intentionally wrong category for NLP query
    ),
    RAGCase(
        label="adv_hardware_architecture",
        query="What are the trade-offs between RISC-V and ARM processor architectures?",
        expected_keywords=["RISC-V", "ARM", "processor", "ISA", "microarchitecture"],
        category="cs.CL",  # NLP papers — no hardware architecture content expected
    ),
    RAGCase(
        label="adv_cryptography",
        query="How do zero-knowledge proofs enable privacy-preserving authentication without revealing secrets?",
        expected_keywords=["zero-knowledge", "proof", "cryptography", "authentication", "verifier"],
        category="cs.RO",  # robotics papers — no cryptography content expected
    ),
    RAGCase(
        label="adv_distributed_systems",
        query="What guarantees does the Raft consensus algorithm provide compared to Paxos?",
        expected_keywords=["Raft", "Paxos", "consensus", "distributed", "leader election"],
        category="cs.AI",  # AI papers — no distributed systems content expected
    ),
    RAGCase(
        label="adv_computer_networking",
        query="How does TCP congestion control adapt transmission rate using the AIMD algorithm?",
        expected_keywords=["TCP", "congestion", "AIMD", "bandwidth", "retransmission"],
        category="cs.LG",  # ML papers — no networking content expected
    ),
    RAGCase(
        label="adv_operating_systems",
        query="How does copy-on-write memory management reduce overhead in process forking?",
        expected_keywords=["copy-on-write", "fork", "virtual memory", "page fault", "kernel"],
        category="cs.CL",  # NLP papers — no OS content expected
    ),
    RAGCase(
        label="adv_computer_graphics",
        query="How does path tracing approximate global illumination in physically-based rendering?",
        expected_keywords=[
            "path tracing",
            "global illumination",
            "Monte Carlo",
            "rendering",
            "BRDF",
        ],
        category="cs.RO",  # robotics papers — no computer graphics content expected
    ),
    RAGCase(
        label="adv_database_systems",
        query="What is the difference between MVCC and two-phase locking for transaction isolation?",
        expected_keywords=["MVCC", "two-phase locking", "transaction", "isolation", "concurrency"],
        category="cs.AI",  # AI papers — no database systems content expected
    ),
    RAGCase(
        label="adv_compilers",
        query="How does LLVM's intermediate representation enable cross-platform optimization passes?",
        expected_keywords=["LLVM", "IR", "optimization", "compiler", "backend"],
        category="cs.LG",  # ML papers — no compiler theory content expected
    ),
    RAGCase(
        label="adv_formal_verification",
        query="How does the Coq proof assistant use dependent types to verify program correctness?",
        expected_keywords=[
            "Coq",
            "dependent types",
            "proof assistant",
            "formal verification",
            "theorem",
        ],
        category="cs.CL",  # NLP papers — no formal methods content expected
    ),
]

RAG_CASES: list[RAGCase] = [
    # Queries are paper-specific — answers require retrieved content, not parametric knowledge.
    # Category filters removed: metadata in this DB does not include a category field.
    RAGCase(
        label="bert_as_judge",
        query="What does BERT-as-a-Judge propose as an alternative to lexical methods for LLM evaluation, and what are its claimed advantages?",
        expected_keywords=["BERT", "lexical", "evaluation", "reference-based", "efficient"],
    ),
    RAGCase(
        label="recallm_lost_in_thought",
        query="What is the lost-in-thought phenomenon that RecaLLM addresses, and how does its in-context retrieval approach work?",
        expected_keywords=[
            "RecaLLM",
            "lost-in-thought",
            "in-context retrieval",
            "reasoning",
            "long-context",
        ],
    ),
    RAGCase(
        label="llm_harmful_mechanism",
        query="What unified mechanism do the retrieved papers identify for how LLMs generate harmful content despite alignment training?",
        expected_keywords=["harmful", "alignment", "mechanism", "jailbreak", "safeguard"],
    ),
    RAGCase(
        label="visionFoundry_approach",
        query="What training approach does VisionFoundry use to improve VLM visual perception, and what problem does it solve?",
        expected_keywords=["VisionFoundry", "synthetic", "visual perception", "VLM", "spatial"],
    ),
    RAGCase(
        label="visor_agentic_rag",
        query="What does VISOR stand for and how does it handle complex multi-step visual queries differently from standard RAG?",
        expected_keywords=["VISOR", "iterative", "visual", "agentic", "over-horizon"],
    ),
    RAGCase(
        label="xfed_federated_attack",
        query="What makes XFED's model poisoning attack against federated learning non-collusive, and why is that significant?",
        expected_keywords=["XFED", "non-collusive", "federated", "poisoning", "Byzantine"],
    ),
    RAGCase(
        label="safemind_quadruped",
        query="What safety framework does SafeMind propose for quadruped locomotion and what guarantees does it provide?",
        expected_keywords=["SafeMind", "quadruped", "safety", "differentiable", "uncertainty"],
    ),
    RAGCase(
        label="echo_chest_xray",
        query="What diffusion approach does ECHO use for chest X-ray report generation and what efficiency problem does it solve?",
        expected_keywords=["ECHO", "chest X-ray", "diffusion", "one-step", "autoregressive"],
    ),
    RAGCase(
        label="many_tier_instruction",
        query="What instruction hierarchy does the retrieved paper describe for LLM agents and why does trust level matter?",
        expected_keywords=[
            "instruction hierarchy",
            "trust",
            "authority",
            "system message",
            "LLM agent",
        ],
    ),
    RAGCase(
        label="process_reward_agents",
        query="How do process reward agents steer knowledge-intensive reasoning differently from outcome-based reward approaches?",
        expected_keywords=[
            "process reward",
            "knowledge-intensive",
            "intermediate",
            "verifiable",
            "reasoning",
        ],
    ),
    RAGCase(
        label="case_grounded_evidence",
        query="What is the case-grounded evidence verification framework and how does it construct evidence-sensitive supervision?",
        expected_keywords=["evidence", "grounded", "supervision", "verification", "claim"],
    ),
    RAGCase(
        label="vl_calibration",
        query="What decoupled calibration approach does VL-Calibration use to reduce hallucinations in vision-language models?",
        expected_keywords=[
            "VL-Calibration",
            "calibration",
            "hallucination",
            "confidence",
            "vision-language",
        ],
    ),
    RAGCase(
        label="safeadapt_rl",
        query="What safety guarantee does SafeAdapt provide for policy updates in reinforcement learning under non-stationary environments?",
        expected_keywords=[
            "SafeAdapt",
            "safe",
            "policy update",
            "non-stationary",
            "reinforcement learning",
        ],
    ),
    RAGCase(
        label="e3_tir_tool_reasoning",
        query="What limitations in tool-integrated reasoning does E3-TIR address and how does it exploit experience differently?",
        expected_keywords=["E3-TIR", "tool-integrated", "reasoning", "experience", "Zero-RL"],
    ),
    RAGCase(
        label="semantic_rate_distortion",
        query="What is the semantic rate-distortion framework for multi-agent communication described in the retrieved papers?",
        expected_keywords=[
            "semantic",
            "rate-distortion",
            "multi-agent",
            "communication",
            "alignment",
        ],
    ),
]
