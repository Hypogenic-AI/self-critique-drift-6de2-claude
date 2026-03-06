# Literature Review: Representation Drift Under Self-Reflection

## Research Area Overview

This research sits at the intersection of three active areas: (1) LLM self-correction and self-critique, (2) probing and analyzing internal representations during reasoning, and (3) representation geometry and engineering. The core question is whether self-critique causes structured, meaningful shifts in a model's internal representations (residual stream activations), or whether it is a superficial process that merely perturbs the output without genuine restructuring of the reasoning subspace.

## Key Papers

### 1. Huang et al. (2024) - "Large Language Models Cannot Self-Correct Reasoning Yet" (ICLR 2024)

- **Key Contribution**: Demonstrates that intrinsic self-correction (without external feedback) degrades LLM performance. Prior claims of self-correction success relied on oracle labels, unfair baselines, or sub-optimal initial prompts.
- **Methodology**: Three-step prompting (generate, critique, revise) on GSM8K, CommonSenseQA, HotpotQA. Up to 2 rounds of self-correction.
- **Datasets**: GSM8K (1,319 test), CommonSenseQA (1,221 dev), HotpotQA (100), CommonGen-Hard
- **Models**: GPT-3.5-Turbo, GPT-4, GPT-4-Turbo, Llama-2-70b-chat
- **Key Results**: GPT-3.5 on CSQA drops 75.8% to 38.1% after self-correction. Models more likely to change correct answers to incorrect than vice versa. GPT-4 retains original answers ~90-96% of the time (more resistant).
- **Relevance**: Establishes the behavioral baseline. If self-correction is indeed shallow, we should see minimal structured representation drift. Their observation that weaker models are more susceptible to self-correction perturbation suggests representation stability correlates with model capability.

### 2. Madaan et al. (2023) - "Self-Refine: Iterative Refinement with Self-Feedback"

- **Key Contribution**: Iterative self-critique framework where the model generates output, provides feedback, then refines.
- **Methodology**: Multi-round generate-feedback-refine loops with separate prompts for each stage.
- **Key Insight**: Huang et al. later showed the apparent gains were due to sub-optimal initial prompts, not genuine self-correction capability.
- **Relevance**: Provides the canonical self-critique pipeline structure (generate -> critique -> revise) that our experiment will follow.

### 3. Shinn et al. (2023) - "Reflexion: Language Agents with Verbal Reinforcement Learning"

- **Key Contribution**: Verbal self-reflection as a feedback mechanism for language agents.
- **Relevance**: Provides self-reflection methodology that can be adapted for our pipeline. Uses external signal (task success/failure) to guide reflection, unlike pure intrinsic self-correction.

### 4. Zou et al. (2023) - "Representation Engineering: A Top-Down Approach to AI Transparency"

- **Key Contribution**: Introduces Representation Engineering (RepE) framework using Linear Artificial Tomography (LAT). Extracts concept directions from hidden states using PCA on stimulus-response contrast pairs.
- **Methodology**:
  - Create contrast pairs (e.g., honest vs. dishonest responses)
  - Extract hidden states from all layers
  - Apply PCA to find the direction that maximally separates the contrasts
  - Use this direction for both reading (monitoring) and control (steering)
- **Models**: LLaMA-2-13B-chat primarily
- **Key Results**: Successfully extracts and manipulates directions for honesty, harmlessness, emotion, power-seeking, etc.
- **Relevance**: The LAT/PCA methodology is directly applicable. We can extract "pre-critique" vs. "post-critique" representation directions to measure whether self-critique creates meaningful geometric shifts. Code: github.com/andyzoujm/representation-engineering

### 5. Marks & Tegmark (2024) - "The Geometry of Truth" (COLM 2024)

- **Key Contribution**: Demonstrates clear linear structure in how LLMs represent truth values. Mass-mean probes outperform logistic regression for causal interventions.
- **Methodology**:
  - Extract residual stream activations at end-of-sentence punctuation token
  - Layer 15 of LLaMA-2-13B (identified via activation patching)
  - PCA visualization shows true/false separation in top 2 PCs
  - Mass-mean probes: direction = mu_true - mu_false
  - Causal interventions: add/subtract truth direction to flip model predictions
- **Datasets**: Curated true/false datasets (cities, translations, number comparisons)
- **Models**: LLaMA-2-7B/13B/70B
- **Key Results**: Linear truth structure emerges with scale. Probes transfer across datasets. Mass-mean directions are more causally implicated than logistic regression.
- **Relevance**: Provides the core probing methodology. We can apply mass-mean probes to compare pre-critique and post-critique representations. PCA visualization can reveal whether self-critique moves representations to a distinct subspace. Code: github.com/saprmarks/geometry-of-truth

### 6. Burns et al. (2023) - "Discovering Latent Knowledge Without Supervision" (CCS)

- **Key Contribution**: Unsupervised method for extracting truth directions from contrast pairs without labeled data.
- **Methodology**: Contrast Consistent Search (CCS) - finds directions where opposite statements have opposite projections.
- **Relevance**: CCS can be applied as an unsupervised baseline for detecting whether self-critique shifts representations toward truth.

### 7. Turpin et al. (2024) - "Language Models Don't Always Say What They Think"

- **Key Contribution**: Shows chain-of-thought explanations can be unfaithful to the model's actual reasoning process. Biasing features in inputs systematically affect answers without being mentioned in CoT.
- **Relevance**: Motivates studying internal representations rather than relying on surface-level critique text. If CoT is unfaithful, self-critique text may not reflect actual internal processing.

### 8. Garner et al. (2025) - "Probing for Arithmetic Errors in Language Models"

- **Key Contribution**: Trains probes on residual stream activations to detect arithmetic errors. Finds models internally represent correct answers even when outputting incorrect ones.
- **Methodology**:
  - Circular, MLP, and logistic probes on residual stream at "=" token
  - Layer-by-layer analysis (all 26 layers of Gemma 2 2B IT)
  - Separate probes for model output vs. ground truth
  - Error detection >90% accuracy by combining both signals
- **Datasets**: Synthetic 3-digit addition (800 examples), GSM8K filtered for addition
- **Key Results**: Ground truth is decodable from hidden states at >90% accuracy even when model outputs wrong answer. Probes generalize from pure arithmetic to CoT setting.
- **Relevance**: Directly demonstrates that internal representations contain richer information than outputs. Our experiment can probe whether self-critique helps align the output with the internally-represented correct answer.

### 9. David (2025) - "Temporal Predictors of Outcome in Reasoning Language Models"

- **Key Contribution**: Linear probes predict eventual CoT correctness from very early hidden states (AUC 0.84 at just 4 tokens).
- **Methodology**:
  - Pool final hidden states of last 4 reasoning tokens at fixed prefix lengths
  - PCA to 128 components, then L2-regularized logistic regression
  - Evaluated at t in {4, 8, 16, 32, 64, 128, 192, 256, 384, 512}
- **Datasets**: Hendrycks MATH (1,500 problems, balanced easy/hard)
- **Models**: Qwen3-8B, Llama3.1-8B-Instruct
- **Key Results**: Early correctness signal is linearly decodable. Difficulty-driven temporal selection effect explains apparent later degradation.
- **Relevance**: Shows correctness information is encoded early in hidden trajectory. We can compare whether self-critique changes the trajectory of these internal correctness signals.

### 10. Anderson (2026) - "The Geometry of Thought"

- **Key Contribution**: Dynamical systems analysis of reasoning trajectories in representation space. Identifies three geometric phases (crystalline, liquid, lattice) across domains.
- **Methodology**:
  - Reasoning trajectory: sequence of final-layer hidden states during CoT
  - Six geometric metrics: global dimension (d95), intrinsic dimension (MLE), trajectory alignment, step-to-step coherence, silhouette score, global-to-local dimension ratio
  - Two-pass generate-then-extract protocol
- **Datasets**: GSM8K, LexGLUE-SCOTUS, GPQA, HumanEval
- **Models**: Llama-3-8B-Instruct, Llama-3.1-70B-Instruct
- **Key Results**: Universal oscillatory signature (step-to-step coherence ~ -0.4). Low intrinsic dimension (15-25) across all conditions. Scale restructures rather than uniformly improves reasoning geometry.
- **Relevance**: Provides the geometric measurement toolkit for our analysis. We can apply their metrics to compare pre-critique vs. post-critique reasoning trajectories.

## Common Methodologies

### Probing Techniques (used across papers)
- **Linear probes / Logistic regression**: Train simple classifiers on hidden states (Marks, Garner, David)
- **Mass-mean probes**: Direction = mean_positive - mean_negative, more causally implicated (Marks)
- **PCA visualization**: Project activations onto top principal components (all papers)
- **CCS (Contrast Consistent Search)**: Unsupervised truth direction extraction (Burns)
- **LAT (Linear Artificial Tomography)**: PCA on contrast pair activations (Zou)

### Representation Extraction
- **Residual stream activations**: Primary representation studied (all papers)
- **Token positions**: End-of-sentence (Marks), "=" token (Garner), last N tokens (David), full trajectory (Anderson)
- **Layers**: Middle layers most informative for truth (layer 15 for LLaMA-2-13B per Marks), error detection peaks in final 3-5 layers (Garner)

### Causal Analysis
- **Activation patching**: Swap activations between runs to identify causal nodes (Marks)
- **Representation steering**: Add/subtract concept directions to change model behavior (Marks, Zou)

## Standard Baselines
- Self-consistency / majority voting (Huang et al.)
- Few-shot prompting accuracy (Marks)
- Next-token entropy (David)
- Random / majority class baselines

## Evaluation Metrics
- **Probe accuracy and ROC-AUC**: For classification probes (all probing papers)
- **Normalized Indirect Effect (NIE)**: For causal interventions (Marks)
- **Cosine similarity**: For measuring representation drift direction
- **PCA variance explained**: For measuring dimensionality
- **Silhouette score**: For measuring clustering quality (Anderson)
- **Trajectory alignment**: For measuring canonical pathway adherence (Anderson)
- **L2 distance in representation space**: For measuring magnitude of drift

## Datasets in the Literature
- **GSM8K**: Grade school math, used by Huang, Garner, Anderson (7,473 train / 1,319 test)
- **MATH (Hendrycks)**: Competition math with difficulty levels, used by David (500-1500 problems)
- **CommonSenseQA**: Commonsense reasoning, used by Huang (1,221 dev)
- **TruthfulQA**: Truthfulness evaluation (817 questions)
- **True/false statement datasets**: Cities, translations, numbers (Marks, custom)

## Gaps and Opportunities

1. **No study of representation drift during self-critique**: Existing work either studies behavioral outcomes of self-correction (Huang) or probes representations during initial generation (Garner, David), but none tracks how representations shift across the generate-critique-revise pipeline.

2. **Missing connection between probing and self-correction**: Garner shows models "know" correct answers internally; Huang shows they can't self-correct behaviorally. The question of whether self-critique moves internal representations toward or away from the internally-represented correct answer is unstudied.

3. **No geometric analysis of self-critique**: Anderson's geometric toolkit has been applied to initial reasoning trajectories but not to the specific perturbation caused by self-critique.

4. **Linear separability before vs. after critique**: Whether self-critique improves the linear decodability of correct reasoning from hidden states has not been measured.

## Recommendations for Our Experiment

### Recommended Datasets
1. **GSM8K** (primary): Well-studied, has ground truth, used across papers, manageable size
2. **MATH-500** (secondary): Provides difficulty stratification for analyzing whether self-critique effects depend on problem difficulty
3. **TruthfulQA** (tertiary): Different domain (factual questions) to test generalization

### Recommended Models
- **LLaMA-3-8B-Instruct** or **Qwen3-8B**: Small enough for activation extraction, large enough for meaningful self-critique behavior. Both used in recent papers.
- Consider also **LLaMA-2-13B**: For comparability with Marks & Tegmark and Zou et al.

### Recommended Methodology
1. **Three-phase pipeline**: Generate initial answer -> Generate self-critique -> Generate revised answer
2. **Extract residual stream activations** at each phase (all layers, key token positions)
3. **Probing**: Train linear probes on pre-critique vs. post-critique representations to predict correctness
4. **Geometric analysis**: Apply Anderson's metrics (d95, trajectory alignment, coherence) to pre/post-critique trajectories
5. **Drift measurement**: Cosine similarity, L2 distance, PCA projection overlap between pre/post-critique states
6. **Control experiment**: Compare self-critique drift to random prompt continuation (to establish baseline drift)

### Recommended Metrics
1. **Probe accuracy/AUC** before vs. after self-critique (does linear separability of correctness improve?)
2. **Cosine similarity** between pre-critique and post-critique residual stream states
3. **PCA overlap**: Do post-critique states occupy a distinct subspace?
4. **Trajectory alignment**: Do revised reasoning traces converge toward a more structured manifold?
5. **Behavioral accuracy change**: Correlate representation drift with actual answer quality change

### Methodological Considerations
- Use the same model for all three phases (generate, critique, revise) to ensure representations are comparable
- Extract activations at the last content token before answer delimiter (following Marks' approach)
- Analyze multiple layers (especially middle layers ~layer 15 for 13B models)
- Include null hypothesis test: random re-prompting without critique as control
- Balance correct/incorrect initial answers in analysis
