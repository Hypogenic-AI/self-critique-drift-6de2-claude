# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?"

## Papers
Total papers downloaded: 15

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Large Language Models Cannot Self-Correct Reasoning Yet | Huang et al. | 2024 | papers/huang2023_self_correct_reasoning.pdf | ICLR 2024; self-correction degrades performance |
| 2 | Self-Refine: Iterative Refinement with Self-Feedback | Madaan et al. | 2023 | papers/madaan2023_self_refine.pdf | Foundational self-critique framework |
| 3 | Reflexion: Language Agents with Verbal RL | Shinn et al. | 2023 | papers/shinn2023_reflexion.pdf | Self-reflection in language agents |
| 4 | Self-Contrast: Better Reflection Through Inconsistent Solving | Mitchell et al. | 2024 | papers/mitchell2024_self_contrast.pdf | Contrastive self-reflection |
| 5 | Representation Engineering: A Top-Down Approach | Zou et al. | 2023 | papers/zou2023_representation_engineering.pdf | RepE framework, LAT vectors, PCA probing |
| 6 | The Geometry of Truth | Marks & Tegmark | 2024 | papers/marks2023_geometry_of_truth.pdf | Linear truth probes, mass-mean method, causal interventions |
| 7 | Discovering Latent Knowledge Without Supervision | Burns et al. | 2023 | papers/burns2023_discovering_latent_knowledge.pdf | CCS unsupervised truth extraction |
| 8 | Language Models Don't Always Say What They Think | Turpin et al. | 2024 | papers/turpin2024_models_dont_say_what_think.pdf | Unfaithful CoT reasoning |
| 9 | Probing for Arithmetic Errors in Language Models | Garner et al. | 2025 | papers/garner2025_probing_arithmetic_errors.pdf | Error detection from residual stream |
| 10 | Temporal Predictors of Outcome in Reasoning LMs | David | 2025 | papers/temporal_predictors_reasoning_2025.pdf | Early correctness probing during CoT |
| 11 | The Geometry of Thought | Anderson | 2026 | papers/geometry_of_thought_2026.pdf | Geometric analysis of reasoning trajectories |
| 12 | Improving Reasoning via Representation Engineering | - | 2025 | papers/improving_reasoning_repeng_2025.pdf | RepE applied to reasoning |
| 13 | System-1.5 Reasoning | - | 2025 | papers/system15_reasoning_2025.pdf | Latent-space reasoning |
| 14 | A Statistical Physics of Language Model Reasoning | - | 2025 | papers/stat_physics_reasoning_2025.pdf | Dynamical systems framework for reasoning |
| 15 | No Global Plan in Chain-of-Thought | - | 2026 | papers/no_global_plan_cot_2026.pdf | Latent planning dynamics in CoT |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | HuggingFace openai/gsm8k | 7,473 train + 1,319 test | Math reasoning | datasets/gsm8k/ | Primary dataset |
| MATH-500 | HuggingFace HuggingFaceH4/MATH-500 | 500 test | Competition math | datasets/math/ | Difficulty-stratified |
| TruthfulQA | HuggingFace truthfulqa/truthful_qa | 817 validation | Truthfulness | datasets/truthfulqa/ | Factual QA |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Representation Engineering | github.com/andyzoujm/representation-engineering | LAT vectors, RepE reading/control | code/representation-engineering/ | Core toolkit for representation extraction |
| Geometry of Truth | github.com/saprmarks/geometry-of-truth | Truth probing, mass-mean probes, PCA | code/geometry-of-truth/ | Probing methodology + datasets |
| Reflexion | github.com/noahshinn/reflexion | Self-reflection agent framework | code/reflexion/ | Self-critique pipeline reference |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Searched Semantic Scholar API with 9 different queries covering self-critique, representation drift, mechanistic interpretability, probing hidden states, and representation engineering
- Targeted papers from 2022-2026 for recency
- Prioritized papers with high citation counts and direct relevance
- Cross-referenced datasets and methods across papers

### Selection Criteria
- Papers selected based on direct relevance to the three pillars: (1) self-correction/critique in LLMs, (2) probing internal representations, (3) geometric analysis of reasoning
- Datasets chosen for: ground-truth availability, use in existing literature, manageable size for activation extraction
- Code repos chosen for: directly applicable methodology, active maintenance, clear documentation

### Challenges Encountered
- Hendrycks MATH full dataset requires authenticated HuggingFace access; used MATH-500 subset instead
- latent-correctness-probe GitHub repo not found at stated URL
- Some papers had mismatched arXiv IDs (corrected during search)

### Gaps and Workarounds
- No existing dataset specifically designed for self-critique representation analysis; will need to generate self-critique traces programmatically using the downloaded datasets as input
- No existing codebase combines self-critique generation with representation extraction; will need to build pipeline combining RepE/geometry-of-truth tools with self-critique prompting

## Recommendations for Experiment Design

### 1. Primary Dataset
**GSM8K** - Well-studied math reasoning benchmark with ground truth answers. Used by Huang et al. for self-correction experiments and by multiple probing papers. The 1,319 test examples provide sufficient data for probing analysis.

### 2. Experimental Pipeline
Using code from `representation-engineering` and `geometry-of-truth`:
1. Feed GSM8K problems to model, extract residual stream activations at answer token
2. Prompt model to critique its answer, extract activations during/after critique
3. Prompt model to revise, extract activations at revised answer token
4. Compare pre/post-critique representations using probes, PCA, and geometric metrics

### 3. Baseline Methods
- Mass-mean probes (from Marks & Tegmark) for correctness direction
- PCA visualization (from Zou et al.) for subspace analysis
- Linear probes (from David, Garner) for correctness prediction
- Random re-prompting control (no critique, just re-answer)

### 4. Evaluation Metrics
- Linear probe accuracy/AUC for correctness (before vs. after critique)
- Cosine similarity between pre/post-critique residual stream states
- PCA subspace overlap (shared variance between conditions)
- Anderson's geometric metrics (d95, trajectory alignment, coherence)
- Behavioral correlation (representation drift vs. answer change)

### 5. Code to Adapt/Reuse
- `code/geometry-of-truth/generate_acts.py` - Activation extraction pipeline
- `code/geometry-of-truth/probes.py` - Mass-mean and logistic probes
- `code/representation-engineering/repe/` - RepE reading vectors
- Self-critique prompting templates from Huang et al. methodology
