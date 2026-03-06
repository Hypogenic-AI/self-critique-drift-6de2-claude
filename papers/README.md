# Downloaded Papers

## Core Papers: Self-Correction and Self-Critique

1. **Large Language Models Cannot Self-Correct Reasoning Yet** (huang2023_self_correct_reasoning.pdf)
   - Authors: Jie Huang, Xinyun Chen, et al. (Google DeepMind / UIUC)
   - Year: 2024 (ICLR 2024), arXiv: 2310.01798
   - Why relevant: Core negative result showing LLMs cannot self-correct without external feedback; performance degrades after self-critique

2. **Self-Refine: Iterative Refinement with Self-Feedback** (madaan2023_self_refine.pdf)
   - Authors: Aman Madaan, Niket Tandon, et al.
   - Year: 2023, arXiv: 2303.17651
   - Why relevant: Foundational self-critique framework; shows iterative refinement paradigm

3. **Reflexion: Language Agents with Verbal Reinforcement Learning** (shinn2023_reflexion.pdf)
   - Authors: Noah Shinn, et al.
   - Year: 2023, arXiv: 2303.11366
   - Why relevant: Self-reflection in language agents; uses verbal self-critique for improvement

4. **Self-Contrast: Better Reflection Through Inconsistent Solving** (mitchell2024_self_contrast.pdf)
   - Authors: Mitchell et al.
   - Year: 2024, arXiv: 2401.02009
   - Why relevant: Alternative approach to self-reflection through contrastive generation

## Core Papers: Representation Analysis and Probing

5. **Representation Engineering: A Top-Down Approach to AI Transparency** (zou2023_representation_engineering.pdf)
   - Authors: Andy Zou, et al. (CAIS / CMU)
   - Year: 2023, arXiv: 2310.01405
   - Why relevant: Foundational RepE framework; LAT vectors, PCA-based probing, representation reading/control. Code: github.com/andyzoujm/representation-engineering

6. **The Geometry of Truth: Emergent Linear Structure in LLM Representations** (marks2023_geometry_of_truth.pdf)
   - Authors: Samuel Marks, Max Tegmark
   - Year: 2024 (COLM 2024), arXiv: 2310.06824
   - Why relevant: Linear probing of truth representations in residual stream; mass-mean probes; causal interventions. Code: github.com/saprmarks/geometry-of-truth

7. **Discovering Latent Knowledge in Language Models Without Supervision** (burns2023_discovering_latent_knowledge.pdf)
   - Authors: Collin Burns, et al.
   - Year: 2023, arXiv: 2212.03827
   - Why relevant: CCS method for unsupervised truth discovery from contrast pairs

8. **Language Models Don't Always Say What They Think** (turpin2024_models_dont_say_what_think.pdf)
   - Authors: Miles Turpin, et al.
   - Year: 2024, arXiv: 2305.04388
   - Why relevant: Shows CoT can be unfaithful to actual model reasoning; motivates studying internal states vs. outputs

## Core Papers: Probing During Reasoning

9. **Probing for Arithmetic Errors in Language Models** (garner2025_probing_arithmetic_errors.pdf)
   - Authors: Garner et al.
   - Year: 2025, arXiv: 2507.12379
   - Why relevant: Probes residual stream to detect errors; finds models internally represent correct answers even when outputting wrong ones

10. **Temporal Predictors of Outcome in Reasoning Language Models** (temporal_predictors_reasoning_2025.pdf)
    - Authors: Joey David
    - Year: 2025, arXiv: 2511.14773
    - Why relevant: Linear probes predict correctness from early CoT hidden states (AUC 0.84 at t=4 tokens)

11. **The Geometry of Thought: How Scale Restructures Reasoning** (geometry_of_thought_2026.pdf)
    - Authors: Anderson
    - Year: 2026, arXiv: 2601.13358
    - Why relevant: Analyzes reasoning trajectories in representation space; geometric metrics for manifold analysis

## Supporting Papers

12. **Improving Reasoning Performance via Representation Engineering** (improving_reasoning_repeng_2025.pdf)
    - Year: 2025, arXiv: 2504.19483
    - Why relevant: Applies RepE specifically to reasoning improvement

13. **System-1.5 Reasoning: Traversal in Language and Latent Spaces** (system15_reasoning_2025.pdf)
    - Year: 2025, arXiv: 2505.18962
    - Why relevant: Latent-space reasoning connecting CoT to internal representations

14. **A Statistical Physics of Language Model Reasoning** (stat_physics_reasoning_2025.pdf)
    - Year: 2025, arXiv: 2506.04374
    - Why relevant: Dynamical systems framework for hidden-state trajectories during reasoning

15. **No Global Plan in Chain-of-Thought** (no_global_plan_cot_2026.pdf)
    - Year: 2026, arXiv: 2602.02103
    - Why relevant: Studies latent planning dynamics during CoT generation
