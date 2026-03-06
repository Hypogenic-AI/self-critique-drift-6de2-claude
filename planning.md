# Research Plan: Representation Drift Under Self-Reflection

## Motivation & Novelty Assessment

### Why This Research Matters
Self-critique and iterative reflection are increasingly used to improve LLM outputs, yet we lack understanding of whether this improvement reflects genuine internal representational restructuring or mere surface-level re-sampling. Understanding this distinction is critical for: (a) designing effective self-improvement agents, (b) knowing when reflection adds value vs. noise, and (c) connecting behavioral gains to mechanistic changes.

### Gap in Existing Work
Based on our literature review:
- Huang et al. (2024) showed self-correction often *degrades* performance behaviorally, but didn't examine internal representations
- Garner et al. (2025) showed models internally represent correct answers even when outputting wrong ones, but didn't study how self-critique affects this
- Marks & Tegmark (2024) developed truth probing tools but applied them to static statements, not to the dynamic process of self-critique
- Anderson (2026) analyzed reasoning geometry but only for initial generation, not across critique-revision cycles
- **No study has tracked how residual stream representations shift across the generate-critique-revise pipeline**

### Our Novel Contribution
We directly measure whether self-critique induces structured geometric shifts in the residual stream by:
1. Extracting activations at three pipeline stages (initial answer, post-critique, revised answer)
2. Measuring drift via cosine similarity, L2 distance, PCA subspace analysis
3. Testing whether linear separability of correct reasoning improves after critique
4. Comparing self-critique drift to a control condition (re-prompting without critique)

### Experiment Justification
- **Experiment 1 (Drift Measurement)**: Quantifies the magnitude and structure of representational change. If drift is near-zero, self-critique is shallow.
- **Experiment 2 (Probe Accuracy)**: Tests whether post-critique representations are more linearly separable for correctness prediction. If probe accuracy increases, critique restructures toward a more informative subspace.
- **Experiment 3 (Control Comparison)**: Compares self-critique drift to simple re-prompting drift. If self-critique drift is not significantly larger/more structured than control drift, critique adds nothing beyond re-sampling.
- **Experiment 4 (Behavioral Correlation)**: Tests whether larger drift correlates with actual answer improvement. Links internal dynamics to behavioral outcomes.

## Research Question
When an LLM generates an initial answer, critiques itself, and then revises, does self-critique induce structured shifts in the model's residual stream representations, or is the representational change indistinguishable from random re-sampling?

## Hypothesis Decomposition
H1: Self-critique induces measurable drift in residual stream activations (cosine similarity < 1.0 between pre/post-critique states)
H2: Post-critique representations show improved linear separability for correctness (probe AUC increases)
H3: Self-critique drift is larger and more structured than control (re-prompting) drift
H4: Drift magnitude correlates with behavioral outcome change (answer flips)

## Proposed Methodology

### Approach
Use a local open-source model (Qwen2.5-7B-Instruct) to run the three-phase self-critique pipeline on GSM8K problems, extracting residual stream activations at each phase. Compare representations using probing, geometric metrics, and statistical tests.

### Model Choice
Qwen2.5-7B-Instruct: Recent, capable at math reasoning, fits on a single 3090, well-supported by HuggingFace transformers. Alternative: Llama-3.1-8B-Instruct.

### Experimental Steps
1. Load model and GSM8K test set (use ~200 problems for tractability)
2. For each problem, run 3 conditions:
   - **Phase 1 (Initial)**: Generate answer, extract last-token residual stream activations from all layers
   - **Phase 2 (Critique)**: Prompt model to critique its own answer, extract activations
   - **Phase 3 (Revised)**: Prompt model to revise based on critique, extract activations
   - **Control**: Re-prompt model to answer again without critique, extract activations
3. Parse answers and evaluate correctness (initial, revised, control)
4. Compute drift metrics between phases
5. Train linear probes on initial vs. revised representations
6. Statistical analysis of all metrics

### Baselines
- **Control (re-prompting)**: Same model re-answers without self-critique
- **Random direction baseline**: Random drift in activation space for comparison
- **Cross-problem baseline**: Drift between different problems (upper bound on meaningful drift)

### Evaluation Metrics
1. **Cosine similarity** between phase activations (per layer)
2. **L2 distance** in representation space
3. **PCA overlap**: Shared variance between pre/post-critique subspaces
4. **Linear probe AUC**: Correctness prediction from representations
5. **Behavioral accuracy**: % correct before/after critique
6. **Drift-outcome correlation**: Spearman correlation between drift magnitude and answer change

### Statistical Analysis Plan
- Paired t-tests / Wilcoxon signed-rank tests for before/after comparisons
- Bootstrap confidence intervals for drift metrics
- Multiple comparison correction via Benjamini-Hochberg FDR
- Significance level: alpha = 0.05
- Effect sizes: Cohen's d for all comparisons

## Expected Outcomes
- If self-critique is *meaningful*: significant drift, improved probe accuracy, structured PCA separation, drift > control
- If self-critique is *shallow*: minimal drift, no probe improvement, drift ~ control re-prompting

## Timeline
1. Environment setup & data prep: 15 min
2. Pipeline implementation: 60 min
3. Run experiments: 60 min
4. Analysis & visualization: 45 min
5. Documentation: 30 min

## Potential Challenges
- Model generation quality on GSM8K (may need prompt tuning)
- Activation extraction memory requirements (mitigate: process in batches)
- Answer parsing reliability (use regex patterns for GSM8K format)

## Success Criteria
- Clear quantitative evidence for or against structured representation drift
- Statistically significant comparison between self-critique and control conditions
- Reproducible pipeline with saved activations and metrics
