# Representation Drift Under Self-Reflection

Does self-critique reshape a language model's internal representations, or is it shallow re-sampling? This project measures residual stream activation drift across the generate-critique-revise pipeline.

## Key Findings

- **Self-critique induces significantly more representational drift than simple re-prompting** (cosine similarity ~0.57 vs ~0.69, p < 0.01 at all 28 layers after FDR correction)
- **The drift is destructive**: accuracy drops from 90.5% to 84.0% after self-critique; simple re-prompting achieves 92.0%
- **Linear separability of correctness decreases** after critique (probe AUC drops in most layers)
- **Drift is global** across all layers, not confined to specific computational stages
- **Drift weakly correlates with answer change** (r ~ 0.15-0.20), suggesting it reflects genuine internal perturbation

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformers accelerate datasets scikit-learn scipy matplotlib seaborn numpy pandas statsmodels

# Run experiment (requires GPU, ~2.5 hours on RTX 3090)
cd src && python run_experiment.py

# Run analysis
python analysis.py
```

## Project Structure

```
.
├── REPORT.md              # Full research report with results
├── planning.md            # Research plan and motivation
├── literature_review.md   # Literature synthesis
├── resources.md           # Resource catalog
├── src/
│   ├── config.py          # Experiment configuration
│   ├── pipeline.py        # Self-critique pipeline with activation extraction
│   ├── analysis.py        # Drift metrics, probing, PCA, statistical tests
│   └── run_experiment.py  # Main experiment runner
├── results/
│   ├── behavioral_results.json    # Per-problem accuracy data
│   ├── activations_*.npy          # Residual stream activations (200x28x3584)
│   ├── analysis_results.json      # All analysis metrics
│   └── plots/                     # Visualizations
├── datasets/              # GSM8K, MATH-500, TruthfulQA
├── papers/                # Reference papers (PDFs)
└── code/                  # Cloned reference implementations
```

## Model & Data

- **Model**: Qwen2.5-7B-Instruct (28 layers, 3584 hidden dim)
- **Dataset**: GSM8K test split (200 problems)
- **Hardware**: NVIDIA RTX 3090, 138 minutes total runtime

See [REPORT.md](REPORT.md) for complete methodology and results.
