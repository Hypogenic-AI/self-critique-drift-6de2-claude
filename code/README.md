# Cloned Repositories

## 1. Representation Engineering (RepE)
- **URL**: https://github.com/andyzoujm/representation-engineering
- **Purpose**: Framework for reading and controlling LLM internal representations using Linear Artificial Tomography (LAT). Provides tools for extracting representation directions via PCA on contrast pairs, and for steering model behavior by adding/subtracting these directions.
- **Location**: code/representation-engineering/
- **Key files**:
  - `repe/` - Core library: RepE reading vectors, control vectors, rep readers
  - `examples/` - Jupyter notebooks demonstrating honesty, emotion, fairness probing
  - `data/` - Stimulus datasets for extracting concept representations
- **Relevance to our research**: Provides the primary toolkit for extracting and analyzing representation directions. The LAT/PCA methodology can be adapted to extract "self-critique" directions by comparing pre- vs. post-critique hidden states.

## 2. Geometry of Truth
- **URL**: https://github.com/saprmarks/geometry-of-truth
- **Purpose**: Probing truth representations in LLMs using mass-mean probes, logistic regression, PCA visualization, and causal interventions on residual stream activations.
- **Location**: code/geometry-of-truth/
- **Key files**:
  - `generate_acts.py` - Extract activations from LLaMA-2 models
  - `probes.py` - Mass-mean probes, logistic regression probes, CCS
  - `visualization_utils.py` - PCA visualization tools
  - `patching.py` - Activation patching for causal analysis
  - `interventions.py` - Causal intervention experiments
  - `datasets/` - True/false statement datasets (cities, translations, numbers)
- **Relevance to our research**: Directly applicable methodology for analyzing linear structure in representations before/after self-critique. Mass-mean probing and PCA visualization can reveal whether self-critique creates structured shifts.

## 3. Reflexion
- **URL**: https://github.com/noahshinn/reflexion
- **Purpose**: Language agents with verbal reinforcement learning through self-reflection.
- **Location**: code/reflexion/
- **Key files**: Implementation of self-reflection loops for code generation and reasoning tasks.
- **Relevance to our research**: Provides self-reflection prompting methodology that can be adapted for our generate-critique-revise pipeline.
