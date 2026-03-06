"""Configuration for the self-critique representation drift experiment."""
import os

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"

# Dataset
NUM_PROBLEMS = 200  # Number of GSM8K problems to use
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "gsm8k", "data")

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0  # Greedy for reproducibility
SEED = 42

# Activation extraction
# We extract the residual stream at the last generated token
# across all layers
LAYERS_TO_ANALYZE = "all"  # Will be set dynamically based on model

# Prompts
INITIAL_PROMPT_TEMPLATE = """Solve the following math problem step by step. After your reasoning, provide your final answer as a number after "#### ".

Problem: {question}

Solution:"""

CRITIQUE_PROMPT_TEMPLATE = """You previously solved this math problem:

Problem: {question}

Your solution was:
{initial_answer}

Now carefully critique your own solution. Identify any errors in reasoning or calculation. Be specific about what might be wrong.

Critique:"""

REVISE_PROMPT_TEMPLATE = """You previously solved this math problem:

Problem: {question}

Your original solution was:
{initial_answer}

Your self-critique identified these issues:
{critique}

Now provide a revised solution, correcting any errors you found. Provide your final answer as a number after "#### ".

Revised solution:"""

CONTROL_PROMPT_TEMPLATE = """Solve the following math problem step by step. After your reasoning, provide your final answer as a number after "#### ".

Problem: {question}

Please solve this carefully.

Solution:"""

# Results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
