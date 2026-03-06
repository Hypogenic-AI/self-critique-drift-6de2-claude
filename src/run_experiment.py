import os
os.environ['USER'] = 'researcher'
os.environ['LOGNAME'] = 'researcher'

import random
import numpy as np
import torch
import sys
import time

from config import *
from pipeline import load_model_and_tokenizer, load_gsm8k, run_pipeline, save_results

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

start = time.time()
model, tokenizer = load_model_and_tokenizer()
problems = load_gsm8k()

results, activations = run_pipeline(model, tokenizer, problems, save_every=25)
save_results(results, activations, partial=False)

elapsed = time.time() - start
n = len(results)
initial_acc = sum(r["initial_correct"] for r in results) / n
revised_acc = sum(r["revised_correct"] for r in results) / n
control_acc = sum(r["control_correct"] for r in results) / n
print(f"\n=== Summary ===")
print(f"N = {n}")
print(f"Initial accuracy: {initial_acc:.3f}")
print(f"Revised accuracy: {revised_acc:.3f}")
print(f"Control accuracy: {control_acc:.3f}")
print(f"Time: {elapsed/60:.1f} minutes")
