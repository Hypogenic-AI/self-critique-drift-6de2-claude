"""Self-critique pipeline with activation extraction.

Runs three phases per problem:
1. Initial answer generation
2. Self-critique of the answer
3. Revised answer based on critique
Plus a control condition (re-answer without critique).

Extracts residual stream activations at the last generated token for each phase.
"""

import torch
import numpy as np
import json
import re
import os
import gc
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *


def load_model_and_tokenizer():
    """Load model with hooks for activation extraction."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def extract_activations(model, input_ids, attention_mask):
    """Extract residual stream activations from all layers at the last token position.

    Returns: dict mapping layer_idx -> activation vector (numpy array)
    """
    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Get last token position (last non-padding token)
            seq_len = attention_mask.sum(dim=1).long() - 1
            last_hidden = hidden[0, seq_len[0]].detach().cpu().float().numpy()
            activations[layer_idx] = last_hidden
        return hook_fn

    # Register hooks on each transformer layer
    for idx, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(make_hook(idx))
        hooks.append(hook)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    return activations


def generate_with_activations(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text and extract activations at the last generated token.

    Returns: (generated_text, activations_dict)
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output[0]
    generated_text = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)

    # Now extract activations at the last generated token
    full_attention = torch.ones(1, generated_ids.shape[0], device=model.device, dtype=torch.long)
    activations = extract_activations(model, generated_ids.unsqueeze(0), full_attention)

    return generated_text, activations


def parse_gsm8k_answer(text):
    """Extract numeric answer from GSM8K format (#### N)."""
    # Look for #### pattern
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Fallback: look for last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def parse_ground_truth(answer_text):
    """Parse ground truth from GSM8K answer field."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def check_correct(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    if predicted is None or ground_truth is None:
        return False
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        return str(predicted).strip() == str(ground_truth).strip()


def load_gsm8k():
    """Load GSM8K test set."""
    dataset = load_from_disk(DATASET_PATH)
    test_data = dataset["test"]
    problems = []
    for i, item in enumerate(test_data):
        if i >= NUM_PROBLEMS:
            break
        problems.append({
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": parse_ground_truth(item["answer"]),
        })
    print(f"Loaded {len(problems)} GSM8K problems")
    return problems


def run_pipeline(model, tokenizer, problems, save_every=25):
    """Run the full self-critique pipeline on all problems.

    For each problem:
    1. Generate initial answer + extract activations
    2. Generate self-critique + extract activations
    3. Generate revised answer + extract activations
    4. Generate control answer (re-prompt without critique) + extract activations
    """
    results = []
    all_activations = {
        "initial": [],
        "critique": [],
        "revised": [],
        "control": [],
    }

    for i, problem in enumerate(problems):
        print(f"\n--- Problem {i+1}/{len(problems)} ---")
        question = problem["question"]
        ground_truth = problem["ground_truth"]

        # Phase 1: Initial answer
        prompt_initial = INITIAL_PROMPT_TEMPLATE.format(question=question)
        initial_text, initial_acts = generate_with_activations(model, tokenizer, prompt_initial)
        initial_answer = parse_gsm8k_answer(initial_text)
        initial_correct = check_correct(initial_answer, ground_truth)
        print(f"  Initial: {initial_answer} (GT: {ground_truth}) {'OK' if initial_correct else 'WRONG'}")

        # Phase 2: Self-critique
        prompt_critique = CRITIQUE_PROMPT_TEMPLATE.format(
            question=question, initial_answer=initial_text[:1000]
        )
        critique_text, critique_acts = generate_with_activations(model, tokenizer, prompt_critique)

        # Phase 3: Revised answer
        prompt_revised = REVISE_PROMPT_TEMPLATE.format(
            question=question,
            initial_answer=initial_text[:1000],
            critique=critique_text[:1000],
        )
        revised_text, revised_acts = generate_with_activations(model, tokenizer, prompt_revised)
        revised_answer = parse_gsm8k_answer(revised_text)
        revised_correct = check_correct(revised_answer, ground_truth)
        print(f"  Revised: {revised_answer} {'OK' if revised_correct else 'WRONG'}")

        # Phase 4: Control (re-prompt without critique)
        prompt_control = CONTROL_PROMPT_TEMPLATE.format(question=question)
        control_text, control_acts = generate_with_activations(model, tokenizer, prompt_control)
        control_answer = parse_gsm8k_answer(control_text)
        control_correct = check_correct(control_answer, ground_truth)
        print(f"  Control: {control_answer} {'OK' if control_correct else 'WRONG'}")

        # Store results
        result = {
            "idx": i,
            "question": question,
            "ground_truth": ground_truth,
            "initial_answer": initial_answer,
            "initial_correct": initial_correct,
            "initial_text": initial_text[:500],
            "critique_text": critique_text[:500],
            "revised_answer": revised_answer,
            "revised_correct": revised_correct,
            "revised_text": revised_text[:500],
            "control_answer": control_answer,
            "control_correct": control_correct,
        }
        results.append(result)

        # Store activations (convert to lists for each layer)
        for phase, acts in [("initial", initial_acts), ("critique", critique_acts),
                           ("revised", revised_acts), ("control", control_acts)]:
            # Store as stacked array: shape (num_layers, hidden_dim)
            layer_keys = sorted(acts.keys())
            stacked = np.stack([acts[k] for k in layer_keys])
            all_activations[phase].append(stacked)

        # Periodic save
        if (i + 1) % save_every == 0:
            save_results(results, all_activations, partial=True)
            gc.collect()
            torch.cuda.empty_cache()

    return results, all_activations


def save_results(results, activations, partial=False):
    """Save results and activations to disk."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_partial" if partial else ""

    # Save behavioral results as JSON
    with open(os.path.join(RESULTS_DIR, f"behavioral_results{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save activations as numpy arrays
    for phase in activations:
        if activations[phase]:
            arr = np.stack(activations[phase])  # (num_problems, num_layers, hidden_dim)
            np.save(os.path.join(RESULTS_DIR, f"activations_{phase}{suffix}.npy"), arr)

    print(f"  Saved {len(results)} results to {RESULTS_DIR}")


if __name__ == "__main__":
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, tokenizer = load_model_and_tokenizer()
    problems = load_gsm8k()
    results, activations = run_pipeline(model, tokenizer, problems)
    save_results(results, activations, partial=False)

    # Print summary
    n = len(results)
    initial_acc = sum(r["initial_correct"] for r in results) / n
    revised_acc = sum(r["revised_correct"] for r in results) / n
    control_acc = sum(r["control_correct"] for r in results) / n
    print(f"\n=== Summary ===")
    print(f"N = {n}")
    print(f"Initial accuracy: {initial_acc:.3f}")
    print(f"Revised accuracy: {revised_acc:.3f}")
    print(f"Control accuracy: {control_acc:.3f}")
