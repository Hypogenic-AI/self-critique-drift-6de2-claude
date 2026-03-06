# Downloaded Datasets

This directory contains datasets for studying representation drift under self-reflection. Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: GSM8K

### Overview
- **Source**: HuggingFace `openai/gsm8k` (config: "main")
- **Size**: Train: 7,473 / Test: 1,319 examples
- **Format**: HuggingFace Dataset
- **Task**: Grade school math word problems with step-by-step solutions
- **License**: MIT
- **Why relevant**: Standard benchmark for testing self-correction in reasoning. Used by Huang et al. (2023) to show self-correction degrades performance.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k/data")
```

### Fields
- `question`: Math word problem text
- `answer`: Step-by-step solution ending with `#### <final_answer>`

---

## Dataset 2: MATH-500

### Overview
- **Source**: HuggingFace `HuggingFaceH4/MATH-500`
- **Size**: Test: 500 examples (curated subset of Hendrycks MATH)
- **Format**: HuggingFace Dataset
- **Task**: Competition-level math problems with difficulty levels 1-5
- **License**: MIT
- **Why relevant**: Provides difficulty stratification for analyzing representation drift. Used by David (2025) for temporal correctness probing.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/MATH-500")
dataset.save_to_disk("datasets/math/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/math/data")
```

### Fields
- `problem`: Math problem text
- `solution`: Step-by-step solution
- `answer`: Final answer
- `subject`: Math subject area
- `level`: Difficulty level (1-5)
- `unique_id`: Unique identifier

---

## Dataset 3: TruthfulQA

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa` (config: "generation")
- **Size**: Validation: 817 examples
- **Format**: HuggingFace Dataset
- **Task**: Questions designed to test model truthfulness vs. common misconceptions
- **License**: Apache 2.0
- **Why relevant**: Tests whether self-critique can improve truthfulness. Provides ground truth for probing honesty-related representations.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset.save_to_disk("datasets/truthfulqa/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthfulqa/data")
```

### Fields
- `question`: Question text
- `best_answer`: Best correct answer
- `correct_answers`: List of acceptable correct answers
- `incorrect_answers`: List of incorrect answers
- `category`: Topic category
- `source`: Source of the question
