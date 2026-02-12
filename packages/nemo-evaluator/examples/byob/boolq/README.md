# BoolQ BYOB Example

This example demonstrates BYOB with the BoolQ yes/no question answering task from Google/SuperGLUE.

## What This Example Shows

- Multi-field prompt templates (passage + question)
- Custom answer extraction logic (yes/no from free-text response)
- Target field transformation (true/false to yes/no)
- `__file__`-relative dataset paths (works from any directory)

## How to Run

### 1. Compile the benchmark

```bash
cd packages/nemo-evaluator
nemo-evaluator-byob examples/byob/boolq/benchmark.py
```

### 2. Run the evaluation

```bash
export PYTHONPATH="~/.nemo-evaluator/byob_packages:$PYTHONPATH"
nemo-evaluator run_eval \
  --eval_type byob_boolq.boolq \
  --model_url http://localhost:8000 \
  --model_id my-model
```

### 3. Check results

Results are written to `byob_results.json` in the output directory.

## Dataset

`data.jsonl` contains 20 BoolQ samples with fields:
- `passage`: Context paragraph
- `question`: Yes/no question about the passage
- `answer`: Ground truth (`"true"` or `"false"`)

## Understanding the Code

```python
@benchmark(
    name="boolq",
    dataset=os.path.join(_SCRIPT_DIR, "data.jsonl"),  # __file__-relative path
    prompt="Read the passage and answer with 'yes' or 'no'.\n\n"
           "Passage: {passage}\n\nQuestion: {question}\n\nAnswer:",
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def boolq_scorer(response, target, metadata):
    # Extract yes/no from model response
    # Compare against ground truth (true->yes, false->no)
    return {"correct": predicted == target_str}
```

## Expected Results

With a capable model, expect ~70-90% accuracy on this 20-sample subset.
