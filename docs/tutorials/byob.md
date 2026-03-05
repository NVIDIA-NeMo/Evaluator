# Write Your Own Benchmark (BYOB)

Create a custom benchmark in ~30 lines of Python.

## Overview

```{mermaid}
flowchart LR
    A[Your Dataset] --> B[EvalEnvironment]
    B --> C["seed(idx)"]
    B --> D["verify(response, expected)"]
    C --> E[Prompt + Expected Answer]
    D --> F[Reward + Scoring Details]
```

Every benchmark implements two methods:
- **`seed(idx)`** -- Returns a prompt and the expected answer for problem `idx`
- **`verify(response, expected)`** -- Scores the model's response and returns a reward

## Step 1: Create the environment file

Create `src/nemo_evaluator/benchmarks/my_reasoning.py`:

```python
from datasets import load_dataset

from nemo_evaluator.environments import EvalEnvironment, SeedResult, VerifyResult, register
from nemo_evaluator.scoring import extract_answer, math_equal


@register("my_reasoning")
class MyReasoningBenchmark(EvalEnvironment):
    def __init__(self):
        super().__init__()
        ds = load_dataset("your-org/reasoning-2026", split="test")
        self._data = [
            {
                "question": row["problem"],
                "answer": row["answer"],
                "category": row.get("domain", "general"),
                "difficulty": row.get("level", "medium"),
            }
            for row in ds
        ]

    def __len__(self):
        return len(self._data)

    def seed(self, idx: int) -> SeedResult:
        row = self._data[idx]
        prompt = f"Solve the following problem step by step.\n\n{row['question']}\n\nPut your final answer in \\boxed{{}}."
        return SeedResult(
            prompt=prompt,
            expected_answer=row["answer"],
            metadata={"category": row["category"], "difficulty": row["difficulty"]},
        )

    def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        extracted = extract_answer(response)
        correct = math_equal(extracted, expected)
        return VerifyResult(
            reward=1.0 if correct else 0.0,
            extracted_answer=extracted,
            scoring_details={"method": "math_equal", "symbolic_match": correct},
        )
```

## Step 2: Register the import

Add the import to `src/nemo_evaluator/benchmarks/__init__.py`:

```python
from nemo_evaluator.benchmarks.my_reasoning import MyReasoningBenchmark
```

## Step 3: Validate

```bash
nel validate --benchmark my_reasoning --samples 10
```

Expected output:

```
  7/10 correct
  [PASS] p0: expected='42' got='42' (1230ms 156tok)
  [PASS] p1: expected='7/3' got='7/3' (980ms 134tok)
  [FAIL] p2: expected='256' got='512' (1100ms 201tok)
  ...
```

Check that:
- At least some problems produce non-zero rewards (otherwise scoring may be broken)
- `seed()` returns valid prompts
- `verify()` returns parseable answers

## Step 4: Run full evaluation

```bash
nel run --benchmark my_reasoning --repeats 4 --output-dir ./results/my_reasoning
```

## Step 5: Serve for Gym training

```bash
nel serve --benchmark my_reasoning --gym-compat --port 9090
```

Now Gym training can point at `http://hostname:9090`.

## Scoring Primitives Reference

| Function | Use case | Import |
|----------|----------|--------|
| `math_equal(a, b)` | Symbolic math comparison via sympy | `from nemo_evaluator.scoring import math_equal` |
| `exact_match(a, b)` | Normalized string comparison | `from nemo_evaluator.scoring import exact_match` |
| `extract_answer(text)` | Extract from `\boxed{}`, "the answer is", or last number | `from nemo_evaluator.scoring import extract_answer` |

## Advanced: Category Metadata

If your `seed()` returns `metadata={"category": "algebra"}`, the evaluation automatically computes per-category breakdowns:

```json
{
  "categories": {
    "algebra": {"n_samples": 40, "mean_reward": 0.825},
    "geometry": {"n_samples": 30, "mean_reward": 0.633},
    "number_theory": {"n_samples": 30, "mean_reward": 0.700}
  }
}
```

## Advanced: Custom Scoring

For non-math benchmarks, implement your own scoring in `verify()`:

```python
def verify(self, response: str, expected: str, **meta) -> VerifyResult:
    # Multiple acceptable answers
    aliases = meta.get("_aliases", [expected])
    correct = any(exact_match(response.strip(), alias) for alias in aliases)

    # Partial credit
    reward = 1.0 if correct else 0.0

    return VerifyResult(
        reward=reward,
        extracted_answer=response.strip().split("\n")[0],
        scoring_details={"method": "exact_match_with_aliases", "n_aliases": len(aliases)},
    )
```
