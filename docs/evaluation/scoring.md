# Scoring

## Overview

Each benchmark defines its own scorer via the `@scorer` decorator. The `scoring/` package provides reusable scorer implementations:

| Module | Purpose |
|--------|---------|
| `scoring/text.py` | `exact_match`, `fuzzy_match` -- text comparison |
| `scoring/pattern.py` | `multichoice_regex`, `answer_line`, `numeric_match` -- pattern extraction |
| `scoring/code_execution.py` | `code_sandbox` -- Docker-sandboxed code execution |
| `scoring/judge.py` | `needs_judge`, `judge_score` -- LLM-as-judge pipeline |
| `scoring/json_schema.py` | `extract_json`, `validate_json_schema` -- structured output validation |
| `scoring/types.py` | `ScorerInput` -- input dataclass for all scorers |

## Scoring Primitives

These functions are the building blocks for `@scorer` functions. All are importable from the top-level package.

### `exact_match(sample)`

Normalized string comparison: lowercase, strip whitespace and punctuation, collapse articles.

```python
from nemo_evaluator import exact_match, ScorerInput

s = ScorerInput(response="  Paris.  ", target="paris")
exact_match(s)  # {"correct": True}
```

### `multichoice_regex(sample)`

Extracts a letter (A-D by default) from "Answer: X" patterns.

```python
from nemo_evaluator import multichoice_regex, ScorerInput

s = ScorerInput(response="The answer is B because...\nAnswer: B", target="B")
multichoice_regex(s)  # {"correct": True, "extracted": "B"}

# Custom pattern for 10-choice (A-J):
multichoice_regex(s, pattern=r"(?i)Answer\s*:\s*([A-J])")
```

### `answer_line(sample)`

Extracts the text after "Answer:" and compares with math normalization.

```python
from nemo_evaluator import answer_line, ScorerInput

s = ScorerInput(response="Step 1: ...\nAnswer: 42", target="42")
answer_line(s)  # {"correct": True, "extracted": "42"}
```

### `numeric_match(sample)`

Extracts the last number in the response.

```python
from nemo_evaluator import numeric_match, ScorerInput

s = ScorerInput(response="The total is 20 + 22 = 42", target="42")
numeric_match(s)  # {"correct": True, "extracted": "42"}
```

### `fuzzy_match(sample)`

Normalized substring containment. Supports multiple correct answers via `metadata["correct_answers"]`.

```python
from nemo_evaluator import fuzzy_match, ScorerInput

s = ScorerInput(response="The capital is Canberra.", target="Canberra",
                metadata={"correct_answers": ["Canberra", "canberra"]})
fuzzy_match(s)  # {"correct": True, "extracted": "The capital is Canberra."}
```

### `code_sandbox(sample)`

Runs code in a Docker container with network isolation, memory limits, and timeouts. Extracts code from markdown fences, concatenates with prompt code and test harness, and checks the exit code.

```python
from nemo_evaluator import code_sandbox, ScorerInput

s = ScorerInput(
    response="```python\ndef add(a, b):\n    return a + b\n```",
    target="add",
    metadata={"_prompt": "def add(a, b):\n", "_test": "assert add(1, 2) == 3",
              "entry_point": "add"},
)
code_sandbox(s)  # {"correct": True, "extracted": "def add(a, b):\n    return a + b"}
```

Requires Docker daemon access.

### `needs_judge(sample)`

Signals that this sample requires LLM-as-judge scoring. Returns `{"correct": False, "needs_judge": True}` so the eval loop's judge post-processor handles it.

Used by SimpleQA and HealthBench.

## LLM-as-Judge Pipeline

Benchmarks that use `needs_judge()` are scored in a post-processing step by `scoring/judge.py`. The judge pipeline:

1. Collects all samples flagged with `needs_judge: True`
2. Constructs judge prompts from the response and expected answer
3. Calls the judge model (configured via `--judge-url` / `JudgeScoringConfig`)
4. Parses the judge verdict and updates rewards

Configure the judge model:

```bash
nel eval run --bench simpleqa \
  --model-url https://api.example.com/v1 --model-id my-model \
  --judge-url https://api.example.com/v1 --judge-id gpt-4o
```

## JSON Schema Scoring

`scoring/json_schema.py` validates structured model outputs against a JSON schema:

```python
from nemo_evaluator.scoring import validate_json_schema

result = validate_json_schema(response_text, schema={"type": "object", "required": ["answer"]})
# {"valid": True, "extracted": {"answer": "42"}, "score": 1.0}
```

## Metrics

### pass@k

Standard Codex-style pass@k. Given `n` attempts per problem with `c` correct:

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

```python
from nemo_evaluator.metrics import pass_at_k

pass_at_k(n=8, c=3, k=1)   # probability at least 1/1 is correct
pass_at_k(n=8, c=3, k=4)   # probability at least 1/4 is correct
```

### Bootstrap Confidence Intervals

95% CI via bootstrap resampling (10,000 iterations):

```python
from nemo_evaluator.metrics import bootstrap_ci

ci = bootstrap_ci(scores)
print(f"pass@1: {ci.value:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
```

### Category Breakdown

When problems include category metadata, per-category accuracy is computed automatically:

```python
from nemo_evaluator.metrics.aggregation import category_breakdown

cats = category_breakdown(results, "category")
for c in cats:
    print(f"{c.category}: {c.mean_reward:.3f} ({c.n_samples} samples)")
```

## Failure Analysis

The `ArtifactCollector` categorizes failures automatically:

| Category | Detection |
|----------|-----------|
| `timeout` | Model error contains "timeout" |
| `rate_limit` | Model error contains "429" or "rate" |
| `refusal` | Response contains "I cannot", "I'm sorry" |
| `empty` | Empty response or whitespace-only |
| `format_error` | Non-empty response but no answer extracted |

Output in `failure_analysis.json`:

```json
{
  "total_failures": 12,
  "failure_rate": 0.06,
  "categories": {
    "refusal": {"count": 5, "rate": 0.025},
    "format_error": {"count": 4, "rate": 0.02},
    "timeout": {"count": 3, "rate": 0.015}
  },
  "exemplars": [
    {"category": "refusal", "problem_idx": 42, "response_preview": "I'm sorry, I cannot..."}
  ]
}
```
