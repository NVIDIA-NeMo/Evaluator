# Scoring & Metrics

## Scoring Primitives

### `math_equal(response, expected)`

Symbolic math comparison. Tries numeric comparison first, falls back to sympy.

```python
from nemo_evaluator.scoring import math_equal

math_equal("72", "72.0")           # True
math_equal("\\frac{1}{2}", "0.5")  # True (sympy)
math_equal("3*x + 1", "1 + 3x")   # True (sympy)
math_equal("42", "43")             # False
```

Requires `pip install -e ".[scoring]"` for sympy.

### `exact_match(response, expected)`

Normalized string comparison (lowercase, strip whitespace and punctuation).

```python
from nemo_evaluator.scoring import exact_match

exact_match("Paris", "paris")    # True
exact_match(" Paris ", "Paris")  # True
exact_match("Paris.", "Paris")   # True
exact_match("London", "Paris")   # False
```

### `extract_answer(text)`

Extracts the final answer from model output. Tries in order:

1. `\boxed{answer}` -- LaTeX boxed format
2. `The answer is answer` -- natural language
3. Last standalone number on the last line

```python
from nemo_evaluator.scoring import extract_answer

extract_answer("The calculation gives \\boxed{42}.")  # "42"
extract_answer("Therefore, the answer is 42.")        # "42"
extract_answer("Step 1: 20+22=42\n42")                # "42"
```

## Metrics

### pass@k

The standard Codex-style pass@k metric. Given `n` attempts per problem with `c` correct:

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

```python
from nemo_evaluator.metrics import pass_at_k

pass_at_k(n=8, c=3, k=1)   # Probability at least 1 of 1 sample is correct
pass_at_k(n=8, c=3, k=4)   # Probability at least 1 of 4 samples is correct
```

### Bootstrap Confidence Intervals

95% CI via bootstrap resampling (10,000 iterations):

```python
from nemo_evaluator.metrics import bootstrap_ci

scores = [pass_at_k(n, c, 1) for n, c in problem_results]
ci = bootstrap_ci(scores)
print(f"pass@1: {ci.value:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
```

### Category Breakdown

When problems have category metadata, per-category accuracy is computed automatically:

```python
from nemo_evaluator.metrics.aggregation import category_breakdown

# results is a list of dicts with metadata.category
cats = category_breakdown(results, "category")
for c in cats:
    print(f"{c.category}: {c.mean_reward:.3f} ({c.n_samples} samples)")
```

## Failure Analysis

The `ArtifactCollector` automatically categorizes failures:

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
