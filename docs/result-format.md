# Evaluation Result Format

This document describes the data formats consumed by `nel compare` and `nel gate`.

## results.jsonl — Per-Problem Records

Each line is a JSON object representing one evaluation problem (or one repeat of a problem).

### Required fields

| Field | Type | Description |
|---|---|---|
| `problem_idx` | int | Unique problem identifier. Same problem across runs must have the same `problem_idx`. |
| `reward` | float | Score for this problem. Typically 0.0 (incorrect) or 1.0 (correct), but can be any float for partial credit or judge scores. |

### Optional fields

| Field | Type | Description |
|---|---|---|
| `repeat` | int | Repeat index (default: 0). When N>1 repeats are used, each repeat of the same `problem_idx` gets a different `repeat` value (0, 1, 2, ...). Repeats are averaged per problem before analysis. |
| `expected_answer` | string | Ground truth answer. Shown in flip analysis reports. |
| `model_response` | string | Model's raw response. Shown truncated (80 chars) in flip reports. |
| `extracted_answer` | string | Parsed/extracted answer from model response. Used if `model_response` is absent. |
| `metadata.category` | string | Category label (e.g., "algebra", "geometry"). Enables per-category breakdown in regression reports and clustered confidence intervals. |
| `scoring_details` | object | Arbitrary scoring metadata. `scoring_details.category` is used as fallback when `metadata.category` is absent. |

### Example

```jsonl
{"problem_idx": 0, "reward": 1.0, "repeat": 0, "expected_answer": "42", "model_response": "The answer is 42.", "metadata": {"category": "arithmetic"}}
{"problem_idx": 0, "reward": 1.0, "repeat": 1, "expected_answer": "42", "model_response": "42", "metadata": {"category": "arithmetic"}}
{"problem_idx": 0, "reward": 0.0, "repeat": 2, "expected_answer": "42", "model_response": "I think it's 41.", "metadata": {"category": "arithmetic"}}
{"problem_idx": 1, "reward": 0.0, "repeat": 0, "expected_answer": "Paris", "model_response": "London", "metadata": {"category": "geography"}}
{"problem_idx": 1, "reward": 0.0, "repeat": 1, "expected_answer": "Paris", "model_response": "Berlin", "metadata": {"category": "geography"}}
{"problem_idx": 1, "reward": 1.0, "repeat": 2, "expected_answer": "Paris", "model_response": "Paris", "metadata": {"category": "geography"}}
```

In this example, problem 0 has 3 repeats with rewards [1.0, 1.0, 0.0] → mean reward 0.67. Problem 1 has rewards [0.0, 0.0, 1.0] → mean reward 0.33.

## eval-*.json — Evaluation Bundle

Each bundle is a JSON file describing the results of one benchmark evaluation. `nel gate` discovers bundles by glob pattern `eval-*.json` in a directory.

### Required fields

| Path | Type | Description |
|---|---|---|
| `benchmark.name` | string | Benchmark identifier. Must match the benchmark name in the gate policy YAML. |
| `benchmark.scores` | object | Metric scores. Each key is a metric name (e.g., `mean_reward`, `pass@1`). |
| `benchmark.scores.<metric>.value` | float | The aggregate score for this metric. |

### Optional fields

| Path | Type | Description |
|---|---|---|
| `run_id` | string | Unique identifier for this evaluation run. |
| `config` | object | Evaluation configuration metadata. |
| `benchmark.scores.<metric>.ci_lower` | float | Lower bound of 95% confidence interval. |
| `benchmark.scores.<metric>.ci_upper` | float | Upper bound of 95% confidence interval. |
| `benchmark.categories` | object | Per-category scores. Each key is a category name, value has `mean_reward` and `n`. |

### Example

```json
{
  "run_id": "eval-20260402T120000Z-nemotron-super-bf16",
  "config": {"benchmark": "mmlu_pro", "model": "nemotron-super-120b"},
  "benchmark": {
    "name": "mmlu_pro",
    "samples": 12000,
    "scores": {
      "mean_reward": {
        "value": 0.782,
        "ci_lower": 0.774,
        "ci_upper": 0.790
      }
    },
    "categories": {
      "algebra": {"mean_reward": 0.85, "n": 500},
      "geometry": {"mean_reward": 0.72, "n": 400}
    }
  }
}
```

## Directory Layout

`nel gate` supports two layouts:

### Flat (bundles in root)

```text
results/
  eval-mmlu_pro.json
  results.jsonl           # per-problem records for the single benchmark
```

### Nested (one subdirectory per benchmark)

```text
results/
  mmlu_pro/
    eval-mmlu_pro.json
    results.jsonl
  gpqa/
    eval-gpqa.json
    results.jsonl
```

`nel compare` accepts either a single `eval-*.json` file or a directory containing one.
