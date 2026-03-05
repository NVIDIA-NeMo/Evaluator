# Regression Testing in CI

Automatically detect score regressions in merge requests.

## Overview

```{mermaid}
flowchart LR
    MR["Merge Request"] --> B["eval:baseline<br/>(target branch)"]
    MR --> C["eval:candidate<br/>(MR branch)"]
    B --> R["regression:check"]
    C --> R
    R -->|"delta > 5%"| FAIL["❌ Block merge"]
    R -->|"delta ≤ 5%"| PASS["✅ Pass"]
```

## GitLab CI Setup

Add to your repo's `.gitlab-ci.yml`:

```yaml
include:
  - local: deploy/gitlab-ci-eval.yml
```

Or copy the template:

```yaml
stages:
  - eval
  - regression

variables:
  BENCHMARK: gsm8k
  MODEL_URL: https://inference-api.nvidia.com/v1
  MODEL_ID: azure/openai/gpt-5.2
  REPEATS: "2"
  MAX_PROBLEMS: "50"
  NEL_IMAGE: nemo-evaluator:latest

eval:candidate:
  stage: eval
  image: $NEL_IMAGE
  script:
    - nel run --benchmark $BENCHMARK
        --model-url $MODEL_URL --model-id $MODEL_ID
        --repeats $REPEATS --max-problems $MAX_PROBLEMS
        --output-dir ./results/candidate --no-progress
  artifacts:
    paths: [results/candidate/]
    expire_in: 30 days
  rules:
    - if: $CI_MERGE_REQUEST_IID

eval:baseline:
  stage: eval
  image: $NEL_IMAGE
  script:
    - git checkout $CI_MERGE_REQUEST_TARGET_BRANCH_NAME
    - pip install -e ".[scoring]"
    - nel run --benchmark $BENCHMARK
        --model-url $MODEL_URL --model-id $MODEL_ID
        --repeats $REPEATS --max-problems $MAX_PROBLEMS
        --output-dir ./results/baseline --no-progress
  artifacts:
    paths: [results/baseline/]
    expire_in: 30 days
  rules:
    - if: $CI_MERGE_REQUEST_IID

regression:check:
  stage: regression
  image: $NEL_IMAGE
  needs: [eval:candidate, eval:baseline]
  script:
    - nel regression results/baseline/eval-*.json results/candidate/eval-*.json
        --output results/regression.json --threshold 0.05 --strict
  artifacts:
    paths: [results/regression.json]
    expire_in: 30 days
  rules:
    - if: $CI_MERGE_REQUEST_IID
```

## CLI Usage

### Compare two bundles

```bash
nel regression ./baseline/eval-*.json ./candidate/eval-*.json
```

### Output

```
Baseline:  eval-20260224T100000Z-gsm8k
Candidate: eval-20260225T143012Z-gsm8k

  pass@1: 0.7200 -> 0.7500  (delta=+0.0300, +4.2%, CI overlap)
  pass@4: 0.8800 -> 0.9100  (delta=+0.0300, +3.4%, CI overlap)

  tokens_per_second: 450 -> 520  (delta=+70.00)

Per-category deltas:
  algebra: 0.8000 -> 0.8500  (delta=+0.0500)
  geometry: 0.6000 -> 0.5800  (delta=-0.0200)

No regressions beyond 5% threshold.
```

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--threshold` | 0.05 | Maximum acceptable score drop |
| `--strict` | off | Exit non-zero if regression detected |
| `--output` | none | Write JSON report to file |

### Python API

```python
from nemo_evaluator.runner.regression import compare_runs, write_regression

report = compare_runs("baseline/eval-*.json", "candidate/eval-*.json")

for metric, delta in report["score_deltas"].items():
    print(f"{metric}: {delta['delta']:+.4f}")

write_regression(report, "regression.json")
```
