# CI Regression Gate

Block merge requests that cause evaluation regressions.

## GitLab CI

Include the eval template in your `.gitlab-ci.yml`:

```yaml
include:
  - local: deploy/gitlab-ci-eval.yml
```

Or add the stages directly:

```yaml
stages:
  - eval
  - regression

eval:candidate:
  stage: eval
  image: nemo-evaluator:latest
  script:
    - nel eval run --bench gsm8k --repeats 2 --max-problems 50 -o results/candidate --no-progress
  artifacts:
    paths: [results/candidate/]
  rules:
    - if: $CI_MERGE_REQUEST_IID

eval:baseline:
  stage: eval
  image: nemo-evaluator:latest
  script:
    - git checkout $CI_MERGE_REQUEST_TARGET_BRANCH_NAME
    - pip install -e ".[scoring]"
    - nel eval run --bench gsm8k --repeats 2 --max-problems 50 -o results/baseline --no-progress
  artifacts:
    paths: [results/baseline/]
  rules:
    - if: $CI_MERGE_REQUEST_IID

regression:check:
  stage: regression
  image: nemo-evaluator:latest
  needs: [eval:candidate, eval:baseline]
  script:
    - nel regression results/baseline/eval-*.json results/candidate/eval-*.json --threshold 0.05 --strict
  artifacts:
    paths: [results/regression.json]
  rules:
    - if: $CI_MERGE_REQUEST_IID
```

## How it works

```{mermaid}
sequenceDiagram
    participant MR as Merge Request
    participant B as eval:baseline
    participant C as eval:candidate
    participant R as regression:check

    MR->>B: Trigger (target branch)
    MR->>C: Trigger (MR branch)
    B-->>R: results/baseline/eval-*.json
    C-->>R: results/candidate/eval-*.json
    R->>R: compare_runs()
    alt delta > threshold
        R-->>MR: ❌ Pipeline failed
    else delta ≤ threshold
        R-->>MR: ✅ Pipeline passed
    end
```

## Statistical significance

With `scipy` installed (`pip install nemo-evaluator[stats]`), regression reports include Mann-Whitney U p-values for each score delta. This distinguishes meaningful regressions from noise.

## Threshold tuning

| Scenario | Threshold | Repeats | Max problems |
|----------|-----------|---------|-------------|
| Quick smoke test | 0.10 | 1 | 50 |
| Standard gate | 0.05 | 2 | 100 |
| High-confidence | 0.03 | 4 | full dataset |

Higher repeats reduce noise in pass@k estimation. More problems reduce sampling variance. P-values require at least 2 samples per run.

## GitHub Actions

```yaml
jobs:
  eval-baseline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { ref: main }
      - run: pip install -e ".[scoring]"
      - run: nel eval run --bench gsm8k --repeats 2 --max-problems 50 -o results/baseline --no-progress
      - uses: actions/upload-artifact@v4
        with: { name: baseline, path: results/baseline/ }

  eval-candidate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[scoring]"
      - run: nel eval run --bench gsm8k --repeats 2 --max-problems 50 -o results/candidate --no-progress
      - uses: actions/upload-artifact@v4
        with: { name: candidate, path: results/candidate/ }

  regression:
    needs: [eval-baseline, eval-candidate]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - run: pip install -e ".[scoring]"
      - run: nel regression baseline/eval-*.json candidate/eval-*.json --strict
```
