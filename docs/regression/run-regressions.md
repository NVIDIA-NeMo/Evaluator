# Run Regressions

Use `nel compare` when you already have a baseline run and a candidate run for the same benchmark and you want to answer three questions:

1. Did the candidate score go down?
2. Which exact problems flipped from correct to incorrect?
3. Is the observed drop big enough, and stable enough, to worry about?

This command is an investigation tool. It is best for diagnosing one benchmark at a time. If you need a release decision across multiple benchmarks, use the quality gate described in [quality-gate.md](quality-gate.md).

## The Problem This Solves

A single headline score can hide the actual failure mode.

- A model can lose 1 point overall because a few hard examples moved around.
- A model can also keep the same overall score while regressing on a small set of important examples and improving on others.
- Small benchmarks are noisy, so a raw score delta alone does not tell you whether you saw a real regression or normal variation.

`nel compare` compares the same problems across two runs and reports:

- score deltas
- per-problem regressions and improvements
- McNemar significance testing on paired flips
- an auto-generated Markdown investigation report

## Before You Start

You need two evaluation outputs produced by `nel eval run`.

- A baseline run
- A candidate run

Each path can be either:

- a directory containing exactly one `eval-*.json` bundle
- the `eval-*.json` bundle itself

For the most useful results, keep the runs comparable:

- same benchmark
- same prompt/configuration shape
- same dataset slice or `--max-problems`
- same repeat count unless you have a deliberate reason to change it

Install the optional stats extra if you want the full statistical analysis:

```bash
pip install -e ".[stats]"
```

Without `scipy`, NEL still compares the runs, but the statistical verdict becomes less informative.

## Step 1: Produce Baseline And Candidate Runs

Example:

```bash
nel eval run --bench mmlu_pro \
  --model-url https://integrate.api.nvidia.com/v1 \
  --model-id baseline-model \
  --repeats 1 \
  --max-problems 200 \
  --output-dir ./results/baseline

nel eval run --bench mmlu_pro \
  --model-url https://integrate.api.nvidia.com/v1 \
  --model-id candidate-model \
  --repeats 1 \
  --max-problems 200 \
  --output-dir ./results/candidate
```

After the runs finish, each directory should contain at least:

- `eval-*.json`
- `results.jsonl`

## Step 2: Run The Comparison

The simplest form is:

```bash
nel compare ./results/baseline ./results/candidate
```

NEL resolves the bundle automatically when the directory contains exactly one `eval-*.json`.

Useful variants:

```bash
nel compare ./results/baseline ./results/candidate --show-flips
nel compare ./results/baseline ./results/candidate --verbose
nel compare ./results/baseline ./results/candidate --compact
nel compare ./results/baseline ./results/candidate --format json
nel compare ./results/baseline ./results/candidate --output regression.json
nel compare ./results/baseline ./results/candidate --report regression_report.md
nel compare ./results/baseline ./results/candidate --no-report
```

## Step 3: Tune The Comparison For Your Benchmark

The most important flags are:

- `--max-drop 0.01`
  Use this when a 1 percentage point regression is already unacceptable.
- `--correct-above 0.5`
  Use this for judge-style or graded rewards where “correct” means “reward above 0.5”, not just “reward > 0”.
- `--strict`
  Use this in CI when you want non-zero exit codes.

Example for a stricter comparison:

```bash
nel compare ./results/baseline ./results/candidate \
  --max-drop 0.01 \
  --correct-above 0.5 \
  --strict
```

## Step 4: Read The Verdict

`nel compare` produces one of these verdicts:

- `PASS`: no evidence of a meaningful regression
- `WARN`: statistically significant change, but still within the allowed practical threshold
- `BLOCK`: statistically significant regression beyond the allowed threshold
- `INCONCLUSIVE`: not enough evidence to confidently rule out a meaningful regression

With `--strict`, exit codes are:

- `0` for `PASS`
- `1` for `BLOCK`
- `2` for `WARN` or `INCONCLUSIVE`

## Step 5: Inspect The Flips

If the run looks suspicious, re-run with:

```bash
nel compare ./results/baseline ./results/candidate --show-flips --verbose
```

That gives you the exact `problem_idx` values that regressed or improved, plus richer statistical detail.

NEL also writes a Markdown report next to the candidate bundle by default:

- `regression_report.md`

This is the best artifact for manual review because it collects:

- the verdict
- score deltas
- flip summaries
- category breakdowns
- problem-level examples

## Common Workflow

For a normal model change, the workflow is:

1. Run the baseline evaluation.
2. Run the candidate evaluation.
3. Compare them with `nel compare`.
4. If the verdict is `PASS`, move on.
5. If the verdict is `WARN` or `INCONCLUSIVE`, inspect flips and decide whether you need more samples.
6. If the verdict is `BLOCK`, inspect the regressed problems and fix the candidate or escalate the regression.

## Troubleshooting

### “No eval-*.json bundle found”

Point `nel compare` at the directory created by `nel eval run`, or pass the bundle path directly.

### “Directory contains multiple eval-*.json bundles”

Pass the exact bundle you want to compare instead of the parent directory.

### `INCONCLUSIVE` with very small runs

Your benchmark may not have enough paired samples to detect the effect size you care about. Re-run with more problems or a larger benchmark.

### Missing or empty `results.jsonl`

NEL can still compare aggregate score deltas, but it cannot do paired flip analysis. You should treat that as weaker evidence.

## Python API

If you want to embed the same workflow in code:

```python
from nemo_evaluator.engine import compare_runs

report = compare_runs(
    "results/baseline/eval-baseline.json",
    "results/candidate/eval-candidate.json",
    reward_threshold=0.0,
    min_effect=0.01,
)

print(report["verdict"])
print(report["verdict_reasons"])
```

Use this API when you want to build your own reporting or CI wrapper around the single-benchmark regression workflow.
