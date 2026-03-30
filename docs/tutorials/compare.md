# Compare Runs

Use `nel compare` when you want to answer a focused question:

- did this benchmark get worse?
- which exact problems regressed?
- is the observed change likely to be real or just noise?

This command is for investigation. It compares one benchmark run against another and gives you both score-level and problem-level evidence.

If you need one release decision across many benchmarks, use [`quality-gate.md`](quality-gate.md) instead.

## What Problem This Solves

A single average score is often not enough.

- A candidate can lose a small amount overall but fail on an important subset of problems.
- A candidate can keep the same headline score while swapping improvements on some questions for regressions on others.
- Small benchmarks are noisy, so a raw delta does not tell you whether the change is meaningful.

`nel compare` keeps the comparison paired at the problem level and reports:

- score deltas
- per-problem regressions and improvements
- paired flip statistics
- a Markdown investigation report

## Step 1: Run A Baseline And A Candidate

Start with two evaluation outputs produced by `nel eval run`.

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

For the best comparison, keep the runs matched on:

- benchmark
- dataset slice
- prompt / config
- repeat count

## Step 2: Run `nel compare`

Compare the two result directories:

```bash
nel compare ./results/baseline ./results/candidate
```

You can also point directly at the bundle files:

```bash
nel compare \
  ./results/baseline/eval-20260330-mmlu-pro.json \
  ./results/candidate/eval-20260331-mmlu-pro.json
```

## Step 3: Use The Most Important Flags

The most useful flags are:

- `--strict`
  Exit non-zero when the comparison blocks or is inconclusive enough to matter in CI.
- `--max-drop 0.01`
  Tighten the practical effect threshold when 1 percentage point matters.
- `--correct-above 0.5`
  Use this for judge-style rewards where “correct” means a score above a threshold, not just `reward > 0`.
- `--show-flips`
  Print the exact regressed and improved problems.
- `--verbose`
  Show the statistical details behind the verdict.
- `--output report.json`
  Save the structured comparison JSON.

Example:

```bash
nel compare ./results/baseline ./results/candidate \
  --max-drop 0.01 \
  --correct-above 0.5 \
  --show-flips \
  --verbose \
  --strict
```

## Step 4: Read The Verdict

`nel compare` returns one of these verdicts:

- `PASS`
- `WARN`
- `BLOCK`
- `INCONCLUSIVE`

How to interpret them:

- `PASS`
  No evidence of a practically meaningful regression.
- `WARN`
  The change is real enough to notice statistically, but still within the configured tolerance.
- `BLOCK`
  The candidate regressed beyond the allowed threshold.
- `INCONCLUSIVE`
  There is not enough evidence to confidently rule out a meaningful regression.

With `--strict`, exit codes are:

- `0` for `PASS`
- `1` for `BLOCK`
- `2` for `WARN` and `INCONCLUSIVE`

## Step 5: Inspect The Investigation Report

By default, NEL writes a Markdown report next to the candidate bundle:

- `regression_report.md`

That report is useful when you need to answer:

- which problems flipped?
- which categories were hit hardest?
- what did the baseline answer vs the candidate?

If you do not want the Markdown artifact:

```bash
nel compare ./results/baseline ./results/candidate --no-report
```

If you want it in a specific path:

```bash
nel compare ./results/baseline ./results/candidate --report ./artifacts/mmlu_compare.md
```

## Step 6: Decide What To Do Next

A practical workflow is:

1. Run the baseline.
2. Run the candidate.
3. Run `nel compare`.
4. If the verdict is `PASS`, continue.
5. If the verdict is `WARN` or `INCONCLUSIVE`, inspect flips and decide whether you need more samples.
6. If the verdict is `BLOCK`, inspect the regressed problems and fix the candidate or reject it.

## Troubleshooting

### The command says there are multiple bundles in a directory

Point `nel compare` at the exact `eval-*.json` file instead of the parent directory.

### The result is `INCONCLUSIVE`

You may not have enough paired samples to detect the effect size you care about. Re-run with more problems or a larger benchmark.

### `results.jsonl` is missing or empty

NEL can still compare aggregate score deltas, but the paired flip analysis becomes weaker.

### I want a release decision across several benchmarks

Use [`quality-gate.md`](quality-gate.md). `nel compare` is the diagnostic tool, not the suite-level policy gate.
