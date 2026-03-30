# Implement Quality Gates

Use `nel gate` when you need a release decision across a benchmark suite rather than a diagnosis for one benchmark.

`nel gate` answers:

- did the candidate satisfy the release policy?
- which benchmarks passed, breached, or lacked enough evidence?
- should this checkpoint be treated as `GO`, `NO-GO`, or `INCONCLUSIVE`?

## What Problem This Solves

Real release processes do not depend on one benchmark.

You usually need rules like:

- GPQA and MMLU-Pro are critical
- HumanEval is supporting
- each benchmark has its own allowed drop
- low-baseline tasks need a relative-drop guardrail
- missing required benchmarks should fail the certification step

`nel gate` turns that policy into a YAML file and applies it across baseline and candidate result directories.

## What The Current Gate Supports

Today the CLI supports:

- aggregate verdicts: `GO`, `NO-GO`, `INCONCLUSIVE`
- tiers: `critical`, `supporting`, `advisory`
- metrics: `mean_reward`, `pass@1`
- directions: `higher_is_better`, `lower_is_better`
- absolute drop thresholds
- optional relative-drop guardrails
- JSON report output

The gate uses confidence-aware decisions when paired per-problem evidence exists and falls back to bundle score intervals when needed.

## Step 1: Prepare Baseline And Candidate Result Directories

The simplest layout is one benchmark subdirectory per result root:

```text
results/
  baseline/
    mmlu_pro/
      eval-mmlu-pro.json
      results.jsonl
    gpqa/
      eval-gpqa.json
      results.jsonl
  candidate/
    mmlu_pro/
      eval-mmlu-pro.json
      results.jsonl
    gpqa/
      eval-gpqa.json
      results.jsonl
```

For strong evidence, keep:

- the same benchmark set on both sides
- the same evaluation configuration
- `results.jsonl` files
- bundle score confidence intervals

## Step 2: Write A Policy File

Create `gate_policy.yaml`:

```yaml
version: 1
defaults:
  tier: supporting
  metric: mean_reward
  direction: higher_is_better
  max_drop: 0.015
benchmarks:
  mmlu_pro:
    tier: critical
    metric: mean_reward
    max_drop: 0.01
  gpqa:
    tier: critical
    metric: mean_reward
    max_drop: 0.01
  humaneval:
    tier: supporting
    metric: pass@1
    max_drop: 0.015
  aime_2025:
    tier: critical
    metric: mean_reward
    max_drop: 0.01
    max_relative_drop: 0.02
    relative_guard_below: 0.20
  triviaqa:
    tier: advisory
    metric: mean_reward
```

How to think about the fields:

- `tier`
  `critical` and `supporting` are required. `advisory` is reported but does not affect the final verdict.
- `metric`
  Must be explicit for required benchmarks. Today use `mean_reward` or `pass@1`.
- `direction`
  Use `higher_is_better` for accuracy-style metrics and `lower_is_better` for metrics like loss or perplexity.
- `max_drop`
  Maximum tolerated absolute damage.
- `max_relative_drop`
  Optional relative guardrail.
- `relative_guard_below`
  Only apply the relative guard when the baseline is below this score.

## Step 3: Run `nel gate`

Basic usage:

```bash
nel gate ./results/baseline ./results/candidate --policy gate_policy.yaml
```

Common production form:

```bash
nel gate ./results/baseline ./results/candidate \
  --policy gate_policy.yaml \
  --output ./artifacts/gate_report.json \
  --strict \
  --verbose
```

If you want machine-readable output on stdout:

```bash
nel gate ./results/baseline ./results/candidate \
  --policy gate_policy.yaml \
  --format json
```

## Step 4: Read The Verdict

The aggregate verdict is:

- `GO`
- `NO-GO`
- `INCONCLUSIVE`

Interpretation:

- `GO`
  All required benchmarks passed policy.
- `NO-GO`
  At least one required benchmark breached policy or was missing.
- `INCONCLUSIVE`
  No required benchmark breached, but at least one required benchmark did not have enough evidence to certify.

Per benchmark, NEL reports:

- `PASS`
- `BREACH`
- `INSUFFICIENT_EVIDENCE`
- `MISSING`

## Step 5: Use `--strict` In CI Or Release Scripts

With `--strict`, exit codes are:

- `0` for `GO`
- `1` for `NO-GO`
- `2` for `INCONCLUSIVE`

That makes the command usable directly in automation.

Example:

```bash
nel gate ./results/baseline ./results/candidate \
  --policy gate_policy.yaml \
  --strict
```

## Step 6: Investigate Failures With `nel compare`

The right workflow is:

1. Run the suite-level gate.
2. If the verdict is `GO`, continue.
3. If the verdict is `NO-GO` or `INCONCLUSIVE`, inspect the failing benchmark(s) with `nel compare`.

Example:

```bash
nel compare ./results/baseline/gpqa ./results/candidate/gpqa --show-flips --verbose
```

This is the important split:

- `nel gate` enforces policy
- `nel compare` explains what changed

## Troubleshooting

### “required benchmark must resolve to an explicit metric”

Add `metric` to the benchmark entry or to the policy defaults.

### “unsupported metric”

The current CLI gate supports `mean_reward` and `pass@1`.

### `MISSING`

NEL could not find the required benchmark on both sides. Check benchmark names and directory layout.

### `INSUFFICIENT_EVIDENCE`

This usually means:

- `results.jsonl` is missing or empty
- too few paired items were available
- the confidence interval is too wide for the selected threshold

### Duplicate benchmark error

The results root contains multiple bundles that resolve to the same benchmark name. Clean the directory or point the command at a narrower root.
