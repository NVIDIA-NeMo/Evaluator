# Implement A Quality Gate

Use the quality gate when you need one release decision across multiple benchmarks, each with its own threshold and importance tier.

The quality gate answers a different question than `nel compare`:

- `nel compare` asks: did this one benchmark get worse, and where?
- the quality gate asks: do this candidate's benchmark results satisfy our release policy?

Today the quality gate is a Python API, not a CLI command. You use:

- `nemo_evaluator.config.load_gate_policy`
- `nemo_evaluator.engine.gate_runs`
- `nemo_evaluator.engine.write_gate_report`

## The Problem This Solves

A release process usually does not depend on one benchmark.

You often need rules like:

- GPQA and MMLU-Pro are critical
- HumanEval is supporting
- a 1.0 pp drop is acceptable on one task but not another
- low-baseline tasks need a relative-drop guardrail
- missing evidence should block certification, not silently pass

The quality gate turns those rules into a machine-readable policy and evaluates the baseline and candidate result directories against it.

## What The Current Gate Supports

The current implementation is intentionally narrow and deterministic.

- Aggregate verdicts: `GO`, `NO-GO`, `INCONCLUSIVE`
- Benchmark tiers: `critical`, `supporting`, `advisory`
- Supported gated metrics: `mean_reward`, `pass@1`
- Metric directions: `higher_is_better`, `lower_is_better`
- Relative-drop guardrails for low-baseline tasks
- Paired per-problem CI-based decisions when `results.jsonl` is available
- Conservative fallback to bundle score intervals when paired records are missing

Important limitation:

- there is no `nel gate` CLI yet

## Step 1: Organize The Baseline And Candidate Results

The gate expects two directories: one for baseline, one for candidate.

The easiest layout is one subdirectory per benchmark:

```text
results/
  baseline/
    mmlu_pro/
      eval-mmlu-pro.json
      results.jsonl
    gpqa/
      eval-gpqa.json
      results.jsonl
    humaneval/
      eval-humaneval.json
      results.jsonl
  candidate/
    mmlu_pro/
      eval-mmlu-pro.json
      results.jsonl
    gpqa/
      eval-gpqa.json
      results.jsonl
    humaneval/
      eval-humaneval.json
      results.jsonl
```

NEL also supports a flat layout with `eval-*.json` files directly in the root, but the nested layout is easier to reason about.

For the strongest evidence:

- use the same benchmark set on both sides
- use the same evaluation settings
- keep `results.jsonl` files
- keep bundle score confidence intervals

## Step 2: Write A Gate Policy

Create a YAML file such as `gate_policy.yaml`:

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
  aime_2025:
    tier: critical
    metric: mean_reward
    max_drop: 0.01
    max_relative_drop: 0.02
    relative_guard_below: 0.20
  humaneval:
    tier: supporting
    metric: pass@1
    max_drop: 0.015
  triviaqa:
    tier: advisory
    metric: mean_reward
```

How to think about the fields:

- `tier`
  `critical` and `supporting` are required and can block the release. `advisory` is reported but does not change the aggregate verdict.
- `metric`
  Must be explicit for required benchmarks. Today use `mean_reward` or `pass@1`.
- `direction`
  Leave `higher_is_better` for accuracy-style metrics. Use `lower_is_better` for metrics like loss or perplexity.
- `max_drop`
  Maximum tolerated absolute damage on the selected metric.
- `max_relative_drop`
  Optional relative guardrail.
- `relative_guard_below`
  Only apply the relative guardrail when the baseline score is below this value.

## Step 3: Run The Gate

Use a short Python entry point:

```python
from nemo_evaluator.config import load_gate_policy
from nemo_evaluator.engine import gate_runs, write_gate_report

policy = load_gate_policy("gate_policy.yaml")
report = gate_runs("results/baseline", "results/candidate", policy)

print(report.verdict)
print(report.verdict_reasons)

for benchmark in report.benchmarks:
    print(benchmark.benchmark, benchmark.status, benchmark.reasons)

write_gate_report(report, "results/candidate/gate_report.json")
```

If you prefer to run it from the shell without creating a separate file:

```bash
python - <<'PY'
from nemo_evaluator.config import load_gate_policy
from nemo_evaluator.engine import gate_runs, write_gate_report

policy = load_gate_policy("gate_policy.yaml")
report = gate_runs("results/baseline", "results/candidate", policy)

print(report.verdict)
print(report.verdict_reasons)
write_gate_report(report, "results/candidate/gate_report.json")
PY
```

## Step 4: Interpret The Results

The aggregate verdict is:

- `GO`: every required benchmark passed
- `NO-GO`: at least one required benchmark breached policy or was missing
- `INCONCLUSIVE`: no required benchmark breached, but at least one required benchmark did not have strong enough evidence to certify

Each benchmark result gets one status:

- `PASS`
- `BREACH`
- `INSUFFICIENT_EVIDENCE`
- `MISSING`

Typical reasons:

- `BREACH`
  the metric damage exceeded `max_drop`, or the relative guardrail triggered
- `INSUFFICIENT_EVIDENCE`
  no paired data, not enough paired items, or the confidence interval straddled the allowed threshold
- `MISSING`
  the policy required that benchmark, but NEL could not find it in both directories

## Step 5: Turn It Into A Release Check

In CI or a release script, fail on `NO-GO` and usually also fail on `INCONCLUSIVE`.

Example:

```python
import sys

from nemo_evaluator.config import load_gate_policy
from nemo_evaluator.engine import gate_runs, write_gate_report

policy = load_gate_policy("gate_policy.yaml")
report = gate_runs("results/baseline", "results/candidate", policy)
write_gate_report(report, "results/candidate/gate_report.json")

if report.verdict == "GO":
    sys.exit(0)

if report.verdict in {"NO-GO", "INCONCLUSIVE"}:
    sys.exit(1)
```

This is the simplest way to turn the policy into a blocking release decision.

## Recommended Workflow

A practical release workflow is:

1. Run the baseline benchmark set.
2. Run the candidate benchmark set.
3. Run `nel compare` on any benchmark that looks suspicious and use it for diagnosis.
4. Run the quality gate across the full benchmark set.
5. Treat `NO-GO` as a hard failure.
6. Treat `INCONCLUSIVE` as “collect more evidence” unless you have an explicit waiver process.

That split is important:

- regression is diagnostic
- quality gate is policy enforcement

## Troubleshooting

### “required benchmark must resolve to an explicit metric”

Add `metric` to the policy defaults or to the benchmark entry itself. Required benchmarks cannot rely on auto-detection.

### “unsupported metric”

Today the gate only supports `mean_reward` and `pass@1` for deterministic gating.

### `MISSING`

Check that the same benchmark names exist in both baseline and candidate directories and that each directory resolves to only one bundle for that benchmark.

### `INSUFFICIENT_EVIDENCE`

Most often this means one of these:

- `results.jsonl` is missing or empty
- too few paired problems were compared
- the confidence interval is too wide for the threshold you chose

### Duplicate benchmark error

NEL fails fast if two bundles in the same side resolve to the same benchmark name. Clean the directory or point the gate at a narrower results root.

## When To Prefer The Gate Over `nel compare`

Prefer the gate when:

- you need one decision across several benchmarks
- benchmarks have different importance tiers
- thresholds differ by benchmark
- a release should fail if required evidence is missing

Prefer `nel compare` when:

- you want to inspect flips on one benchmark
- you want a quick baseline vs candidate investigation
- you need human-readable debugging before changing the release policy
