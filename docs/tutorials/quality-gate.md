# Implementing Quality Gates

## The Problem

Real model releases do not depend on one benchmark.

A quantized checkpoint might preserve MMLU accuracy while degrading math reasoning.  A fine-tuned model might improve code generation while regressing on instruction following.  A single comparison on a single benchmark cannot catch these asymmetric regressions.

The release process typically requires rules like:

- MMLU-Pro and GPQA are critical -- any regression above 1 pp blocks the release.
- HumanEval and TriviaQA are supporting -- a slightly relaxed threshold of 1.5 pp.
- IFEval is tracked for information but does not block.
- Low-baseline benchmarks (AIME, HLE) need a relative-drop guardrail because 1 pp absolute means very different things at 80% vs 10% baseline accuracy.
- If a required benchmark was not evaluated, the release cannot be certified.

Without automation, teams run `nel compare` on each benchmark separately, open a spreadsheet, copy the deltas, eyeball whether each one is within tolerance, and manually declare "go" or "no-go."  This is error-prone, unreproducible, and does not scale.

## The Approach

`nel gate` automates the multi-benchmark quality decision.  It takes two inputs:

1. **Result directories** containing evaluation bundles for baseline and candidate (one per benchmark).
2. **A policy YAML** that declares which benchmarks are required, what tier each belongs to, and what thresholds to apply.

The gate then:

1. **Discovers** all eval bundles in both directories and matches them by benchmark name.
2. **Checks** that every required benchmark is present in both baseline and candidate.
3. **Evaluates** each matched pair using paired per-item data: loads `results.jsonl`, computes the delta on the primary metric, builds a 95% confidence interval on the paired delta.
4. **Applies** the policy threshold: is the delta within tolerance?  Is the relative drop within the guardrail?  Is the evidence sufficient?
5. **Aggregates** per-benchmark statuses into a single verdict.

### How the Gate Differs from `nel compare`

`nel compare` uses McNemar's significance test: "is this regression statistically real?"  That is an investigation tool.

`nel gate` uses threshold-based gating: "does this regression exceed the allowed tolerance?"  That is a release decision.

A benchmark can be statistically significant but within tolerance (acceptable).  A benchmark can exceed the tolerance but lack statistical significance because the sample is small (insufficient evidence).  The gate distinguishes these cases.

### Per-Benchmark Status

| Status | Meaning |
|--------|---------|
| **PASS** | The primary metric delta is within the allowed threshold |
| **BREACH** | The delta exceeds the threshold (absolute or relative) |
| **INSUFFICIENT_EVIDENCE** | Not enough paired data to compute the delta reliably (< 10 paired items, or missing `results.jsonl`) |
| **MISSING** | The benchmark is required by the policy but was not found in the results |

### Aggregate Verdict

| Verdict | Rule |
|---------|------|
| **GO** | All critical and supporting benchmarks passed |
| **NO-GO** | At least one critical or supporting benchmark is BREACH or MISSING |
| **INCONCLUSIVE** | At least one critical or supporting benchmark has INSUFFICIENT_EVIDENCE, but none breached |

Advisory benchmarks appear in the report but never affect the aggregate verdict.

### Benchmark Tiers

| Tier | Blocks release? | Typical use |
|------|-----------------|-------------|
| **critical** | Yes | Core capability benchmarks (MMLU-Pro, GPQA, AIME) |
| **supporting** | Yes (relaxed threshold) | Important but secondary benchmarks (HumanEval, TriviaQA) |
| **advisory** | No | Experimental or informational benchmarks (IFEval, TruthfulQA) |

Both critical and supporting benchmarks have hard caps.  The distinction is the threshold value, not whether the benchmark blocks.  If you want a truly non-blocking benchmark, use `advisory`.

## Walkthrough

### Step 1: Organize Your Evaluation Results

Run your benchmark suite against both baseline and candidate.  Organize results with one subdirectory per benchmark:

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

The `results.jsonl` files are essential.  Without per-problem data, the gate can only report INSUFFICIENT_EVIDENCE.  Always run evaluations that produce `results.jsonl`.

### Step 2: Write the Policy

Create a YAML file that declares your release criteria:

```yaml
# gate_policy.yaml
version: 1

defaults:
  tier: supporting
  metric: mean_reward
  direction: higher_is_better
  max_drop: 0.015          # 1.5 pp for supporting benchmarks

benchmarks:
  mmlu_pro:
    tier: critical
    max_drop: 0.01          # 1.0 pp for critical

  gpqa:
    tier: critical
    max_drop: 0.01

  humaneval:
    metric: pass@1           # code benchmarks use pass@1
    max_drop: 0.015

  aime_2025:
    tier: critical
    max_drop: 0.01
    max_relative_drop: 0.02  # 2% relative guardrail
    relative_guard_below: 0.20  # only apply when baseline < 20%

  triviaqa:
    tier: advisory            # tracked, does not block
```

**How to think about each field:**

- **`tier`**: Controls whether a breach blocks the release.  `critical` and `supporting` both block; `advisory` does not.
- **`metric`**: The single number the gate checks.  Use `mean_reward` for accuracy-style benchmarks, `pass@1` for code execution benchmarks.  This must be a metric present in the eval bundle's scores.
- **`direction`**: `higher_is_better` (default) means a decrease is a regression.  Use `lower_is_better` for metrics like perplexity or loss.
- **`max_drop`**: Maximum tolerated absolute regression on the 0-1 scale.  `0.01` = 1 percentage point.
- **`max_relative_drop`**: Optional relative guardrail.  `0.02` = 2%.  Only meaningful for low-baseline benchmarks where a small absolute drop is a large relative change.
- **`relative_guard_below`**: The relative guardrail only activates when the baseline score is below this value.  At 80% accuracy, a 1 pp drop is a 1.25% relative change (harmless).  At 10% accuracy, a 1 pp drop is a 10% relative change (significant).

**Benchmarks not listed in the policy** inherit the defaults.  If the gate discovers an eval bundle for an unlisted benchmark, it applies the default tier and threshold.  If you want to require specific benchmarks, list them explicitly -- the gate will report MISSING for any listed benchmark not found in the results.

### Step 3: Run the Gate

```bash
nel gate ./results/baseline ./results/candidate --policy gate_policy.yaml
```

Sample output:

```text
GO
Policy:    gate_policy.yaml
Baseline:  ./results/baseline
Candidate: ./results/candidate

VERDICT REASONS
  - All 4 gated benchmark(s) passed

BENCHMARKS
  name                 tier         status                 metric       delta
  -------------------------------------------------------------------------------
  gpqa                 critical     PASS                   mean_reward  -0.0080
  humaneval            supporting   PASS                   pass@1       -0.0050
  mmlu_pro             critical     PASS                   mean_reward  -0.0030
  triviaqa             advisory     PASS                   mean_reward  +0.0120
```

### Step 4: Handle Failures

When the gate returns NO-GO, the output tells you which benchmarks failed and why:

```text
NO-GO
Policy:    gate_policy.yaml
Baseline:  ./results/baseline
Candidate: ./results/candidate

VERDICT REASONS
  - BREACH: gpqa [critical]

BENCHMARKS
  name                 tier         status                 metric       delta
  -------------------------------------------------------------------------------
  gpqa                 critical     BREACH                 mean_reward  -0.0150
    - Absolute drop 0.0150 exceeds threshold 0.01
  mmlu_pro             critical     PASS                   mean_reward  -0.0030
```

To investigate the breach, use `nel compare` on the specific benchmark:

```bash
nel compare ./results/baseline/gpqa ./results/candidate/gpqa \
  --show-flips --verbose
```

This gives you the per-problem detail: which questions regressed, what the baseline and candidate answered, and whether the regression is statistically significant.

### Step 5: Use in CI and Release Scripts

With `--strict`, exit codes work directly in automation:

| Exit code | Verdict |
|-----------|---------|
| 0 | GO |
| 1 | NO-GO |
| 2 | INCONCLUSIVE |

```bash
nel gate ./results/baseline ./results/candidate \
  --policy gate_policy.yaml \
  --strict
```

For machine-readable output:

```bash
nel gate ./results/baseline ./results/candidate \
  --policy gate_policy.yaml \
  --format json \
  --output ./artifacts/gate_report.json
```

### Step 6: Use the Python API

```python
from nemo_evaluator.config.gate_policy import load_gate_policy
from nemo_evaluator.engine.gate import gate_runs, write_gate_report

policy = load_gate_policy("gate_policy.yaml")
report = gate_runs("./results/baseline", "./results/candidate", policy)

print(report.verdict)  # GO / NO-GO / INCONCLUSIVE

for b in report.benchmarks:
    status = "OK" if b.status == "PASS" else b.status
    print(f"  {b.benchmark}: {status}  delta={b.delta}")

if report.verdict != "GO":
    write_gate_report(report, "gate_report.json")
```

## Reference: All Flags

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `--policy` / `-p` | Yes | -- | Path to the gate policy YAML |
| `--strict` | No | on | Exit non-zero on NO-GO (1) or INCONCLUSIVE (2) |
| `--output` / `-o` | No | none | Write JSON report to file |
| `--format` | No | text | `text` or `json` |
| `--verbose` | No | off | Show per-benchmark reasons and warnings |

## Common Patterns

### Quantization Release Gate

For NVFP4/INT4 checkpoint qualification where the baseline is the BF16 source artifact:

```yaml
version: 1
defaults:
  tier: supporting
  metric: mean_reward
  max_drop: 0.015
benchmarks:
  mmlu_pro:
    tier: critical
    max_drop: 0.01
  gpqa:
    tier: critical
    max_drop: 0.01
  aime_2025:
    tier: critical
    max_drop: 0.01
    max_relative_drop: 0.02
    relative_guard_below: 0.15
  aa_omniscience:
    tier: critical
    max_drop: 0.01
  scicode:
    metric: pass@1
  aa_lcr: {}
  ifeval:
    tier: advisory
```

### Per-Commit CI Gate

For catching regressions in model training or prompt engineering:

```yaml
version: 1
defaults:
  tier: critical
  metric: mean_reward
  max_drop: 0.02
benchmarks:
  gsm8k: {}
  mmlu_pro: {}
  humaneval:
    metric: pass@1
```

### Perplexity-Aware Gate

For lower-is-better metrics:

```yaml
version: 1
defaults:
  metric: mean_reward
  max_drop: 0.015
benchmarks:
  wikitext_ppl:
    direction: lower_is_better
    max_drop: 0.5
    tier: critical
  mmlu_pro:
    tier: critical
    max_drop: 0.01
```

## Troubleshooting

### MISSING

The gate could not find a required benchmark in both the baseline and candidate directories.  Check:

- Benchmark names match between policy YAML and eval bundle `benchmark.name` field.
- Both directories have the benchmark subdirectory with an `eval-*.json` file.
- Spelling is exact (the match is case-sensitive).

### INSUFFICIENT_EVIDENCE

The gate found the benchmark but could not compute a reliable delta.  Common causes:

- `results.jsonl` is missing or empty -- re-run the evaluation.
- Fewer than 10 paired items -- use `--max-problems` to increase the sample size.

### Duplicate Benchmark Error

The results directory contains multiple eval bundles that resolve to the same benchmark name.  This happens when stale results from previous runs remain in the directory.  Clean the directory or point the command at a more specific subdirectory.

### "required benchmark must resolve to an explicit metric"

Add `metric` to the benchmark entry or to the policy defaults.  The gate needs to know which score to check.

## The Workflow

The recommended workflow for release qualification:

1. **Evaluate** the benchmark suite against baseline and candidate.
2. **Gate** with `nel gate --policy ... --strict`.
3. If **GO**: proceed to performance testing, compliance, and other release gates.
4. If **NO-GO**: use `nel compare --show-flips --verbose` on each failing benchmark to diagnose the regression.
5. If **INCONCLUSIVE**: increase sample size or add distributional metrics for the affected benchmarks.
