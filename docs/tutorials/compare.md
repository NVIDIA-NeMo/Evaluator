# Comparing Evaluation Runs

## The Problem

You trained a new model, changed a prompt, or quantized a checkpoint.  The headline score moved by a fraction of a percent.  Is the change real, or is it noise?

A single average score hides critical information:

- A candidate can **swap** improvements on some problems for regressions on others and keep the same headline.
- A small benchmark (HumanEval: 164 items) can fluctuate by several percentage points between runs just from sampling noise.
- A regression on an important subset (math word problems, long-context retrieval) can be invisible in the aggregate.

Eyeballing two numbers and declaring "it went down 0.3 points" is not a measurement.  It is a guess.

## The Approach

`nel compare` treats the comparison as a **paired statistical test**.  It evaluates the same problems with both models and analyzes what changed at the individual problem level:

1. **Pair** results by `(problem_idx, repeat)` across baseline and candidate.
2. **Classify** each pair: did the problem flip from correct to incorrect (regression), incorrect to correct (improvement), or stay the same?
3. **Test** whether the regression count is significantly larger than would occur by chance, using McNemar's exact binomial test.
4. **Report** the effect size, confidence interval, and a human-readable verdict.

This is the same methodology described in "When LLMs Get Significantly Worse" (ICLR 2026).  The key insight is that paired analysis dramatically reduces noise compared to comparing two independent averages, because the difficulty of each problem is "controlled for" -- you are measuring the *change in behavior* on the same inputs.

### Why McNemar and Not a t-test

McNemar's test focuses on **discordant pairs** -- problems where the two models disagree.  It ignores the (often vast majority of) problems both models get right or both get wrong.  This makes it far more sensitive to targeted regressions than a t-test on overall accuracy, which dilutes the signal with concordant pairs.

### What the Verdicts Mean

| Verdict | Meaning | What to do |
|---------|---------|------------|
| **PASS** | No evidence of a meaningful regression | Proceed |
| **WARN** | Statistically significant change, but below the practical threshold | Inspect the flipped problems; decide whether the affected areas matter for your use case |
| **BLOCK** | Significant regression that exceeds the configured tolerance | Investigate the regressed problems; fix the candidate or reject it |
| **INCONCLUSIVE** | Not enough paired data to detect regressions at the configured threshold | Re-run with more problems, or use a larger benchmark |

## Walkthrough

### Step 1: Produce Baseline and Candidate Evaluations

Run the same benchmark with the same configuration against both models:

```bash
# Baseline
nel eval run --bench mmlu_pro \
  --model-url https://integrate.api.nvidia.com/v1 \
  --model-id baseline-model \
  --repeats 1 \
  --max-problems 500 \
  -o ./results/baseline

# Candidate
nel eval run --bench mmlu_pro \
  --model-url https://integrate.api.nvidia.com/v1 \
  --model-id candidate-model \
  --repeats 1 \
  --max-problems 500 \
  -o ./results/candidate
```

For a valid comparison, match the runs on: benchmark, dataset slice, prompt template, repeat count, and max problems.  Differences in any of these confound the comparison.

### Step 2: Run the Comparison

```bash
nel compare ./results/baseline ./results/candidate
```

`nel compare` accepts directories (it finds the `eval-*.json` bundle inside) or direct bundle paths.

The command outputs:

- **Score deltas**: how each metric changed (absolute and relative)
- **Flip summary**: how many problems regressed, improved, or stayed the same
- **McNemar test**: p-value, effect size, and confidence interval
- **Verdict**: PASS, WARN, BLOCK, or INCONCLUSIVE
- **Markdown report**: auto-generated investigation document next to the candidate bundle

### Step 3: Tighten the Threshold for Sensitive Comparisons

The default practical threshold is 5% (0.05 on the 0-1 scale).  For quantization or fine-tuning where 1 percentage point matters:

```bash
nel compare ./results/baseline ./results/candidate --max-drop 0.01
```

This tells the verdict logic: "a regression is only practically meaningful if the net effect exceeds 1 pp."  Smaller effects get WARN instead of BLOCK.

### Step 4: Inspect What Actually Changed

Add `--show-flips` to see the individual problems that flipped:

```bash
nel compare ./results/baseline ./results/candidate --show-flips --verbose
```

This prints:

- Each regressed problem: index, category, expected answer, what baseline said, what candidate said
- Each improved problem: same detail
- Category breakdown: which subjects or topics were hit hardest
- Statistical detail: p-value, effect size, discordant pair count, minimum detectable effect

The `--verbose` flag adds the statistical detail.  Without it, you get the flip list and verdict but not the p-values.

### Step 5: Use the Investigation Report

By default, `nel compare` writes `regression_report.md` in the candidate's result directory.  This Markdown file contains:

- Side-by-side model responses for each flipped problem
- Category-level regression rates
- Statistical summary
- Suggested next steps

This report is designed for humans reviewing a merge request or a model release.  Attach it to the MR or the model card review.

To write it to a specific path:

```bash
nel compare ./results/baseline ./results/candidate --report ./review/mmlu_comparison.md
```

To suppress it:

```bash
nel compare ./results/baseline ./results/candidate --no-report
```

### Step 6: Use in CI Pipelines

With `--strict`, the command returns exit codes suitable for CI:

| Exit code | Verdict |
|-----------|---------|
| 0 | PASS |
| 1 | BLOCK |
| 2 | WARN or INCONCLUSIVE |

```bash
nel compare ./results/baseline ./results/candidate \
  --max-drop 0.01 --strict
```

For machine-readable output, use `--format json`:

```bash
nel compare ./results/baseline ./results/candidate --format json > report.json
```

### Step 7: Use the Python API

For programmatic access:

```python
from nemo_evaluator.engine.comparison import compare_runs, write_regression

report = compare_runs("./results/baseline/eval-mmlu.json",
                       "./results/candidate/eval-mmlu.json")

print(report["verdict"])  # PASS / WARN / BLOCK / INCONCLUSIVE

# Per-metric deltas
for metric, d in report["score_deltas"].items():
    print(f"{metric}: {d['baseline']:.4f} -> {d['candidate']:.4f}  "
          f"(delta={d['delta']:+.4f}, {d['relative_pct']:+.1f}%)")

# Flip summary
flip = report["flip_report"]["summary"]
print(f"Regressions: {flip['n_regressions']}, "
      f"Improvements: {flip['n_improvements']}, "
      f"Paired: {flip['n_paired']}")

# McNemar test
m = report["mcnemar"]
if m.get("p_value") is not None:
    print(f"McNemar p={m['p_value']:.4f}, effect={m['effect_size']:.4f}")

write_regression(report, "comparison.json")
```

## Reference: All Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--max-drop` / `-t` | 0.05 | Practical effect threshold (0-1 scale) |
| `--strict` | off | Exit non-zero on BLOCK, WARN, or INCONCLUSIVE |
| `--correct-above` | 0.0 | Reward threshold for "correct" classification.  Use `0.5` for judge-scored benchmarks where reward is a continuous score. |
| `--show-flips` | off | Print per-problem flip details |
| `--verbose` | off | Show statistical details (p-values, effect sizes, power) |
| `--compact` | off | Short output for Slack or CI logs |
| `--format` | text | `text` or `json` |
| `--output` / `-o` | none | Write JSON report to file |
| `--report` | auto | Write Markdown report (default: next to candidate bundle) |
| `--no-report` | off | Suppress Markdown report |

## Understanding Sample Size and Power

A common mistake is running a small benchmark and treating the result as definitive.  The comparison's statistical power depends on the number of **discordant pairs** -- problems where the two models disagree.

Rule of thumb:

| Discordant pairs | Minimum detectable effect (80% power) |
|------------------|---------------------------------------|
| 10 | ~28% |
| 50 | ~12.5% |
| 100 | ~8.8% |
| 500 | ~3.9% |
| 1000 | ~2.8% |

If your benchmark has 164 items (HumanEval) and 90% concordance, you might have only ~16 discordant pairs.  That means you can reliably detect ~22% regression rates, not 1-2 point deltas.  `nel compare` reports the minimum detectable effect and will return INCONCLUSIVE when the test is underpowered.

## When to Use `nel compare` vs `nel gate`

| Need | Tool |
|------|------|
| Diagnose what changed in one benchmark | `nel compare` |
| Investigate why a specific benchmark regressed | `nel compare --show-flips --verbose` |
| Make a release decision across a suite of benchmarks | `nel gate` |
| CI gate on a single benchmark | `nel compare --strict` |
| CI gate on multiple benchmarks with per-benchmark thresholds | `nel gate --strict` |

`nel compare` is the diagnostic tool.  `nel gate` is the policy enforcement tool.  A typical workflow uses `nel gate` first, then `nel compare` on any failing benchmarks to understand what went wrong.
