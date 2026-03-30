# Quickstart

Run a complete evaluation in under 5 minutes.

## Step 1: Set your model endpoint

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

## Step 2: Run an evaluation

```bash
nel eval run --bench gsm8k \
  --model-url https://integrate.api.nvidia.com/v1/chat/completions \
  --model-id nvidia/nemotron-3-super-120b-a12b \
  --repeats 2 --max-problems 20
```

Live progress output:

```
gsm8k  [12/20]  60.0%  ████████████░░░░░░░░  acc=0.750  1.2k tok/s  ETA 0:32
```

## Step 3: Read the results

After completion, `nel` prints a summary and writes artifacts:

```
gsm8k
  problems=20  repeats=2
  pass@1: 0.7500  [0.5500, 0.9000]
  pass@2: 0.8500  [0.6500, 0.9500]
  tokens=12,400  speed=2.1/s

  Output: ./eval_results/
    bundle: eval-20260225T143012Z-gsm8k.json
    results: results.jsonl
    trajectories: trajectories.jsonl
    runtime_stats: runtime_stats.json
    failure_analysis: failure_analysis.json
```

### Artifact reference

| File | Contents | Use case |
|------|----------|----------|
| `eval-*.json` | Full bundle: config, scores, CI, categories | Regression comparison, model cards |
| `results.jsonl` | Per-problem: prompt, response, reward, extracted answer | Debugging, error analysis |
| `trajectories.jsonl` | Full step records: timing, tokens, scoring details | RL pipeline input, audit |
| `runtime_stats.json` | Latency percentiles, token distributions, throughput | Performance analysis |
| `failure_analysis.json` | Categorized failures with exemplars | Model improvement |

## Step 4: Compare against a baseline

```bash
nel eval run --bench gsm8k --repeats 4 -o ./results/candidate

nel compare ./results/baseline ./results/candidate --strict
```

`nel compare` accepts directories or bundle files. It pairs results at the problem level, runs a McNemar exact test for regression detection, and generates an investigation report.

When `scipy` is installed (`pip install nemo-evaluator[stats]`), comparisons include McNemar significance testing and confidence intervals on effect sizes.

See {doc}`../tutorials/compare` for the full walkthrough.

## Step 5: Gate a release across multiple benchmarks

For release qualification, use `nel gate` with a policy file:

```yaml
# gate_policy.yaml
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
```

```bash
nel gate ./results/baseline ./results/candidate --policy gate_policy.yaml --strict
```

The gate applies per-benchmark thresholds and returns `GO`, `NO-GO`, or `INCONCLUSIVE`. With `--strict`, exit codes work directly in CI (0/1/2).

See {doc}`../tutorials/quality-gate` for the full walkthrough.

## Step 6: Use a config file

For reproducible multi-benchmark evaluations:

```yaml
# eval_config.yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

benchmarks:
  - name: gsm8k
    repeats: 4
    solver:
      type: simple
      service: nemotron
      system_prompt: "Solve step by step. Put your final answer in \\boxed{}."

  - name: triviaqa
    repeats: 1
    max_problems: 100
    solver:
      type: simple
      service: nemotron
```

```bash
nel eval run eval_config.yaml
```

Each task gets its own output directory with the full artifact suite.

## Step 7: Resume a failed suite

If a benchmark fails mid-suite (e.g., a network error on benchmark 3/5), the remaining benchmarks still execute. Re-run with `--resume` to retry only the failed ones:

```bash
nel eval run eval_config.yaml --resume
```

Completed benchmarks are skipped. Failed benchmarks are retried.

## Next Steps

- {doc}`../tutorials/compare` -- Deep dive into comparing runs and investigating regressions
- {doc}`../tutorials/quality-gate` -- Set up multi-benchmark release gates with policy files
- {doc}`../tutorials/byob` -- Write your own benchmark with `@benchmark` + `@scorer`
- {doc}`../tutorials/gym-integration` -- Serve benchmarks for Gym training
- {doc}`../tutorials/distributed-eval` -- Scale to thousands of problems
- {doc}`../architecture/index` -- Understand how the system works
