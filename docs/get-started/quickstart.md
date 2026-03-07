# Quickstart

Run a complete evaluation in under 5 minutes.

## Step 1: Set your model endpoint

```bash
export NEMO_API_KEY="your-api-key-here"
```

## Step 2: Run an evaluation

```bash
nel run --env gsm8k \
  --model-url https://api.example.com/v1 \
  --model-id my-model \
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
nel run --env gsm8k --repeats 4 -o ./results/candidate

nel regression ./results/baseline/eval-*.json ./results/candidate/eval-*.json --strict
```

Output:

```
Baseline:  eval-20260224T100000Z-gsm8k
Candidate: eval-20260225T143012Z-gsm8k

  pass@1: 0.7200 -> 0.7500  (delta=+0.0300, +4.2%, CI overlap)
  pass@4: 0.8800 -> 0.9100  (delta=+0.0300, +3.4%, CI overlap)

No regressions beyond 5% threshold.
```

## Step 5: Use a config file

For reproducible multi-benchmark evaluations:

```yaml
# eval_config.yaml
evaluation:
  model_url: https://inference-api.nvidia.com/v1
  model_id: azure/openai/gpt-5.2

  tasks:
    - env: gsm8k
      repeats: 4
      system_prompt: "Solve step by step. Put your final answer in \\boxed{}."

    - env: triviaqa
      repeats: 1
      max_problems: 100
```

```bash
nel run eval_config.yaml
```

Each task gets its own output directory with the full artifact suite.

## Next Steps

- {doc}`../tutorials/byob` -- Write your own benchmark with `@benchmark` + `@scorer`
- {doc}`../tutorials/gym-integration` -- Serve benchmarks for Gym training
- {doc}`../tutorials/distributed-eval` -- Scale to thousands of problems
- {doc}`../architecture/index` -- Understand how the system works
