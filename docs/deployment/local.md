# Local Deployment

Run evaluations directly on your workstation.

## Single benchmark (CLI)

```bash
nel eval run --bench gsm8k --repeats 4 \
    --model-url https://integrate.api.nvidia.com/v1/chat/completions \
    --model-id nvidia/nemotron-3-super-120b-a12b \
    --output-dir ./results
```

## Config file

```yaml
# eval.yaml
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
    max_problems: 200
    solver:
      type: simple
      service: nemotron
```

```bash
nel eval run eval.yaml
```

## Multi-model evaluation

Use named services to evaluate multiple models in the same config:

```yaml
services:
  solver:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

  judge:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

benchmarks:
  - name: simpleqa
    solver:
      type: simple
      service: solver
    scoring:
      metrics:
        - type: judge
          name: correctness
          service: judge
```

## Resume a partially completed suite

If a benchmark fails mid-suite, the remaining benchmarks still execute. Re-run with `--resume` to retry only the failed ones:

```bash
nel eval run eval.yaml --resume
```

Completed benchmarks are skipped automatically. Without `--resume`, all benchmarks are re-run from scratch.

## Serve for Gym

```bash
nel serve -b gsm8k --gym-compat --port 9090
```

## Validate a benchmark

Quick sanity check (10 samples, prints pass/fail per sample):

```bash
nel validate -b gsm8k --samples 10
```

## Environment variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `NVIDIA_API_KEY` | API key for NVIDIA endpoints | `nvapi-...` |
| `OPENAI_API_KEY` | API key for OpenAI endpoints | `sk-...` |
| `NEL_SHARD_IDX` | Shard index for distributed eval | `0` |
| `NEL_TOTAL_SHARDS` | Total shards for distributed eval | `8` |
