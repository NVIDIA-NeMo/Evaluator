# Local Deployment

Run evaluations directly on your workstation.

## Single benchmark

```bash
nel eval run --bench gsm8k --repeats 4 \
    --model-url https://inference-api.nvidia.com/v1 \
    --model-id azure/openai/gpt-5.2 \
    --output-dir ./results
```

## Config file

```yaml
# eval.yaml
model:
  url: https://inference-api.nvidia.com/v1
  id: azure/openai/gpt-5.2

benchmarks:
  - name: gsm8k
    repeats: 4
    system_prompt: "Solve step by step. Put your final answer in \\boxed{}."
  - name: triviaqa
    repeats: 1
    max_problems: 200
```

```bash
nel eval run eval.yaml
```

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
| `NEMO_API_KEY` | API key for model endpoint | `sk-...` |
| `NEL_SHARD_IDX` | Shard index for distributed eval | `0` |
| `NEL_TOTAL_SHARDS` | Total shards for distributed eval | `8` |
