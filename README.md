# NeMo Evaluator

Benchmark environments, evaluation runner, and environment-compatible service for LLM evaluation.

## Install

```bash
pip install -e .                # core
pip install -e ".[scoring]"     # + sympy for math scoring
pip install -e ".[all]"         # + ray, scipy, verifiers
```

## Quick Start

```bash
# Configure your model endpoint
export NEMO_MODEL_URL=https://api.example.com/v1
export NEMO_MODEL_ID=my-model

# Run a benchmark
nel run --benchmark gsm8k --repeats 2 --max-problems 50

# From a config file
nel run examples/configs/single_benchmark.yaml

# Validate a benchmark (quick sanity check, 5 samples)
nel validate --benchmark gsm8k

# List available benchmarks
nel list-environments
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `nel run` | Run evaluation (single benchmark, multi-task config, or remote adapter) |
| `nel serve` | Serve a benchmark as HTTP environment |
| `nel validate` | Quick sanity check on a benchmark |
| `nel regression` | Compare two evaluation bundles, detect regressions |
| `nel list-environments` | List registered benchmarks |
| `nel slurm eval` | Generate/submit SLURM job arrays for distributed eval |
| `nel slurm serve` | Generate/submit SLURM job for serving |
| `nel slurm merge` | Merge sharded evaluation results |

## Python API

```python
import asyncio
from nemo_evaluator import get_environment, run_evaluation, ModelClient
from nemo_evaluator.runner.artifacts import write_all

env = get_environment("gsm8k")
client = ModelClient(base_url="https://api.example.com/v1", model="my-model")
bundle = asyncio.run(run_evaluation(env, client, n_repeats=2, max_problems=10))
write_all(bundle, "./eval_results")
```

## Writing a Benchmark (BYOB)

```python
from nemo_evaluator.environments import EvalEnvironment, SeedResult, VerifyResult, register
from nemo_evaluator.scoring import math_equal, extract_answer

@register("my_benchmark")
class MyBenchmark(EvalEnvironment):
    def __init__(self):
        super().__init__()
        self._data = [{"question": "2+2=?", "answer": "4"}]  # load your dataset

    def __len__(self):
        return len(self._data)

    def seed(self, idx: int) -> SeedResult:
        row = self._data[idx]
        return SeedResult(prompt=row["question"], expected_answer=row["answer"])

    def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        extracted = extract_answer(response)
        correct = math_equal(extracted, expected)
        return VerifyResult(
            reward=1.0 if correct else 0.0,
            extracted_answer=extracted,
            scoring_details={"method": "math_equal"},
        )
```

Then: `nel validate -b my_benchmark --model-url $NEMO_MODEL_URL --model-id $NEMO_MODEL_ID`

## Serving for Gym

```bash
# Serve with Gym-compatible protocol
nel serve --benchmark gsm8k --gym-compat --port 9090

# Gym training points at: http://hostname:9090
# Endpoints: /seed_session, /verify, /health, /dataset_size
```

## Distributed Evaluation

```bash
# SLURM: 16 shards with auto-merge
nel slurm eval gsm8k --shards 16 --repeats 8 --submit

# Kubernetes: indexed job (see deploy/k8s/)
kubectl apply -f deploy/k8s/eval-indexed-job.yaml

# Ray: distributed across workers
python -m nemo_evaluator.runner.ray_launcher --benchmark gsm8k --shards 8

# Docker Compose
docker compose -f deploy/docker-compose.yaml up eval-local

# Manual sharding (any orchestrator)
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=4 nel run --benchmark gsm8k
```

## Project Structure

```
src/nemo_evaluator/
    benchmarks/        Built-in benchmarks (gsm8k, triviaqa)
    environments/      EvalEnvironment base class, registry, HTTP server
    adapters/          Consume external environments (HTTP, Prime Intellect, Gym harness)
    runner/            Eval loop, model client, artifacts, sharding, regression, Ray launcher
    observability/     StepRecord, RuntimeStats, FailureReport, progress tracking
    scoring/           Scoring primitives (math_equal, exact_match, extraction)
    metrics/           Statistical metrics (pass@k, bootstrap CI, aggregation)
    cli/               CLI commands (run, serve, validate, regression, slurm)
examples/
    configs/           YAML evaluation configs
    run_evaluation.py  Programmatic evaluation example
    gym_integration.py Serve, consume, and export for Gym
    pi_integration.py  Prime Intellect adapter example
deploy/
    Dockerfile, docker-compose, K8s manifests, SLURM scripts, GitLab CI
```

## Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `scoring` | sympy | Symbolic math comparison |
| `stats` | scipy | Normal confidence intervals |
| `ray` | ray | Distributed eval via Ray |
| `pi` | verifiers, openai | Prime Intellect environment adapter |
| `skills` | nemo-skills | NeMo Skills benchmark integration |
| `all` | all of the above | Everything |

## License

Apache-2.0. See [LICENSE](LICENSE).
