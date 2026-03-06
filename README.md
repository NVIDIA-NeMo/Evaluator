# NeMo Evaluator

Benchmark environments, evaluation runner, and environment-compatible service for LLM evaluation.

## Install

```bash
pip install -e .                # core
pip install -e ".[scoring]"     # + sympy for math scoring
pip install -e ".[all]"         # everything (ray, scipy, verifiers, lm-eval)
```

## Quick Start

```bash
# Configure your model endpoint
export NEMO_MODEL_URL=https://api.example.com/v1
export NEMO_MODEL_ID=my-model

# Run a benchmark
nel run --benchmark gsm8k --repeats 2 --max-problems 50

# From a config file (validated against schema, env vars expanded)
nel run examples/configs/single_benchmark.yaml

# Validate a benchmark (quick sanity check, 5 samples)
nel validate --benchmark gsm8k

# List available benchmarks
nel list-environments
```

## External Harnesses

NEL wraps existing eval harnesses by injecting itself into the model call path.
Both BYOB benchmarks and harness adapters are first-class -- use BYOB when you
need full scoring control, and harnesses for breadth.

```bash
# simple-evals (OpenAI): mmlu, math, gpqa, drop, humaneval, simpleqa, mgsm, browsecomp
nel harness run --harness simple-evals --eval mmlu --examples 500

# lm-evaluation-harness (EleutherAI): generation-based tasks
nel harness run --harness lm-eval --tasks gsm8k,minerva_math --fewshot 5

# List chat-API-compatible tasks (loglikelihood tasks are not supported via chat API)
nel harness list --harness lm-eval --generate-only

# List everything
nel harness list
```

Or in a config YAML (mixed harnesses in one run, validated with Pydantic schema):

```yaml
evaluation:
  model_url: ${NEMO_MODEL_URL}
  model_id: ${NEMO_MODEL_ID}
  tasks:
    - harness: simple-evals
      eval: math
      examples: 500
    - harness: lm-eval
      tasks: [gsm8k, minerva_math]
      fewshot: 5
    - benchmark: gsm8k  # native BYOB
      repeats: 4
    - adapter: gym://localhost:9090  # remote environment
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `nel run` | Run evaluation (BYOB, adapter, harness, or mixed config) |
| `nel harness run` | Run via external harness (simple-evals, lm-eval) |
| `nel harness list` | List available evals (`--generate-only` for chat-API-compatible) |
| `nel serve` | Serve a benchmark as HTTP environment (Gym-compatible) |
| `nel validate` | Quick sanity check on a benchmark |
| `nel report` | Aggregate bundles into comparison table (markdown, LaTeX, CSV, JSON) |
| `nel regression` | Compare two evaluation bundles, detect regressions |
| `nel list-environments` | List registered BYOB benchmarks |
| `nel slurm eval` | Generate/submit SLURM job arrays for distributed eval |
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

### Single-turn (EvalEnvironment)

```python
from nemo_evaluator.environments import EvalEnvironment, SeedResult, VerifyResult, register
from nemo_evaluator.scoring import math_equal, extract_answer

@register("my_benchmark")
class MyBenchmark(EvalEnvironment):
    def __init__(self):
        super().__init__()
        self._data = [{"question": "2+2=?", "answer": "4"}]

    def __len__(self):
        return len(self._data)

    def seed(self, idx: int) -> SeedResult:
        row = self._data[idx]
        return SeedResult(prompt=row["question"], expected_answer=row["answer"])

    def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        extracted = extract_answer(response)
        correct = math_equal(extracted, expected)
        return VerifyResult(reward=1.0 if correct else 0.0, extracted_answer=extracted,
                            scoring_details={"method": "math_equal"})
```

### Multi-turn (StepEnvironment)

For interactive / agent benchmarks where the model takes multiple steps:

```python
from nemo_evaluator.environments import StepEnvironment, Observation, register

@register("my_agent_bench")
class MyAgentBench(StepEnvironment):
    def __init__(self):
        self._tasks = [{"instruction": "Solve the puzzle", "answer": "42"}]

    def __len__(self):
        return len(self._tasks)

    def reset(self, idx: int) -> Observation:
        self._current = self._tasks[idx]
        return Observation(content=self._current["instruction"],
                           tools=[{"name": "calculator", "schema": {"expr": "string"}}])

    def step(self, action: str) -> Observation:
        correct = action.strip() == self._current["answer"]
        return Observation(content="Correct!" if correct else "Try again",
                           done=correct, reward=1.0 if correct else 0.0)

    @property
    def max_steps(self) -> int:
        return 10
```

Then: `nel validate -b my_benchmark --model-url $NEMO_MODEL_URL --model-id $NEMO_MODEL_ID`

## Serving for Gym

```bash
# Serve with Gym-compatible protocol (auto-detects evaluator and gym verify formats)
nel serve --benchmark gsm8k --gym-compat --port 9090

# Endpoints: /seed_session, /verify, /health, /dataset_size
```

## Adapter Proxy (trajectory capture for external environments)

When an external system (Gym, Harbor, PI) owns the model call, the adapter proxy
sits between the agent and the model endpoint to capture full trajectories:

```python
from nemo_evaluator.adapters.proxy import AdapterProxy

proxy = AdapterProxy(target_url="https://api.nvidia.com/v1", api_key="...")
# Run proxy.app on a port, set agent's model URL to:
#   http://proxy-host:port/problem/{task_id}/v1/chat/completions
# Retrieve trajectories: proxy.get_trajectories(task_id)
```

## Distributed Evaluation

```bash
# SLURM: 16 shards with auto-merge
nel slurm eval gsm8k --shards 16 --repeats 8 --submit

# Kubernetes: indexed job (see deploy/k8s/)
kubectl apply -f deploy/k8s/eval-indexed-job.yaml

# Ray
python -m nemo_evaluator.runner.ray_launcher --benchmark gsm8k --shards 8

# Docker Compose
docker compose -f deploy/docker-compose.yaml up eval-local

# Manual sharding (any orchestrator)
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=4 nel run --benchmark gsm8k
```

## Project Structure

```
src/nemo_evaluator/
    environments/      EvalEnvironment + StepEnvironment base classes, registry, HTTP server
    harnesses/         External harness adapters (simple-evals, lm-eval)
    benchmarks/        BYOB reference implementations (gsm8k, triviaqa)
    adapters/          Consume external environments (HTTP, PI, Gym, proxy)
    runner/            Eval loop, model client, artifacts, sharding, caching, checkpointing, async bridge
    observability/     StepRecord, RuntimeStats, FailureReport, progress tracking
    scoring/           Scoring primitives (math_equal, mcq, judge, exact_match, json_schema)
    metrics/           Statistical metrics (pass@k, bootstrap CI, aggregation)
    cli/               CLI commands (run, harness, serve, validate, report, regression, slurm)
    config_schema.py   Pydantic YAML config validation with env var expansion
    models.py          Pydantic models for configs, bundles, regression
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
| `lm-eval` | lm_eval | EleutherAI lm-evaluation-harness |
| `harnesses` | lm_eval | All external harness adapters |
| `all` | all of the above | Everything |

## License

Apache-2.0. See [LICENSE](LICENSE).
