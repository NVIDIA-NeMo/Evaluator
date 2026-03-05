# API Reference

## Core Types

### `SeedResult`

Returned by `EvalEnvironment.seed()`.

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | The prompt to send to the model |
| `expected_answer` | `str` | Ground-truth answer for verification |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (category, difficulty, aliases) |

### `VerifyResult`

Returned by `EvalEnvironment.verify()`.

| Field | Type | Description |
|-------|------|-------------|
| `reward` | `float` | Score (0.0 = wrong, 1.0 = correct) |
| `extracted_answer` | `str \| None` | What was extracted from the model response |
| `scoring_details` | `dict[str, Any]` | Method used, intermediate values |
| `metadata` | `dict[str, Any]` | Additional verification metadata |

### `EvalEnvironment`

Abstract base class for all benchmarks.

```python
from nemo_evaluator.environments import EvalEnvironment, SeedResult, VerifyResult, register

@register("my_benchmark")
class MyBenchmark(EvalEnvironment):
    def __init__(self):
        super().__init__()
        self._dataset = [...]

    def __len__(self) -> int:
        return len(self._dataset)

    def seed(self, idx: int) -> SeedResult:
        ...

    def verify(self, response: str, expected: str, **metadata) -> VerifyResult:
        ...
```

## Runner

### `ModelClient`

Async client for OpenAI-compatible endpoints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `https://inference-api.nvidia.com/v1` | API base URL |
| `model` | `str` | `azure/openai/gpt-5.2` | Model identifier |
| `api_key` | `str \| None` | `None` (uses `NEMO_API_KEY`) | API key |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Max completion tokens |
| `timeout` | `float` | `120.0` | Request timeout in seconds |
| `max_concurrent` | `int` | `8` | Max parallel requests |

```python
from nemo_evaluator.runner import ModelClient

client = ModelClient(
    base_url="https://inference-api.nvidia.com/v1",
    model="azure/openai/gpt-5.2",
    api_key="sk-...",
    max_concurrent=16,
)
response = await client.chat("What is 2+2?", system="Answer briefly.")
```

### `run_evaluation()`

Core evaluation loop.

```python
async def run_evaluation(
    env: EvalSource,
    client: ModelClient,
    n_repeats: int = 1,
    max_problems: int | None = None,
    system_prompt: str | None = None,
    config: dict[str, Any] | None = None,
    progress: ProgressTracker | None = None,
    problem_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
```

| Parameter | Description |
|-----------|-------------|
| `env` | `EvalEnvironment` or `EnvironmentAdapter` |
| `client` | Model client for chat completions |
| `n_repeats` | Number of times to evaluate each problem |
| `max_problems` | Limit to first N problems |
| `system_prompt` | System message prepended to each request |
| `config` | Metadata included in the output bundle |
| `progress` | Progress tracker (default: no-op) |
| `problem_range` | `(start, end)` for sharded execution |

Returns a bundle dict containing metrics, results, config, and artifacts.

### `write_all()`

Write all artifacts to disk.

```python
from nemo_evaluator.runner import write_all

write_all(bundle, "./eval_results")
```

Writes: `eval-*.json` (bundle), `results.jsonl`, `trajectories.jsonl`, `runtime_stats.json`, `failure_analysis.json`.

### `compare_runs()`

Compare two evaluation bundles for regression.

```python
from nemo_evaluator.runner import compare_runs

report = compare_runs("baseline/eval-*.json", "candidate/eval-*.json")
```

## Adapters

### `EnvironmentAdapter`

Abstract base for external environment adapters.

```python
class EnvironmentAdapter(ABC):
    name: str

    async def seed(self, idx: int) -> SeedResult: ...
    async def verify(self, response: str, expected: str, **meta) -> VerifyResult: ...
    async def dataset_size(self) -> int: ...
```

### `GymAdapter`

HTTP adapter for consuming `nel serve` endpoints or Gym resource servers.

```python
from nemo_evaluator.adapters import GymAdapter

adapter = GymAdapter("http://localhost:9090")
bundle = await run_evaluation(adapter, client, n_repeats=4)
```

### `PIAdapter`

Decomposes `verifiers.SingleTurnEnv` into seed/verify.

```python
from nemo_evaluator.adapters import PIAdapter

adapter = PIAdapter("simpleqa")
bundle = await run_evaluation(adapter, client, n_repeats=2,
                               system_prompt=adapter.system_prompt)
```

### `SkillsAdapter`

Wraps any NeMo Skills benchmark as an EvalSource with full observability.

```python
from nemo_evaluator.adapters import SkillsAdapter

adapter = SkillsAdapter("gpqa")
bundle = await run_evaluation(adapter, client, n_repeats=4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark` | `str` | required | Skills benchmark name (e.g., `gpqa`, `aime24`) |
| `split` | `str \| None` | `None` | Dataset split (default from benchmark config) |
| `data_dir` | `str \| None` | `None` | Override data directory |
| `prompt_template` | `str \| None` | `None` | Custom prompt template with `{problem}` placeholder |
| `eval_type` | `str \| None` | `None` | Override scoring type (default from benchmark config) |

### `run_container_eval()`

Run a legacy evaluator container and parse results.

```python
from nemo_evaluator.adapters.container import ContainerConfig, run_container_eval

cfg = ContainerConfig(
    task_type="simple_evals.GPQA_diamond",
    model_url="https://inference-api.nvidia.com/v1",
    model_id="azure/openai/gpt-5.2",
    limit_samples=50,
)
bundle = run_container_eval(cfg, output_dir="./results")
```

### `GymHarness`

Wraps an `EvalEnvironment` for Gym consumption.

```python
from nemo_evaluator.adapters.gym_harness import GymHarness

harness = GymHarness(env)
path = harness.export_jsonl("/tmp/data")
```

## Observability

### `ModelResponse`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Model output text |
| `model` | `str` | Model identifier |
| `finish_reason` | `str` | stop, length, etc. |
| `prompt_tokens` | `int` | Input tokens |
| `completion_tokens` | `int` | Output tokens |
| `total_tokens` | `int` | Total tokens |
| `reasoning_tokens` | `int` | Reasoning tokens (if available) |
| `latency_ms` | `float` | Request latency |
| `raw_response` | `dict` | Full API response |

### `StepRecord`

Complete record for one seed → model → verify cycle.

| Field | Type | Description |
|-------|------|-------------|
| `problem_idx` | `int` | Problem index |
| `repeat` | `int` | Repeat index |
| `prompt` | `str` | The prompt sent |
| `expected_answer` | `str` | Ground truth |
| `model_response` | `ModelResponse \| None` | Full model response |
| `reward` | `float` | Verification score |
| `extracted_answer` | `str \| None` | Extracted from response |
| `scoring_method` | `str` | Method used for scoring |
| `scoring_details` | `dict` | Scoring intermediaries |
| `seed_ms` | `float` | Time to seed |
| `model_ms` | `float` | Time for model call |
| `verify_ms` | `float` | Time to verify |
| `total_ms` | `float` | Total step time |
| `model_error` | `str \| None` | Error if model call failed |

## CLI Commands

| Command | Description |
|---------|-------------|
| `nel run` | Run evaluation (benchmark name, YAML config, or adapter URI) |
| `nel serve` | Start HTTP server for an environment |
| `nel validate` | Quick validation of a benchmark |
| `nel list-environments` | Show registered built-in benchmarks |
| `nel list-harnesses` | Show known legacy evaluator container harnesses |
| `nel list-skills` | Show available NeMo Skills benchmarks |
| `nel container-eval` | Run evaluation via legacy evaluator container |
| `nel regression` | Compare two evaluation bundles |
| `nel slurm eval` | Generate SLURM sbatch for distributed eval |
| `nel slurm serve` | Generate SLURM sbatch for environment server |
| `nel slurm merge` | Merge sharded evaluation results |

### `nel run`

```
nel run [BENCHMARK_OR_CONFIG]
    --benchmark, -b TEXT     Benchmark name
    --repeats, -n INT        Repeats per problem [1]
    --max-problems, -m INT   Limit problem count
    --model-url TEXT         Model API base URL
    --model-id TEXT          Model identifier
    --system-prompt TEXT     System prompt
    --adapter TEXT           Adapter URI (gym://host:port, pi://env, skills://benchmark)
    --output-dir, -o TEXT    Output directory [./eval_results]
    --no-progress            Disable progress bar
```

### `nel serve`

```
nel serve
    --benchmark, -b TEXT     Benchmark to serve
    --port INT               Port [9090]
    --host TEXT              Host [0.0.0.0]
    --gym-compat             Enable Gym-compatible endpoints
    --export-data TEXT       Export JSONL to path and exit
```

### `nel regression`

```
nel regression BASELINE CANDIDATE
    --threshold FLOAT        Max acceptable drop [0.05]
    --strict                 Exit non-zero on regression
    --output TEXT             Write JSON report
```
