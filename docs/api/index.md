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

    async def seed(self, idx: int) -> SeedResult:
        ...

    async def verify(self, response: str, expected: str, **metadata) -> VerifyResult:
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
| `temperature` | `float \| None` | `None` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `int \| None` | `None` | Max completion tokens |
| `top_p` | `float \| None` | `None` | Nucleus sampling threshold (0.0-1.0) |
| `seed` | `int \| None` | `None` | RNG seed for reproducibility |
| `stop` | `list[str] \| None` | `None` | Stop sequences |
| `frequency_penalty` | `float \| None` | `None` | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | `float \| None` | `None` | Presence penalty (-2.0 to 2.0) |
| `timeout` | `float` | `120.0` | Request timeout in seconds |
| `max_concurrent` | `int` | `8` | Max parallel requests |

```python
from nemo_evaluator.engine import ModelClient

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
    env: EvalEnvironment,
    solver: Solver,
    n_repeats: int = 1,
    max_problems: int | None = None,
    config: dict[str, Any] | None = None,
    progress: ProgressTracker | None = None,
    problem_range: tuple[int, int] | None = None,
    max_concurrent: int = 32,
    judge_client: Any = None,
    shard_info: tuple[int, int] | None = None,
) -> dict[str, Any]:
```

| Parameter | Description |
|-----------|-------------|
| `env` | `EvalEnvironment` instance |
| `solver` | `Solver` instance (e.g. `ChatSolver`, `HarborSolver`, `GymSolver`) |
| `n_repeats` | Number of times to evaluate each problem |
| `max_problems` | Limit to first N problems |
| `config` | Metadata included in the output bundle |
| `progress` | Progress tracker (default: no-op) |
| `problem_range` | `(start, end)` for sharded execution |
| `max_concurrent` | Max parallel solve tasks (default: 32) |
| `judge_client` | Optional `ModelClient` for LLM-as-judge post-processing |
| `shard_info` | `(shard_idx, total_shards)` — auto-computes `problem_range` if not set |

Returns a bundle dict containing metrics, results, config, and artifacts.

### `write_all()`

Write all artifacts to disk.

```python
from nemo_evaluator.engine import write_all

write_all(bundle, "./eval_results")
```

Writes: `eval-*.json` (bundle), `results.jsonl`, `trajectories.jsonl`, `runtime_stats.json`, `failure_analysis.json`.

### `compare_runs()`

Compare two evaluation bundles for regression.

```python
from nemo_evaluator.engine import compare_runs

report = compare_runs("baseline/eval-*.json", "candidate/eval-*.json")
```

## Environment Integrations

### `GymEnvironment`

HTTP client for consuming `nel serve` endpoints or Gym resource servers.

```python
from nemo_evaluator.environments.gym import GymEnvironment
from nemo_evaluator.solvers import ChatSolver

env = GymEnvironment("http://localhost:9090")
solver = ChatSolver(client)
bundle = await run_evaluation(env, solver, n_repeats=4)
```

### `SkillsEnvironment`

Wraps any NeMo Skills benchmark as an `EvalEnvironment`.

```python
from nemo_evaluator.environments.skills import SkillsEnvironment

env = SkillsEnvironment("gpqa")
solver = ChatSolver(client)
bundle = await run_evaluation(env, solver, n_repeats=4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark` | `str` | required | Skills benchmark name (e.g., `gpqa`, `aime24`) |
| `split` | `str \| None` | `None` | Dataset split (default from benchmark config) |
| `data_dir` | `str \| None` | `None` | Override data directory |
| `prompt_template` | `str \| None` | `None` | Custom prompt template with `{problem}` placeholder |
| `eval_type` | `str \| None` | `None` | Override scoring type (default from benchmark config) |

### `VLMEvalKitEnvironment`

Wraps VLMEvalKit datasets as an `EvalEnvironment`. Supports MCQ, VQA, and Y/N dataset types.

```python
from nemo_evaluator.environments.vlmevalkit import VLMEvalKitEnvironment

env = VLMEvalKitEnvironment("MMBench_DEV_EN")
solver = VLMSolver(client)
bundle = await run_evaluation(env, solver, n_repeats=1)
```

### `generate_app()`

Serve an `EvalEnvironment` as a Gym-compatible HTTP endpoint.

```python
from nemo_evaluator.serving.app import generate_app

app = generate_app(env, gym_compat=True)
# Use with uvicorn: uvicorn.run(app, port=9090)
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
| `nel eval run` | Run evaluation (benchmark name or YAML config) |
| `nel eval report` | Generate evaluation report |
| `nel eval merge` | Merge sharded evaluation results |
| `nel serve` | Start HTTP server for an environment |
| `nel validate` | Quick validation of a benchmark |
| `nel list` | Show available benchmarks and environments |
| `nel regression` | Compare two evaluation bundles |
| `nel config` | Persistent user config |
| `nel package` | Containerize BYOB benchmark |

### `nel eval run`

```
nel eval run [CONFIG_FILE]
    --bench, -b TEXT         Benchmark name
    --repeats, -n INT        Repeats per problem [1]
    --max-problems, -m INT   Limit problem count
    --model-url TEXT         Model API base URL
    --model-id TEXT          Model identifier
    --system-prompt TEXT     System prompt
    --adapter TEXT           Adapter URI (gym://host:port, skills://benchmark)
    --output-dir, -o TEXT    Output directory [./eval_results]
    --resume                 Resume partially completed suite
    -O, --override TEXT      Dot-path config overrides (e.g. services.model.model=foo)
    --no-progress            Disable progress bar
```

### `nel serve`

```
nel serve
    --bench, -b TEXT         Benchmark to serve
    --port INT               Port [9090]
    --host TEXT              Host [0.0.0.0]
    --gym-compat             Enable Gym-compatible endpoints
    --export-data TEXT       Export JSONL to path and exit
```

### `nel validate`

```
nel validate
    --bench, -b TEXT         Benchmark to validate
    --samples INT            Number of samples [10]
```

### `nel list`

```
nel list
    --source TEXT            Filter by source (e.g., lm-eval)
```

### `nel eval report`

```
nel eval report RESULTS_DIR
    --format, -f TEXT        Output format (markdown, html, csv, json, latex)
    --output, -o TEXT        Output report path
    --all-formats            Generate all formats
```

### `nel eval merge`

```
nel eval merge OUTPUT_DIR
    --repeats, -n INT        Override n_repeats (auto-detected if omitted)
```

### `nel regression`

```
nel regression BASELINE CANDIDATE
    --threshold FLOAT        Max acceptable drop [0.05]
    --strict                 Exit non-zero on regression
    --output TEXT             Write JSON report
```
