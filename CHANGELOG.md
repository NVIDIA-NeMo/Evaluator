# Changelog

## 0.3.0 (2026-02-25)

### Architecture

- **StepEnvironment**: New base class for multi-turn / interactive BYOB benchmarks (`reset(idx) -> Observation`, `step(action) -> Observation`). Single-turn `EvalEnvironment` unchanged.
- **Adapter Server proxy**: Reverse proxy (`adapters/proxy.py`) captures per-task model call trajectories when external systems (Gym, Harbor, PI) own the model call. Path-prefix routing: `/problem/{task_id}/v1/...`.
- **Config schema validation**: YAML configs are now validated with Pydantic (`config_schema.py`). Environment variables (`${VAR:-default}`) are expanded. Invalid configs produce clear error messages.
- **Unified verify endpoint**: Server `/verify` auto-detects evaluator and Gym request formats regardless of `--gym-compat` flag. GymAdapter and Gym training can both talk to the same server.
- **Type system consolidation**: Removed duplicate runtime types from `models.py`. `observability/types.py` is the single runtime type system; `models.py` handles config validation and serialized output only.

### Harness adapters

- **lm-eval loglikelihood**: Now raises `UnsupportedTaskTypeError` instead of silently returning garbage. Added `list_generate_tasks()` and `nel harness list --generate-only` to discover chat-API-compatible tasks.
- **Async bridge**: Replaced per-call `asyncio.run()` / `ThreadPoolExecutor` pattern with a dedicated background event loop (`runner/async_bridge.py`). Safe in Jupyter, FastAPI, and nested async contexts.

### Scoring

- **`judge_score()`**: End-to-end async LLM-as-judge scoring with model call, swap-check for position bias detection, and full audit trail.
- **`json_schema.py`**: JSON extraction from model output + schema validation (type, required, properties, items, enum). For function-calling and structured output benchmarks.

### Infrastructure

- **Cache eviction**: LRU eviction with configurable `max_entries` (50K) and `max_size_mb` (2GB). Touch-on-read for LRU ordering.
- **Checkpoint locking**: `fcntl.flock` file locking prevents corruption from concurrent writers. Atomic write via temp file + rename.
- **K8s manifests**: Fixed env var substitution (ConfigMap refs), added init container for merge job dependency, added ConfigMap resource.
- **Docker Compose**: Fixed sharding with explicit per-shard services (each with its own `NEL_SHARD_IDX`).
- **GymAdapter**: Shared `httpx.AsyncClient` with connection pooling (was creating one per request).

### Tests

- 150 tests (up from 112): config schema (8), unified verify (6), StepEnvironment (5), async bridge (3), JSON schema scoring (8), plus existing suite.

## 0.2.0

### Features

- External harness integration: `NelSampler` (simple-evals) and `NelLM` (lm-eval) route model calls through `ModelClient` for observability.
- CLI: `nel harness run`, `nel harness list`, `nel report`, `nel slurm`.
- BYOB benchmarks: gsm8k, triviaqa as reference templates.
- Scoring: `math_equal`, `exact_match`, `extract_answer`, `mcq_score`, `build_judge_prompt`.
- Observability: `StepRecord`, `RuntimeStats`, `FailureReport`, `ArtifactCollector`.
- Response caching, checkpointing, sharding with merge.
- Adapters: Gym, Prime Intellect, NeMo Skills, legacy containers.
- Deployment: Dockerfile, docker-compose, K8s manifests, SLURM scripts, GitLab CI.
