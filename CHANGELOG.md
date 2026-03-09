# Changelog

## 0.6.0 (2026-03-09)

### Feature Parity

- **VLM support**: `VLMSolver` with `ModelClient.vlm_chat()` for vision-language benchmarks. `SeedResult.images` field, `endpoint_type: vlm` in config. Local file paths auto-converted to base64 data URIs.
- **Embedding support**: `EmbeddingSolver` with `ModelClient.embed()`/`embed_batch()`. `CrossEncoderSolver` for reranking via `/v1/rerank`.
- **MTEB integration**: `MTEBEnvironment` wraps the MTEB library for embedding evaluation. Usage: `mteb://STSBenchmark`. Runs in a background thread to avoid blocking the event loop.
- **ContainerEnvironment**: Run legacy eval-factory containers as opaque environments. Usage: `container://nvcr.io/eval-factory/mmlu:latest#mmlu_pro`. Async subprocess execution with timeout.
- **Multi-node vLLM**: `RayMultiNodeDeployment` for tensor + pipeline parallelism across nodes. Config fields: `pipeline_parallel_size`, `num_nodes`.
- **WandB/MLflow export**: Plugin-based export system. `WandBExporter` and `MLflowExporter` push scores and artifact bundles to experiment trackers. Config: `output.export: [wandb, mlflow]`.
- **SLURM container mode**: `ClusterConfig.container_image` generates `srun --container-image` commands in sbatch scripts. Container mounts, env vars, and home mount configurable.
- **BYOB containerization**: `nel package -m my_bench/ -t my-bench:latest` generates a Dockerfile and builds a container image for BYOB benchmark modules.
- **Persistent config**: `nel config set/get/show/unset` manages `~/.config/nemo-evaluator/config.yaml`.
- **CLI overrides**: `nel eval run config.yaml -O model.temperature=0.5` applies dot-style overrides before Pydantic validation.
- **Reasoning trace handling**: `ModelConfig.reasoning_pattern` strips `<think>...</think>` tokens before scoring.
- **SLURM auto-resume**: `ClusterConfig.auto_resume` resubmits failed jobs via `afternotok` dependency chain, up to `max_resume_attempts`.
- **Telemetry**: Opt-in usage analytics with `TelemetryHandler`. Levels: off/minimal/default. Controlled via env var or persistent config.
- **CSV/TSV datasets**: `@benchmark` dataset specs now accept `.csv` and `.tsv` files via `csv.DictReader`.

### Bug Fixes (27 total across 3 audit rounds)

**Critical (13):**
- Fixed `_NELEncoder.encode()` calling `asyncio.run()` inside a running event loop (MTEB crash).
- Fixed `set -euo pipefail` making SLURM auto-resume dead code; changed to `set -uo pipefail` with explicit `NEL_EXIT_CODE` tracking.
- Fixed `embed_batch()` having zero retry logic; extracted shared `_post_with_retry()` used by all HTTP methods.
- Fixed `ContainerEnvironment.run_batch()` using synchronous `subprocess.run()` inside async; replaced with `asyncio.create_subprocess_exec()`.
- Fixed SLURM `$SRUN_PREFIX` defined but never used in task commands.
- Fixed `-o` CLI flag collision between `--output-dir` and `--override`; override now uses `-O`.
- Fixed SLURM auto-resume grep for `"_failed": false` which never exists in bundles; now checks `checkpoint.json`.
- Fixed auto-resume treating missing `checkpoint.json` as "all completed"; now correctly triggers resubmit.
- Fixed local runner auto-service dropping `pipeline_parallel_size`/`num_nodes`.
- Fixed `_parse_response` crash on `None` content from model refusals.
- Fixed SLURM health wait silently continuing with dead services; now exits on timeout.
- Fixed API key leaking to disk via batch config dict.
- Fixed `write_all()` crashing with `KeyError: 'run_id'` on Container/MTEB bundles.

**Architectural (4):**
- Fixed service handles' runtime URLs ignored in favor of static config resolution; now prefers actual URL from running service.
- Fixed judge clients crossing `asyncio.run()` boundaries; now created fresh per-benchmark inside the async function.
- Fixed `env.verify()` exceptions silently dropping results; now catches and records zero-reward with error details.
- Wired dead `BenchmarkConfig.fewshot` field through to environment factories.

**Design (10):**
- Refactored `chat()` to use shared `_post_with_retry()`, eliminating duplicated 50-line retry loop.
- Fixed inconsistent latency measurement between `chat()` and `vlm_chat()`.
- Wrapped `mteb.MTEB.run()` in `loop.run_in_executor()` to avoid blocking the event loop.
- Fixed telemetry `emit()` mutating caller's event object; now uses `dataclasses.replace()`.
- Added retry + semaphore to `CompletionSolver` and `CrossEncoderSolver` via `ModelClient._post_with_retry()`.
- Fixed `package_cmd.py` single-file module creating wrong directory structure for Python imports.
- Fixed WandB table column mismatch across benchmarks with different metrics; now uses union of all keys.
- Fixed container `pre_cmd` pattern where `exec "$@"` had no arguments.
- Fixed `GymEnvironment.dataset_size()` returning -1 causing silent empty evaluation; now raises `ValueError`.
- Fixed `_resolve_services()` silently dropping `pipeline_parallel_size`/`num_nodes`, `RayMultiNodeDeployment._fallback` not initialized, and fragile `_parse_results()` fallback logic.

### Concurrency and Configuration

- **Unified concurrency control**: `BenchmarkConfig.max_concurrent` (default 32) is now the single knob for parallelism. It is passed to both `ModelClient` (HTTP semaphore) and `run_evaluation` (task semaphore) with the same value, eliminating the prior mismatch where 32 tasks competed for 8 HTTP slots. Configurable per-benchmark in YAML or via `-O benchmarks.0.max_concurrent=64`.
- **`reasoning_pattern` in advanced mode**: `ServiceConfig` now supports `reasoning_pattern` for stripping reasoning tokens (e.g. `<think>...</think>`) in multi-service configurations, not just simple mode.

## 0.5.0 (2026-03-07)

### CLI Redesign

- **`nel eval` command group**: `run`, `status`, `stop`, `report` subcommands replace `nel run`, `nel slurm eval`, `nel harness run`, and `nel container-eval`.
- **`nel list`**: Unified benchmark discovery across built-in, NeMo Skills, and lm-eval sources. Replaces `nel list-environments`, `nel list-harnesses`, `nel list-skills`.
- **`nel serve`**: Flat command for serving benchmarks as HTTP endpoints.
- Removed all legacy commands: `nel run`, `nel slurm`, `nel harness`, `nel container-eval`, `nel list-environments`, `nel list-harnesses`, `nel list-skills`.

### Unified Config Schema

- **Two-tier `EvalConfig`**: Simple mode (`model:` + `benchmarks:`) for quick evaluations, advanced mode (`services:` + `benchmarks:`) for multi-model setups with managed infrastructure.
- **`ClusterConfig`**: SLURM, Docker, and local execution from a single config file.
- **`OutputConfig`**: Multi-format report generation (HTML, Markdown, CSV, JSON, LaTeX).
- Removed old `evaluation: { tasks: [] }` config format and `config_schema.py`.

### Evaluation Orchestration

- **`eval/local_runner.py`**: Service lifecycle management (vLLM, SGLang, Gym servers) with health polling.
- **`eval/slurm_gen.py`**: Self-contained sbatch script generation from `EvalConfig`.
- **`eval/ssh.py`**: Remote SLURM submission via SSH.
- **`eval/lifecycle.py`**: Executor-aware process management (`status`/`stop`) for local, SLURM, and Docker.

### Resilience and Resume

- **Checkpoint-based resume**: `run_local()` uses `CheckpointManager` to track per-benchmark completion. Completed benchmarks are skipped on re-run; failed benchmarks are logged and isolated.
- **`--resume` flag**: `nel eval run config.yaml --resume` picks up from where a prior run left off. Without `--resume`, checkpoints are cleared for a fresh start.
- **Failure isolation**: A single benchmark failure no longer kills the entire suite. Failed benchmarks are recorded and the run continues to the next benchmark.

### Regression Analysis

- **Mann-Whitney U p-values**: `compare_runs()` now loads per-sample rewards from `results.jsonl` and computes a two-sided Mann-Whitney U test via scipy. Each score delta entry includes `p_value` and `significant` fields.
- scipy is an optional dependency under `[stats]` — p-values are `null` if scipy is not installed.

### Bug Fixes

- Fixed `str.format()` crash in sbatch template generation.
- Fixed PID file written before output directory exists.
- Fixed Gym service template generating nonexistent CLI subcommand.
- Fixed shell injection via unescaped quotes in system prompts.
- Fixed `num_examples` silently dropped for all environment types.
- Renamed `models.EvalConfig` → `RunSnapshot` to avoid collision with `eval.config.EvalConfig`.

### Cleanup

- Deleted dead modules: `cli/run.py`, `cli/slurm.py`, `cli/harness.py`, `config_schema.py`, `environments/definitions.py`.
- Removed all backward-compatibility aliases and deprecation stubs.
- Updated all documentation and examples to use new CLI and config format.

## 0.4.0 (2026-03-07)

### Architecture

- **Everything is an Environment**: Collapsed adapters, harnesses, and environments into one `EvalEnvironment` base class. Gym, Skills, PI, lm-eval -- all are environments.
- **`@benchmark` + `@scorer` API**: Universal benchmark definition mechanism. All 11 built-in benchmarks use it. External BYOB uses the same API.
- **Solver protocol**: `ChatSolver`, `CompletionSolver`, `AgentSolver`. Eval loop calls `solver.solve()` instead of hardcoded `client.chat()`. Unblocks agent benchmarks.
- **Executor layer**: `LocalExecutor`, `DockerExecutor`, `SlurmExecutor` manage full lifecycle (deploy model, run eval, collect results, tear down).
- **Model deployment**: `NIMDeployment`, `VLLMDeployment`, `ProcessModelDeployment`, `APIDeployment` with health polling and graceful shutdown.
- **Judge post-processing**: Eval loop optionally runs LLM-as-judge scoring after `verify()`.

### Benchmarks (11 built-in)

MMLU, MMLU-Pro, MATH-500, GPQA Diamond, GSM8K, DROP, MGSM, TriviaQA, HumanEval (Docker sandbox), SimpleQA (judge), HealthBench (judge).

### Environments

- `GymEnvironment` + `ManagedGymEnvironment` (server lifecycle)
- `SkillsEnvironment` (auto-prepares datasets)
- `LMEvalEnvironment` (generate_until tasks)
- `PIEnvironment` (Prime Intellect verifiers)

### Scoring

- Scoring package with dedicated modules: `text.py`, `pattern.py`, `sandbox.py`, `judge.py`, `json_schema.py`
- Benchmarks own their scoring via `@scorer` functions calling reusable primitives
- Primitives: `exact_match`, `fuzzy_match`, `multichoice_regex`, `answer_line`, `numeric_match`, `code_sandbox`, `needs_judge`
- LLM-as-judge: `JudgeScoringConfig`, `judge_score`, `needs_judge` signal
- JSON schema validation: `extract_json`, `validate_json_schema`

### Eval Loop

- Async parallel with semaphore-bounded concurrency (default 32)
- Back-pressure pipeline: seeds 2x buffer, never loads full dataset into task queue
- Proper resource cleanup: `env.close()` and `solver.close()` in finally blocks

### Security

- Docker sandbox: code piped via stdin (not `-c`), `--pids-limit`, `--read-only`, `--security-opt no-new-privileges`
- API keys passed via env vars (not CLI args) in AgentSolver
- Prompt template KeyError warnings (not silent swallows)

### Tests

- Integration tests for `run_evaluation()` end-to-end
- Golden tests for all scoring primitives
- Scorer import tests for each built-in benchmark
- 144 total tests

## 0.3.0 (2026-02-25)

Initial architecture with unified eval loop, harness adapters, and observability.
