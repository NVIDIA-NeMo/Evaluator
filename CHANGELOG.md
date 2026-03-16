# Changelog

## 0.9.0 (2026-03-16)

### Gym Integration

- **Unified `GymEnvironment`**: Collapsed three Gym client classes (`GymEnvironment`, `NativeGymEnvironment`, and the inner `GymEnvironment` delegate) into a single `GymEnvironment` with an explicit `protocol` parameter (`"evaluator"` or `"native"`). No auto-detection -- the caller chooses the protocol.
- **`GymDataset`**: New helper class that loads Gym JSONL task files with `__len__`/`__getitem__`. Separates file I/O from the HTTP client. Passed to `GymEnvironment` via the `dataset` parameter.
- **Metadata smuggling fix**: The old `NativeGymEnvironment` smuggled `_responses_create_params` through `SeedResult.metadata`, which leaked into step logs and got splat into `verify(**meta)`. The unified `GymEnvironment` keeps metadata clean and looks up the original dataset row by `problem_idx` at verify time.
- **`ManagedGymEnvironment`**: Now accepts `protocol` and `dataset` parameters, passes them to the inner `GymEnvironment`. Health check falls back to `/openapi.json` for native Gym servers that don't expose `/health`.
- **`gym_protocol.py`**: Public helpers `wrap_text_as_gym_response()`, `wrap_text_as_responses_create_params()`, `extract_prompt_from_rcp()`, `messages_from_rcp()`, and `extract_assistant_text()`.
- **Registry**: `gym://` URIs now support `?protocol=native&data=/path.jsonl` query params. Removed the opaque `gym://native:` URI scheme.

### Two-Container Agentic Architecture

- **`StatelessSandbox`**: New sandbox lifecycle strategy for agentic benchmarks (SWE-bench, Harbor). Runs the agent in one container and verification in a fresh container, transferring state via shared volumes and `capture_cmd`/`apply_cmd`.
- **`StatefulSandbox`**: Lazy sandbox acquisition -- sandbox is created on first call to `get_agent_sandbox()` or `get_verify_sandbox()`, avoiding unnecessary container startup for non-sandbox benchmarks.
- **SWE-bench Verified and SWE-bench Multilingual**: Integrated as BYOB benchmarks with configurable `image_template` for per-problem Docker images.
- **Harbor**: Refactored to use the two-container model, making it compatible with all solvers (not just `SandboxSolver`).
- **`SandboxManager.resolve_spec`**: Merge-based strategy -- `image_template` from config overrides the image while preserving other spec fields; volumes are appended rather than replaced.

### Test Suite

- **Comprehensive offline tests**: 293 tests covering environments, solvers, sandbox lifecycle, sandbox manager, eval loop, cross-compatibility, Gym integration, server protocol, sharding, metrics, observability, and more.
- **Golden fixture generation**: `tests/generate_fixtures.py` generates reference responses against a real model API and stores them as JSON fixtures for deterministic offline replay.
- **`FixturedEnvironment` + `CachedSolver`**: Mock implementations in `tests/conftest.py` for offline E2E testing without real model calls.
- **GitLab CI**: Split into `test-offline` (required, no network) and `test-network` (HF downloads, allowed to fail) stages with JUnit reports.

### Module Refactor

Restructured the codebase for clear separation of concerns. All old import paths are **removed** (breaking change).

- **`environments/define.py` → `environments/byob.py`**: Renamed to reflect its purpose (Bring-Your-Own-Benchmark API). Contains `@benchmark`, `@scorer`, `ByobEnvironment`, `BenchmarkDefinition`, and dataset loaders.
- **`environments/server.py` → `serving/app.py`**: New `serving/` package for the FastAPI HTTP server (`generate_app`). Import: `from nemo_evaluator.serving.app import generate_app`.
- **`runner/sandbox_lifecycle.py` → `sandbox/lifecycle.py`**: Sandbox lifecycle strategies moved to the `sandbox/` package where they belong. `NoSandbox`, `StatefulSandbox`, `StatelessSandbox`, `pick_lifecycle` now exported from `nemo_evaluator.sandbox`.
- **`runner/deployment.py` → `eval/deployment.py`**: Model deployment strategies moved next to `eval/config.py`.
- **`runner/solver.py` deleted**: Was a re-export shim. Import solvers from `nemo_evaluator.solvers`.
- **`runner/nat_solver.py` deleted**: Was a re-export shim. Import `NatSolver` from `nemo_evaluator.solvers.nat`.

### Breaking Changes

- **Import paths changed**: `environments.define` → `environments.byob`, `environments.server` → `serving.app`, `runner.sandbox_lifecycle` → `sandbox.lifecycle`, `runner.deployment` → `eval.deployment`. The old modules are deleted, not deprecated.
- **`runner.solver` and `runner.nat_solver` removed**: Import from `nemo_evaluator.solvers` instead.
- **`NativeGymEnvironment` removed**: Use `GymEnvironment(endpoint, protocol="native", dataset=GymDataset(path))`.
- **`gym://native:` URI scheme removed**: Use `gym://host:port?protocol=native&data=/path.jsonl`.

### Documentation

- **Getting Started guide**: Moved workshop files (`workshop.md`, `workshop_trivia.py`, `workshop_mixed.yaml`) into `examples/`. The workshop content is now available as [`examples/getting_started.md`](examples/getting_started.md) with updated paths and a link from the README Quick Start section.

### Fixes

- **`mean_reward` vs `pass@1`**: `mean_reward` is now only emitted as the headline metric when there are fractional (non-binary) rewards. Binary benchmarks correctly report `pass@1`.
- **`SandboxSpec` priority**: User-configured `image_template` now overrides `seed.sandbox_spec.image` instead of being ignored.
- **`GymEnvironment.dataset_size`**: Catches generic `Exception` instead of only `httpx` errors, preventing crashes on connection refused.
- **Python 3.12 compatibility**: Fixed `tempfile.mkdtemp` `mode` parameter incompatibility in `StatelessSandbox`.
- **`FixturedEnvironment.verify`**: Matches on both `expected_answer` and `model_response` to prevent collisions in multi-problem fixtures.

## 0.8.0 (2026-03-11)

### NAT Agent Integration

- **`NatSolver`**: New solver that communicates with NeMo Agent Toolkit agents via the `/generate/full` SSE endpoint. Sends task prompts, collects full trajectories (LLM calls, tool invocations, results), converts them to PinchBench-compatible transcript format. Works with any benchmark -- set `endpoint_type: nat` in config.
- **Trajectory conversion**: NAT `IntermediateStep` events (LLM_END, TOOL_START, TOOL_END) are mapped to the OpenClaw-style transcript format for PinchBench grade functions.

### PinchBench

- **New built-in benchmark**: PinchBench agentic task benchmark with 23 tasks. Tasks with embedded `grade()` functions are scored automatically; tasks without use LLM-as-judge via the `needs_judge` mechanism.
- **Judge-scored tasks**: PinchBench populates `expected_answer` with grading criteria for non-automated tasks, delegating scoring to the judge pipeline.

### lm-eval Improvements

- **IFEval scoring fix**: Added `prompt_level_strict_acc` and `prompt_level_loose_acc` to the lm-eval reward metric key lookup. Handle list-valued metrics (averaged).

### Fixes

- **HumanEval entry_point**: Fixed metadata flow -- `entry_point` is now stored as `_entry_point` to avoid being stripped by the `@benchmark` decorator. `code_sandbox` scorer retrieves it correctly.

## 0.7.0 (2026-03-09)

### Sandbox Orchestration

Per-problem isolated execution for agent evaluations and code execution. This is the major feature of 0.7.0, adding the missing "problem-level" isolation layer.

- **`sandbox/` package**: New async `Sandbox` protocol with `exec()`, `upload()`, `download()`, `start()`, `stop()`. Infrastructure-only — the sandbox knows nothing about agents or evaluation logic.
- **Three backends**:
  - `DockerSandbox` — bridge network by default, per-task images, `host.docker.internal` rewriting for model server access
  - `SlurmSandbox` — Pyxis/Enroot with multiplexed container slots per node
  - `LocalSandbox` — async subprocess in temp dir, no isolation, for development
- **Per-task images**: `SandboxSpec` in `SeedResult` allows environments to specify different container images per problem (critical for SWE-bench where every problem has its own image). Fallback chain: `sandbox_spec` → `image_template` → `default_image`.
- **`SandboxManager`**: Concurrency semaphore, bulk pre-pull via `env.sandbox_specs()`, emergency cleanup (`atexit` + `SIGTERM`/`SIGINT` handlers), SLURM round-robin node multiplexing.
- **Eval loop integration**: `run_evaluation()` accepts optional `sandbox_manager` and `model_url`. Sandbox acquired per-problem, passed through `solve(sandbox=)` and `verify(sandbox=)`, released in `finally`.
- **`SandboxedAgentSolver`**: New solver that runs agents inside per-problem sandboxes. Supports exec-server mode (no entrypoint → `sandbox.exec()`) and agent-server mode (entrypoint starts HTTP agent → solver connects via `container_ip`).
- **`code_sandbox_async()`**: Async scorer that executes code inside an existing sandbox via the `Sandbox` protocol. Legacy synchronous `code_sandbox()` retained for backward compatibility.
- **`SandboxConfig`**: New config section under `BenchmarkConfig.sandbox` — `backend`, `image`, `image_template`, `memory`, `cpus`, `timeout`, `concurrency`, `network`, `sandbox_nodes`, `slots_per_node`.
- **SLURM sandbox nodes**: sbatch generator allocates extra nodes for sandboxes, generates pre-pull step, passes node list via `$NEL_SANDBOX_NODES`.

### Breaking Changes

- **`verify()` signature**: All `EvalEnvironment.verify()` implementations now accept an optional `sandbox: Sandbox | None` parameter before `**metadata`. Existing environments that don't use sandboxes are unaffected (the parameter defaults to `None`), but subclasses must update their signature.
- **`SeedResult` extended**: New optional `sandbox_spec: SandboxSpec | None` field. Backward compatible (defaults to `None`).
- **`EvalEnvironment` extended**: New optional `sandbox_specs()` method returns `list[SandboxSpec] | None` for bulk pre-pull. Default returns `None`.
- **`SeedSessionResponse` extended**: Gym protocol gains optional `sandbox_spec` dict field for remote sandbox requirements.

### Config

```yaml
benchmarks:
  - name: gym://swebench-server
    sandbox:
      backend: docker
      image_template: "swebench/sweb.eval.x86_64.{instance_id}:latest"
      concurrency: 8
      memory: 4g
      network: bridge
```

### Documentation

- New [sandbox architecture](docs/architecture/sandbox.md) doc with design principles, backend comparison, configuration examples, SWE-bench integration example, and instructions for adding new backends.
- Updated architecture index with sandbox subgraph in system diagram, package map entry, and eval flow sequence diagram showing sandbox acquire/release.
- Updated README with sandbox section and `SandboxedAgentSolver` in solver table.

## 0.6.0 (2026-03-07)

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

### SLURM Container Architecture

- **Per-benchmark container resolution**: The SLURM sbatch generator now automatically selects the correct container image per benchmark based on its URI scheme. `lm-eval://` tasks run in the `-lm-eval` image, `skills://` in `-skills`, etc. No more single `SRUN_PREFIX` for all tasks.
- **Container image mapping**: `resources/containers.toml` defines scheme-to-image mappings. Users can override via `cluster.container_image` (single image for all) or via the mapping.
- **Deployment containers**: Model servers (vLLM, SGLang) now run inside their own deployment containers when in SLURM mode.
- **Pre-built images via CI**: GitLab CI builds 5 container variants (base, lm-eval, skills, mteb, full) using Kaniko, pushed to the project registry.

### Executor Abstraction

- **`executors/` package**: Proper `Executor` protocol with `run()`, `status()`, `stop()`, `detect()`. One class per backend: `LocalExecutor`, `DockerExecutor`, `SlurmExecutor`. Registry via `get_executor(name)` and `detect_executor(output_dir)`.
- **CLI dispatch via executors**: `nel eval run/status/stop` now dispatches through the executor registry instead of if/elif trees. Adding a new backend (e.g. Kubernetes) requires only a new executor class and a registry entry.
- **Deleted `eval/lifecycle.py`**: All lifecycle management (PID files, docker inspect, squeue) absorbed into the executor classes.

### Architecture Simplification

- **Breaking**: Removed `adapters/` module (`ContainerConfig`, `GymHarness`, `ProxyServer`). Replaced by `environments/container.py` and `environments/server.py`.
- **Breaking**: Removed `gym-managed://` URI scheme. Use `gym://name` instead -- auto-detects `host:port` (remote) vs bare name (managed). `ManagedGymEnvironment` still exists internally.
- Moved `DeployConfig` from `executors/base.py` to `runner/deployment.py`.
- Extracted `EvalConfig.resolved_services()` from duplicated logic in `local_runner.py` and `slurm_gen.py`.
- Collapsed `_TASK`/`_TASK_NO_CONTAINER` and `_VLLM_SERVICE`/`_SGLANG_SERVICE` into shared templates in `slurm_gen.py`.
- **Dual container mode**: `ClusterConfig.env_mode` supports `"colocated"` (default, orchestrator + env in same container) and `"separated"` (env as Gym server, orchestrator in base image).
- Removed empty `contrib/` and `harnesses/` directories.

### URI Scheme Consistency

- **Breaking**: `lm-eval/task` renamed to `lm-eval://task`. All environment types now use the consistent `scheme://task` URI syntax. No more `/` vs `://` inconsistency.

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
- **Execution modes**: Local, Docker, and SLURM execution via `cluster.type` config. Lifecycle management (status/stop) for all modes.
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
