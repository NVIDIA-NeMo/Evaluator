# Changelog

## 0.13.0 (unreleased)

### Shared Metric Contract

- Added public `MetricInput -> MetricResult` scorer/metric runtime types and `ScorerFunctionMetric`.
- Extended BYOB `@scorer` with typed scorer metadata and `to_metric()` while preserving current dict scorer behavior.
- Added optional `config_schema` support for typed scorer configs while keeping raw dict configs as the default.
- Split typed scorer config binding into strict `bind(config=ConfigModel(...))` and coercive `bind_raw_config(config={...})` paths.
- Added `@scorer` support for class-based `Metric` objects.
- Added a reusable undecorated `ExactMatchMetric` and an `ExactMatchScorer` BYOB wrapper.

### Adapter Proxy (Breaking — replaces LiteLLM)

- **LiteLLM removed**: The `litellm` dependency, `proxy` and `proxy-full` extras, and `litellm_settings` config field are all removed. The adapter proxy is now built-in with zero external proxy dependencies.
- **New adapter system**: In-process Starlette proxy server with 13 composable async interceptors, replacing both the LiteLLM subprocess and the 4 LiteLLM `CustomLogger`-based interceptors.
- **Interceptor pipeline**: `RequestInterceptor → RequestToResponseInterceptor → ResponseInterceptor` chain with short-circuit support (cache hits), `best_effort` error handling, and `ContextVar`-based per-request context.
- **Built-in interceptors**: `endpoint`, `drop_params`, `modify_tools`, `turn_counter`, `system_message`, `payload_modifier`, `raise_client_errors`, `caching`, `log_tokens`, `response_stats`, `reasoning`, `progress_tracking`, `logging`.
- **Disk cache**: Async SQLite cache with WAL mode, canonical cache keys, and graceful degradation on corruption. Requires node-local storage (not NFS/Lustre).
- **Explicit proxy config fields**: `proxy.request_timeout`, `proxy.max_retries`, `proxy.retry_on_status`, `proxy.max_concurrent_upstream` replace the opaque `litellm_settings` dict.

#### Migration guide

| Old config | New config |
|---|---|
| `pip install -e ".[proxy]"` | No extra needed — built-in |
| `proxy.litellm_settings.request_timeout: 600` | `proxy.request_timeout: 600` |
| `proxy.litellm_settings.num_retries: 3` | `proxy.max_retries: 3` |
| `proxy.litellm_settings.max_parallel_requests` | `proxy.max_concurrent_upstream` |
| `proxy.litellm_settings.set_verbose: true` | `proxy.verbose: true` |

## 0.12.0 (2026-03-20)

### Config Schema v2 (Breaking)

- **Fully explicit configuration**: Replaced the two-tier simple/advanced config mode with a single explicit format. All architectural choices (services, solvers, sandboxes, metrics) must be spelled out — no sugar, no defaults, no shorthand.
- **`services:` dict**: Named service definitions with discriminated `type` field (`api`, `vllm`, `sglang`, `trt_llm`, `gym_resource`, `nat_agent`, `custom`). Each service declares its full endpoint URL, wire protocol, generation config, and interceptors.
- **`ExternalApiService`**: `url` is the full endpoint path (e.g., `https://api.example.com/v1/chat/completions`), `protocol` declares the wire format independently (supports experimental endpoints with decoupled URL and protocol).
- **`SolverConfig` discriminated union**: `simple`, `harbor`, `tool_calling`, `gym_delegation`, `openclaw`, `container`, `custom` — each with explicit `service` reference.
- **`SandboxConfig` discriminated union**: `docker`, `apptainer`, `ecs_fargate`, `slurm`, `local`, `none`, `custom` — referenceable by name from top-level `sandboxes:` dict or inline in benchmarks.
- **`MetricConfig` discriminated union**: `default`, `judge`, `reward_model`, `pairwise_judge`, `ensemble_judge`, `custom`. Supports multi-judge and multi-service evaluation.
- **`ClusterConfig`**: `local`, `slurm`, `docker` types. SLURM clusters use `node_pools:` dict for multi-topology resource placement. Services and sandboxes reference pools by name.
- **Self-judge prevention**: `JudgeMetric` validates that judge service differs from solver service unless `allow_self_judge: true`.
- **Dependency graph**: Services declare `depends_on:` for startup ordering with cycle detection.
- **Cross-reference validation**: All service, sandbox, and node pool references validated at parse time.
- **Timeout hierarchy validation**: Warns if combined service startup + benchmark timeouts exceed SLURM walltime.
- **`GenerationConfig`**: Per-service generation parameters (temperature, top_p, max_tokens) with range validation and `merge_onto()` for field-by-field overrides.
- **`InterceptorConfig`**: Per-service LiteLLM interceptor configuration.
- **Extensibility**: `CustomService`, `CustomSandbox`, `CustomSolverConfig`, `CustomMetric` with `class_path` + `config` for plugin support.
- All 25 example configs rewritten to the new format.

### Eval Sharding

- **SLURM array jobs**: `SlurmCluster.shards` field generates `#SBATCH --array` directives. Each array task evaluates a disjoint slice of the dataset via `NEL_SHARD_IDX` / `NEL_TOTAL_SHARDS` env vars.
- **`nel eval merge`**: New CLI command to merge results from sharded evaluation runs. Auto-discovers benchmarks across `shard_N/` directories, detects `n_repeats` from bundles, deduplicates overlapping results.
- **Local runner sharding**: `local_runner.py` reads shard env vars automatically and passes `problem_range` to the eval loop.
- **Validation**: `shards` is mutually exclusive with heterogeneous SLURM jobs (multiple node pools) and `auto_resume`.
- New example: `15a_slurm_gsm8k_vllm_sharded.yaml`.

### Test Fixes

- Fixed `MockSandboxManager.resolve_spec()` missing `base_override` keyword argument.
- Fixed `FixturedEnvironment` missing `prepare()` and `image_build_requests()` methods.
- Fixed aiohttp mock patterns in `test_gym_integration.py` (context manager protocol).
- Fixed `test_model_client.py` event loop requirement for `_get_client()`.
- Fixed `test_solvers.py` NatSolver test still patching `httpx` instead of `aiohttp`.
- Removed unused imports flagged by ruff across test and source files.

### Dependencies

- Replaced `httpx` with `aiohttp` across all HTTP client code (model client, gym environment, NAT solver).

## 0.11.0 (2026-03-19)

### LiteLLM Proxy

- **Optional local proxy**: New `model.proxy` config section starts a LiteLLM proxy subprocess in front of the upstream LLM endpoint. All agent traffic is routed through the proxy, giving full observability into request/response payloads without modifying agent code. Verbose mode (`verbose: true`) streams proxy logs to stderr in real time.
- **Interceptor framework**: Custom `litellm.integrations.custom_logger.CustomLogger` subclasses can be loaded via `proxy.interceptors`. Built-in `log_tokens` interceptor included. Interceptors live in `nemo_evaluator/interceptors/` and are resolved by dotted path.
- **ECS reverse tunnel**: When a proxy is active and the sandbox backend is ECS Fargate, a reverse SSH tunnel (`-R 4000:127.0.0.1:4000`) is automatically established so the container's `localhost:4000` reaches the host-side proxy.

### Inspect AI Exporter

- **`inspect` export plugin**: Converts NEL evaluation results to `inspect_ai`-compatible `EvalLog` format. ATIF trajectory steps are mapped to `ChatMessage` sequences; rewards become `Score` objects. Output as `*_inspect.json` or `*_inspect.eval`.
- **In-memory bundle forwarding**: `_generate_reports` now receives in-memory bundles (with full `_results` including trajectories) instead of reloading from disk, ensuring exporters have access to complete per-sample data.

### Harbor Solver

- **API key via `llm_kwargs`**: Agents that call litellm directly (e.g. Terminus-2) now receive the API key through `llm_kwargs`, which flows through `LiteLLM.__init__(**kwargs)` → `_build_base_kwargs()` → `litellm.acompletion(api_key=...)`. Previously the key was silently dropped by agents that don't have an explicit `api_key` parameter.
- **`openai/` prefix for custom endpoints**: When `model_url` is set, the model ID is automatically prefixed with `openai/` to give litellm the provider hint it needs for OpenAI-compatible endpoints. Applied uniformly to all Harbor agents.
- **`api_base` parameter alignment**: Harbor agents receive `api_base` (not `model_url`) matching their expected parameter name.
- **OpenHands security analyzer disabled**: `SECURITY_CONFIRMATION_MODE=false` and `SECURITY_ENABLE_SECURITY_ANALYZER=false` are injected by default for OpenHands agents, preventing `security_risk` parameter validation failures with models that don't emit it.
- **LiteLLM noise silenced**: `LITELLM_LOG=ERROR` and `LITELLM_TELEMETRY=false` set by default in both container and host environments.

### New Configs

- **`07a_swebench_harbor_docker.yaml`**: SWE-bench via local Docker sandbox (OpenHands agent).
- **`08_terminalbench_harbor.yaml`**: Terminal-Bench 2.0 via ECS Fargate (Terminus-2 agent) with Inspect AI export.
- **`08a_terminalbench_harbor_docker.yaml`**: Terminal-Bench via local Docker sandbox.
- **`12_pinchbench_openclaw.yaml`**: PinchBench via OpenClaw solver with ECS.
- **`18_humaneval_ecs.yaml`**: HumanEval code execution on ECS Fargate.
- **`19_gym_runtime_ecs.yaml`**: Gym runtime evaluation on ECS.
- **`20_slurm_gym_runtime.yaml`**: Gym runtime on SLURM.
- Config renumbering: 08–16 shifted to 09–17 for consistent ordering.

### Dependencies

- New optional extras: `proxy` (`litellm[proxy]`), `inspect` (`inspect_ai`).

### Fixes

- Fixed Harbor agent API key not reaching litellm for non-InstalledAgent agents (Terminus-2).
- Fixed `OPENAI_API_KEY` environment variable leaking globally and misrouting requests to api.openai.com.
- Fixed ECS image resolution when `image_template` was in the wrong config section.
- Fixed proxy subprocess logs not appearing in terminal (now redirected to stderr when verbose).

## 0.10.0 (2026-03-17)

### Harbor Agent Framework

- **`HarborSolver`**: New solver that runs Harbor agents (OpenHands, Terminus-2, mini-swe-agent, etc.) inside NEL sandboxes via `SandboxEnvironmentAdapter`. Supports any agent registered with Harbor's `AgentFactory`.
- **`SandboxEnvironmentAdapter`**: Wraps NEL's `Sandbox` protocol to satisfy Harbor's `BaseEnvironment` interface. Merges `persistent_env` into every `exec()` call, enabling transparent environment variable injection.
- **`ByobInstalledAgent`**: Adapter for running custom agent directories as Harbor agents.
- **Agent log download**: Automatic retrieval of `/logs/agent/` from containers after agent execution.

### New Sandbox Backends

- **Apptainer sandbox**: `ApptainerSandbox` for HPC environments where Docker is unavailable. Supports SLURM integration via `salloc`/`srun`.
- **ECS Fargate enhancements**: CodeBuild-based image building with ECR content-hash tagging for per-task Docker images. `OutsideEndpoint` mechanism for making host services reachable from containers via SSH tunnels.

### New Environments

- **VLMEvalKit**: `VLMEvalKitEnvironment` for vision-language model evaluation using the VLMEvalKit framework. Usage: `vlmevalkit://MMBench_DEV_EN`.
- **Composite environment**: `CompositeEnvironment` for combining multiple benchmarks into a single evaluation run.
- **XSTest**: Built-in safety evaluation benchmark.

### Solver Refactor

- **`GymSolver`**: Dedicated solver for Gym protocol interactions, extracted from the generic solver.
- **Removed `SandboxedAgentSolver`**: Replaced by the more flexible `HarborSolver` + `SandboxEnvironmentAdapter` pattern.

### Config Overhaul

- Numbered config examples (01–16) with README catalog replacing ad-hoc config files.
- Removed legacy example configs (`multi_benchmark.yaml`, `quick_eval.yaml`, `swebench_*.yaml`, etc.).
- Removed `examples/getting_started.md`, `gym_integration.py`, and other legacy examples.

### Fixes

- Fixed in-container runtime detection for Docker-in-Docker scenarios.
- Fixed Docker image building CI with proper naming and tagging.
- Performance improvements in eval loop and sandbox lifecycle.

## 0.9.0 (2026-03-16)

### Native Gym Servers on SLURM

- **`server_cmd` for gym services**: `ServiceConfig(type="gym")` now honors the `server_cmd` field on SLURM. When provided, the sbatch generator starts the custom command instead of `nel serve`. This makes the integration fully abstract -- any server that speaks `/seed_session` + `/verify` works, whether it's a native Gym resource server, a custom FastAPI app, or anything else.
- **Multi-path health checks**: SLURM health waits for gym services with `server_cmd` now try both `/health` and `/openapi.json`, matching the `ManagedGymEnvironment` fallback behavior for native Gym servers.
- **`scoring_details`-based breakdowns**: The eval loop now produces per-group breakdowns from `scoring_details` fields returned by verify (e.g., per-`db_id` accuracy for Spider2, per-`label` safety rates for XSTest). All aggregation is computed by NEL, not the server.
- **Example configs**: `gym_spider2_lite.yaml`, `gym_xstest.yaml`, `gym_finance.yaml`, and `gym_slurm_suite.yaml` demonstrate running Spider 2.0-Lite, XSTest, and Finance SEC Search as native Gym resource servers.
- **`Dockerfile.gym`**: New container variant (`docker/Dockerfile.gym`) that bundles NEL with the full NeMo Gym repository and all resource server dependencies. On SLURM, the `gym://` scheme now auto-selects this image. CI builds and pushes `${REGISTRY}:${TAG}-gym` via Kaniko.

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

### Config Schema (v1)

- **`EvalConfig`**: `services:` + `benchmarks:` config with managed infrastructure.
- **`ClusterConfig`**: SLURM, Docker, and local execution from a single config file.
- **`OutputConfig`**: Multi-format report generation (HTML, Markdown, CSV, JSON, LaTeX).
- Removed old `evaluation: { tasks: [] }` config format and `config_schema.py`.
- Note: Superseded by Config Schema v2 in 0.12.0.

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
