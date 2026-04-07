# Architecture

## System Overview

```{mermaid}
flowchart TB
    subgraph CLI["CLI (nel)"]
        RUN["nel eval run"]
        SERVE["nel serve"]
        VALIDATE["nel validate"]
        REPORT["nel eval report"]
        REGRESSION["nel regression"]
    end

    subgraph ENVS["Environments"]
        REG["Registry"]
        BYOB["@benchmark + @scorer"]
        GYM["GymEnvironment"]
        SKILLS["SkillsEnvironment"]
        LMEVAL["LMEvalEnvironment"]
        VLM["VLMEvalKitEnvironment"]
    end

    subgraph ENGINE["Engine"]
        LOOP["eval_loop"]
        SOLVER["Solver"]
        MC["ModelClient"]
    end

    subgraph CFG["Config"]
        EVALCFG["EvalConfig"]
        COMPOSE["compose"]
    end

    subgraph ORCH["Orchestration"]
        DEPLOY["ModelServer"]
        LOCALRUN["Orchestrator"]
        SLURMGEN["SlurmGen"]
    end

    subgraph EXEC["Executors"]
        LOCAL["LocalExecutor"]
        DOCKER["DockerExecutor"]
        SLURM["SlurmExecutor"]
    end

    subgraph SAND["Sandbox"]
        SBASE["Sandbox Protocol"]
        SDOCK["DockerSandbox"]
        SSLURM["SlurmSandbox"]
        SLOCAL["LocalSandbox"]
        SECS["ECSFargateSandbox"]
        SMGR["SandboxManager"]
    end

    subgraph PROXY["Proxy"]
        LITELLM["LiteLLM Proxy"]
        INTERCEPT["Interceptors"]
    end

    subgraph OBS["Observability"]
        TYPES["StepRecord<br/>ModelResponse"]
        COLLECT["ArtifactCollector"]
        PROG["ConsoleProgress"]
    end

    subgraph SCORE["Scoring"]
        JUDGE["judge.py"]
        JSCHEMA["json_schema.py"]
    end

    RUN --> EVALCFG
    RUN --> EXEC
    EXEC --> ORCH
    ORCH --> DEPLOY
    ORCH --> LOOP
    LOOP --> SMGR
    SERVE --> REG
    VALIDATE --> LOOP

    LOOP --> REG
    LOOP --> SOLVER
    SOLVER --> MC
    LOOP --> COLLECT
    LOOP --> PROG

    REG --> BYOB
    REG --> GYM
    REG --> SKILLS
    REG --> LMEVAL
    REG --> VLM

    LOOP --> JUDGE
    LOOP --> LITELLM
    LITELLM --> INTERCEPT

    style CLI fill:#e8eaf6
    style ENVS fill:#e8f5e9
    style ENGINE fill:#fff3e0
    style CFG fill:#e8eaf6
    style ORCH fill:#fff8e1
    style EXEC fill:#f3e5f5
    style OBS fill:#fce4ec
    style SAND fill:#e0f7fa
    style SCORE fill:#fff9c4
    style PROXY fill:#e1f5fe
```

## Package Map

| Package | Responsibility | Key types |
|---------|---------------|-----------|
| `environments/` | Base class, registry, `@benchmark` BYOB API, environment types | `EvalEnvironment`, `SeedResult`, `VerifyResult`, `BenchmarkDefinition` |
| `benchmarks/` | 15 built-in benchmarks (all `@benchmark` + `@scorer`) | Scorer functions |
| `solvers/` | Pluggable inference strategies | `Solver`, `ChatSolver`, `VLMSolver`, `HarborSolver`, `GymSolver`, `ReActSolver`, `NatSolver`, `OpenClawSolver` (config `type`: `simple` → ChatSolver/VLMSolver, `harbor`, `tool_calling` → ReActSolver, `gym_delegation`, etc.) |
| `engine/` | Core eval loop, model client, checkpoint, comparison, export plugins | `run_evaluation()`, `ModelClient`, `CheckpointManager`, `InspectExporter`, `WandBExporter` |
| `config/` | Pydantic config schemas, env expansion, YAML composition | `EvalConfig`, `parse_eval_config()`, `compose_config()`, `services.py`, `sandboxes.py`, `solvers.py`, `scoring.py`, `clusters.py` |
| `orchestration/` | Suite orchestration, model server management, SLURM generation, proxy lifecycle | `run_local()`, `DeployConfig`, `generate_sbatch()`, `start_proxy()` |
| `serving/` | HTTP server for environments (evaluator + Gym protocol) | `generate_app()` |
| `executors/` | Executor protocol and backends (local, Docker, SLURM) | `Executor`, `get_executor()`, `detect_executor()`, `ProcessState` |
| `sandbox/` | Per-problem isolated execution and strategy patterns | `Sandbox`, `SandboxSpec`, `SandboxManager`, `ECSFargateSandbox`, `NoSandbox`, `StatefulSandbox`, `StatelessSandbox` |
| `interceptors/` | LiteLLM proxy callback plugins for request/response interception | `resolve_interceptors()`, `BaseInterceptor` |
| `scoring/` | Verification scorers, judge pipeline, JSON schema | `exact_match`, `code_sandbox`, `needs_judge` |
| `observability/` | Rich telemetry capture | `StepRecord`, `ModelResponse`, `RuntimeStats`, `ArtifactCollector` |
| `metrics/` | Statistical aggregation | `pass_at_k()`, `bootstrap_ci()`, `category_breakdown()` |
| `cli/` | CLI commands | `nel eval run`, `nel eval report`, `nel eval merge`, `nel serve`, `nel validate`, `nel list`, `nel regression`, `nel config`, `nel package` |

## Environment Abstraction

Everything is an `EvalEnvironment`. Built-in benchmarks, remote Gym servers, NeMo Skills tasks, lm-eval harness tasks, and VLMEvalKit benchmarks all implement the same contract:

```{mermaid}
classDiagram
    class EvalEnvironment {
        <<abstract>>
        +str name
        +seed(idx) SeedResult
        +verify(response, expected, sandbox?) VerifyResult
        +sandbox_specs() list~SandboxSpec~ | None
        +dataset_size() int
        +close()
    }

    class ByobEnvironment {
        +BenchmarkDefinition definition
    }

    class GymEnvironment {
        +str endpoint
    }

    class SkillsEnvironment {
        +str benchmark
        +str eval_type
    }

    class LMEvalEnvironment {
        +str task_name
    }

    class VLMEvalKitEnvironment {
        +str dataset_name
    }

    class ContainerEnvironment {
        +str image_uri
        +str task_name
    }

    EvalEnvironment <|-- ByobEnvironment
    EvalEnvironment <|-- GymEnvironment
    EvalEnvironment <|-- SkillsEnvironment
    EvalEnvironment <|-- LMEvalEnvironment
    EvalEnvironment <|-- VLMEvalKitEnvironment
    EvalEnvironment <|-- ContainerEnvironment
```

### Resolution

The registry resolves environment names in order:

1. **URI scheme** -- `lm-eval://task`, `skills://name`, `gym://host:port`, `gym://name`, `vlmevalkit://dataset`, `container://image#task`
2. **Built-in registry** -- names registered via `@benchmark` or `@register`

```python
from nemo_evaluator import get_environment

env = get_environment("mmlu")                        # built-in
env = get_environment("lm-eval://aime25")            # lm-eval task
env = get_environment("skills://gpqa")               # NeMo Skills
env = get_environment("gym://localhost:9090")         # remote Gym
env = get_environment("vlmevalkit://MMBench_DEV_EN") # VLMEvalKit
```

## Evaluation Flow

```{mermaid}
sequenceDiagram
    participant CLI as nel eval run
    participant Exec as Executor
    participant Runner as Orchestrator
    participant Deploy as ModelServer
    participant Loop as eval_loop
    participant Solver as Solver
    participant Env as Environment
    participant Obs as ArtifactCollector

    CLI->>Exec: executor.run(config)
    Exec->>Runner: run_local(config)
    Runner->>Deploy: start()
    Deploy-->>Runner: model_url
    opt proxy configured
        Runner->>Runner: start_proxy(model_url)
        Note right of Runner: model_url rewritten to proxy URL
    end

    Runner->>Loop: run_evaluation(env, solver, config, sandbox_manager)

    loop For each problem x repeat
        Loop->>Env: seed(idx)
        Env-->>Loop: SeedResult(prompt, expected, sandbox_spec?)
        opt sandbox configured
            Loop->>Loop: manager.acquire(spec)
        end
        Loop->>Solver: solve(task, sandbox?)
        Solver-->>Loop: SolveResult(response)
        Loop->>Env: verify(response, expected, sandbox?)
        Env-->>Loop: VerifyResult(reward, details)
        opt sandbox acquired
            Loop->>Loop: manager.release(sandbox)
        end
        Loop->>Obs: record(StepRecord)
    end

    Loop-->>Runner: bundle
    Runner->>Deploy: stop()
```

## Solver Protocol

Solvers decouple inference strategy from benchmark logic. The eval loop calls `solver.solve(task)` and receives a response. In YAML configs, solvers are configured via `solver.type` in each benchmark.

| Solver | Config `type` | Protocol | Use case |
|--------|---------------|----------|----------|
| `ChatSolver` | `simple` | `/chat/completions` | Standard benchmarks (default) |
| `VLMSolver` | `simple` (images) | `/chat/completions` + images | Vision-language benchmarks |
| `HarborSolver` | `harbor` | Harbor agent SDK | Agentic evaluation (OpenHands, SWE-agent) |
| `ReActSolver` | `tool_calling` | NEL-driven ReAct loop | Full-observability tool use (Gym HTTP tools, sandbox tools, or both) |
| `GymSolver` | `gym_delegation` | HTTP to nemo-gym | Delegate solve to gym server |
| `NatSolver` | (via service) | SSE `/generate/full` | NAT agent benchmarks |
| `OpenClawSolver` | `openclaw` | OpenClaw CLI | OpenClaw benchmarks |

```python
class Solver(Protocol):
    async def solve(self, task: SeedResult) -> SolveResult: ...
```

## Executor Protocol

All execution backends implement the `Executor` protocol (`executors/__init__.py`):

```python
class Executor(Protocol):
    name: str
    def run(self, config, *, dry_run=False, resume=False,
            background=False, submit=False) -> None: ...
    def status(self, output_dir) -> ProcessState: ...
    def stop(self, output_dir) -> bool: ...
    @staticmethod
    def detect(output_dir) -> bool: ...
```

The CLI dispatches via `get_executor(config.cluster.type)` and `detect_executor(output_dir)` -- no if/elif trees.

| Executor | Config | Metadata file | What it does |
|----------|--------|---------------|-------------|
| `LocalExecutor` | `cluster.type: local` | `nel.pid` | In-process eval with optional model deployment, checkpointing, and failure isolation |
| `DockerExecutor` | `cluster.type: docker` | `docker.json` | Runs eval inside a Docker container with the correct per-harness image |
| `SlurmExecutor` | `cluster.type: slurm` | `slurm_job.json` | Generates self-contained sbatch scripts with per-benchmark containers |

SLURM uses `node_pools` to declare resource topology. Services and sandboxes reference pools by name, enabling heterogeneous jobs (e.g., GPU nodes for model serving + CPU nodes for sandboxes).

Adding a new executor (e.g. Kubernetes) requires only a new class, a metadata file convention, and a registry entry.

```bash
# Local with external API
nel eval run --bench mmlu --model-url https://api.example.com/v1

# Docker
nel eval run config.yaml    # with cluster.type: docker

# SLURM (generates + submits sbatch)
nel eval run config.yaml    # with cluster.type: slurm
```

## Model Deployment

The `orchestration/model_server.py` module manages model server lifecycle:

| Config `type` | Internal class | Description |
|---------------|---------------|-------------|
| `api` | `APIDeployment` | External API, no server management |
| `vllm` | `ProcessModelDeployment` | Local vLLM process |
| `sglang` | `ProcessModelDeployment` | Local SGLang process |
| `trt_llm` | `ProcessModelDeployment` | Local TensorRT-LLM process |

All deployments implement `start() -> url`, `health_wait()`, and `stop()`.

## Observability Data Model

```{mermaid}
classDiagram
    class StepRecord {
        +str step_id
        +int problem_idx
        +int repeat
        +str prompt
        +str expected_answer
        +ModelResponse model_response
        +float reward
        +str extracted_answer
        +str scoring_method
        +dict scoring_details
        +float seed_ms
        +float model_ms
        +float verify_ms
        +str failure_category
    }

    class ModelResponse {
        +str content
        +str model
        +str finish_reason
        +int prompt_tokens
        +int completion_tokens
        +int reasoning_tokens
        +float latency_ms
        +dict raw_response
    }

    class RuntimeStats {
        +int total_steps
        +int total_tokens
        +float elapsed_seconds
        +dict latency_percentiles_ms
        +float tokens_per_second
        +float steps_per_second
        +int model_errors
    }

    class FailureReport {
        +int total_failures
        +float failure_rate
        +dict categories
        +list exemplars
    }

    StepRecord --> ModelResponse
```

## Resilience and Resume

Multi-benchmark suites use `CheckpointManager` to track per-benchmark completion:

```{mermaid}
flowchart LR
    START["nel eval run suite.yaml"] --> B1["Benchmark 1"]
    B1 -->|"completed"| CKPT1["checkpoint: done"]
    CKPT1 --> B2["Benchmark 2"]
    B2 -->|"FAILED"| CKPT2["checkpoint: failed"]
    CKPT2 --> B3["Benchmark 3"]
    B3 -->|"completed"| CKPT3["checkpoint: done"]
    CKPT3 --> SUMMARY["Summary: 2 done, 1 failed"]
    SUMMARY --> RESUME["nel eval run suite.yaml --resume"]
    RESUME --> B2R["Retry Benchmark 2"]
```

- **Failure isolation**: A failing benchmark is caught, logged, and skipped. The suite continues to the next benchmark.
- **Checkpoint tracking**: Each benchmark's status (completed/failed) is persisted to disk under the output directory.
- **Resume**: `--resume` skips completed benchmarks and retries failed ones. Without `--resume`, checkpoints are cleared for a fresh run.

## Sharding

```{mermaid}
flowchart LR
    subgraph "Environment Variables"
        IDX["NEL_SHARD_IDX<br/>or SLURM_ARRAY_TASK_ID"]
        TOT["NEL_TOTAL_SHARDS<br/>or SLURM_ARRAY_TASK_COUNT"]
    end

    IDX --> RANGE["get_shard_range(total, idx, shards)"]
    TOT --> RANGE
    RANGE --> LOOP["run_evaluation(..., problem_range=(start, end))"]
    LOOP --> OUTPUT["shard_N/results.jsonl"]
    OUTPUT --> MERGE["merge_results(shard_dirs)"]
    MERGE --> FINAL["merged/eval-*.json<br/>Recomputed pass@k, CI"]
```

Sharding is transparent to the eval loop. `problem_range` changes the iteration bounds. The merge step recomputes global metrics from concatenated per-shard results.
