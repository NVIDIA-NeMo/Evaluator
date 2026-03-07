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
        PI["PIEnvironment"]
    end

    subgraph RUNNER["Runner"]
        LOOP["eval_loop"]
        SOLVER["Solver"]
        MC["ModelClient"]
        DEPLOY["Deployment"]
    end

    subgraph EXEC["Executors"]
        LOCAL["LocalExecutor"]
        DOCKER["DockerExecutor"]
        SLURM["SlurmExecutor"]
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

    RUN --> EXEC
    EXEC --> DEPLOY
    EXEC --> LOOP
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
    REG --> PI

    LOOP --> JUDGE

    style CLI fill:#e8eaf6
    style ENVS fill:#e8f5e9
    style RUNNER fill:#fff3e0
    style EXEC fill:#f3e5f5
    style OBS fill:#fce4ec
    style SCORE fill:#fff9c4
```

## Package Map

| Package | Responsibility | Key types |
|---------|---------------|-----------|
| `environments/` | Base class, registry, `@benchmark` API, environment types | `EvalEnvironment`, `SeedResult`, `VerifyResult`, `BenchmarkDefinition` |
| `benchmarks/` | 11 built-in benchmarks (all `@benchmark` + `@scorer`) | Scorer functions |
| `runner/` | Eval loop, solvers, model client, deployment | `run_evaluation()`, `ChatSolver`, `CompletionSolver`, `AgentSolver`, `ModelClient` |
| `executors/` | Orchestration: model deploy + eval + teardown | `LocalExecutor`, `DockerExecutor`, `SlurmExecutor` |
| `scoring/` | Judge pipeline and JSON schema validation | `judge.py`, `json_schema.py` |
| `observability/` | Rich telemetry capture | `StepRecord`, `ModelResponse`, `RuntimeStats`, `ArtifactCollector` |
| `metrics/` | Statistical aggregation | `pass_at_k()`, `bootstrap_ci()`, `category_breakdown()` |
| `cli/` | CLI commands | `nel eval run`, `nel serve`, `nel validate`, `nel eval report`, `nel regression` |

## Environment Abstraction

Everything is an `EvalEnvironment`. Built-in benchmarks, remote Gym servers, NeMo Skills tasks, lm-eval harness tasks, and Prime Intellect environments all implement the same contract:

```{mermaid}
classDiagram
    class EvalEnvironment {
        <<abstract>>
        +str name
        +seed(idx) SeedResult
        +verify(response, expected) VerifyResult
        +dataset_size() int
        +close()
    }

    class ByobEnvironment {
        +BenchmarkDefinition definition
    }

    class GymEnvironment {
        +str endpoint
    }

    class ManagedGymEnvironment {
        +start()
        +stop()
    }

    class SkillsEnvironment {
        +str benchmark
        +str eval_type
    }

    class LMEvalEnvironment {
        +str task_name
    }

    class PIEnvironment {
        +str env_name
    }

    EvalEnvironment <|-- ByobEnvironment
    EvalEnvironment <|-- GymEnvironment
    EvalEnvironment <|-- ManagedGymEnvironment
    EvalEnvironment <|-- SkillsEnvironment
    EvalEnvironment <|-- LMEvalEnvironment
    EvalEnvironment <|-- PIEnvironment
```

### Resolution

The registry resolves environment names in order:

1. **URI scheme** -- `gym://host:port`, `skills://name`, `pi://name`, `gym-managed://...`
2. **Namespace prefix** -- `lm-eval/task_name`
3. **Built-in registry** -- names registered via `@benchmark` or `@register`

```python
from nemo_evaluator import get_environment

env = get_environment("mmlu")                        # built-in
env = get_environment("lm-eval/aime25")              # lm-eval task
env = get_environment("skills://gpqa")               # NeMo Skills
env = get_environment("gym://localhost:9090")         # remote Gym
env = get_environment("pi://simpleqa")               # Prime Intellect
```

## Evaluation Flow

```{mermaid}
sequenceDiagram
    participant CLI as nel eval run
    participant Exec as Executor
    participant Deploy as Deployment
    participant Loop as eval_loop
    participant Solver as Solver
    participant Env as Environment
    participant Obs as ArtifactCollector

    CLI->>Exec: execute(RunConfig)
    Exec->>Deploy: start()
    Deploy-->>Exec: model_url

    Exec->>Loop: run_evaluation(env, solver, config)

    loop For each problem x repeat
        Loop->>Env: seed(idx)
        Env-->>Loop: SeedResult(prompt, expected)
        Loop->>Solver: solve(task)
        Solver-->>Loop: SolveResult(response)
        Loop->>Env: verify(response, expected)
        Env-->>Loop: VerifyResult(reward, details)
        Loop->>Obs: record(StepRecord)
    end

    Loop-->>Exec: RunArtifacts
    Exec->>Deploy: stop()
```

## Solver Protocol

Solvers decouple inference strategy from benchmark logic. The eval loop calls `solver.solve(task)` and receives a response.

| Solver | Protocol | Use case |
|--------|----------|----------|
| `ChatSolver` | Single `/chat/completions` call | Standard benchmarks (default) |
| `CompletionSolver` | `/completions` endpoint | Legacy models, prompt-based eval |
| `AgentSolver` | External agent subprocess | Multi-turn agents (OpenHands, SWE-agent) |

```python
class Solver(Protocol):
    async def solve(self, task: SeedResult) -> SolveResult: ...
```

## Executor Model

Executors manage the full lifecycle: deploy model, run evaluation, collect results, tear down.

| Executor | What it does |
|----------|-------------|
| `LocalExecutor` | In-process eval with optional Docker/vLLM model deployment |
| `DockerExecutor` | Model server + eval in Docker containers |
| `SlurmExecutor` | SLURM sbatch with model server + eval jobs |

```bash
# Local with external API
nel eval run --bench mmlu --model-url https://api.example.com/v1

# Local with NIM deployment
nel eval run --bench mmlu --executor local --deploy nim --deploy-image nvcr.io/nim/llama3-8b

# SLURM cluster
nel eval run --bench mmlu --executor slurm --deploy nim --deploy-image nvcr.io/nim/llama3-8b
```

## Model Deployment

The `runner/deployment.py` module manages model server lifecycle:

| Type | Class | Description |
|------|-------|-------------|
| `api` | `APIDeployment` | External API, no server management |
| `nim` | `NIMDeployment` | NIM container with standard NIM env vars |
| `docker` | `DockerModelDeployment` | Generic Docker container |
| `vllm` | `ProcessModelDeployment` | Local vLLM process |
| `sglang` | `ProcessModelDeployment` | Local SGLang process |

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
