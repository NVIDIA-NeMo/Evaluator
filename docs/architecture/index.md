# Architecture

## System Overview

```{mermaid}
flowchart TB
    subgraph CLI["CLI (nel)"]
        RUN["nel run"]
        SERVE["nel serve"]
        VALIDATE["nel validate"]
        REGRESSION["nel regression"]
        SLURM["nel slurm"]
    end

    subgraph CORE["Core"]
        ENV["environments/"]
        SCORE["scoring/"]
        METRICS["metrics/"]
    end

    subgraph RUNNER["Runner"]
        LOOP["eval_loop"]
        MC["ModelClient"]
        ART["artifacts"]
        SHARD["sharding"]
        REG["regression"]
        RAYL["ray_launcher"]
    end

    subgraph OBS["Observability"]
        TYPES["StepRecord<br/>ModelResponse"]
        COLLECT["ArtifactCollector"]
        PROG["ConsoleProgress"]
    end

    subgraph ADAPT["Adapters"]
        GYM["GymAdapter"]
        PI["PIAdapter"]
        SKILLS["SkillsAdapter"]
        CONTAINER["ContainerAdapter"]
        HARNESS["GymHarness"]
    end

    RUN --> LOOP
    SERVE --> ENV
    VALIDATE --> LOOP

    LOOP --> ENV
    LOOP --> ADAPT
    LOOP --> MC
    LOOP --> COLLECT
    LOOP --> PROG
    LOOP --> METRICS

    ENV --> SCORE
    COLLECT --> ART
    SLURM --> SHARD

    style CLI fill:#e8eaf6
    style CORE fill:#e8f5e9
    style RUNNER fill:#fff3e0
    style OBS fill:#fce4ec
    style ADAPT fill:#e0f7fa
```

## Package Map

| Package | Responsibility | Key types |
|---------|---------------|-----------|
| `environments` | Base class, registry, HTTP server, Gym protocol | `EvalEnvironment`, `SeedResult`, `VerifyResult` |
| `benchmarks` | Built-in benchmarks (BYOB examples) | `GSM8KEnvironment`, `TriviaQAEnvironment` |
| `adapters` | Consume external environments | `EnvironmentAdapter`, `GymAdapter`, `PIAdapter`, `SkillsAdapter`, `ContainerAdapter`, `GymHarness` |
| `runner` | Evaluation orchestration | `run_evaluation()`, `ModelClient`, `write_all()` |
| `observability` | Rich telemetry capture | `StepRecord`, `ModelResponse`, `RuntimeStats`, `ArtifactCollector` |
| `scoring` | Scoring primitives | `math_equal()`, `exact_match()`, `extract_answer()` |
| `metrics` | Statistical aggregation | `pass_at_k()`, `bootstrap_ci()`, `category_breakdown()` |
| `cli` | CLI commands | `nel run`, `nel serve`, `nel validate`, `nel regression`, `nel slurm` |

## Evaluation Flow

```{mermaid}
sequenceDiagram
    participant CLI as nel run
    participant Loop as eval_loop
    participant Env as Environment
    participant Model as ModelClient
    participant Obs as ArtifactCollector
    participant Art as Artifacts

    CLI->>Loop: run_evaluation(env, client, n_repeats)

    loop For each problem (possibly sharded)
        loop For each repeat
            Loop->>Env: seed(idx)
            Env-->>Loop: SeedResult(prompt, expected)
            Loop->>Model: chat(prompt)
            Model-->>Loop: ModelResponse(content, tokens, latency)
            Loop->>Env: verify(response, expected)
            Env-->>Loop: VerifyResult(reward, extracted, scoring_details)
            Loop->>Obs: record(StepRecord)
        end
    end

    Loop->>Obs: build(elapsed)
    Obs-->>Loop: RunArtifacts(runtime, failures)
    Loop->>Art: build_artifact_bundle(results, metrics)
    Art-->>CLI: bundle dict
    CLI->>Art: write_all(bundle, output_dir)
```

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
        +request_hash()
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

    class RunArtifacts {
        +list~StepRecord~ steps
        +RuntimeStats runtime
        +FailureReport failures
    }

    StepRecord --> ModelResponse
    RunArtifacts --> StepRecord
    RunArtifacts --> RuntimeStats
    RunArtifacts --> FailureReport
```

## Environment Abstraction

Both local benchmarks and remote adapters implement the same contract:

```{mermaid}
classDiagram
    class EvalSource {
        <<union>>
        EvalEnvironment | EnvironmentAdapter
    }

    class EvalEnvironment {
        +str name
        +seed(idx) SeedResult
        +verify(response, expected) VerifyResult
        +__len__() int
    }

    class EnvironmentAdapter {
        +str name
        +seed(idx) SeedResult
        +verify(response, expected) VerifyResult
        +dataset_size() int
    }

    class GymAdapter {
        +str endpoint
    }

    class PIAdapter {
        +str env_name
        +str system_prompt
    }

    class SkillsAdapter {
        +str benchmark
        +str eval_type
    }

    EvalSource <|-- EvalEnvironment
    EvalSource <|-- EnvironmentAdapter
    EnvironmentAdapter <|-- GymAdapter
    EnvironmentAdapter <|-- PIAdapter
    EnvironmentAdapter <|-- SkillsAdapter
```

The eval loop uses `EvalSource = Union[EvalEnvironment, EnvironmentAdapter]` and dispatches to sync or async methods via helpers:

```python
async def _seed(src: EvalSource, idx: int):
    if isinstance(src, EnvironmentAdapter):
        return await src.seed(idx)
    return src.seed(idx)
```

## Sharding Model

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

Key design: the sharding is transparent to the eval loop. `problem_range` simply changes the iteration bounds `for idx in range(start, end)`. The merge step recomputes global metrics from the concatenated per-shard results.
