# Legacy Evaluator Containers

Run any existing NeMo Evaluator harness container (simple-evals, lm-evaluation-harness, nemo-skills, mtbench, etc.) through NEL's unified interface.

## How It Works

```{mermaid}
flowchart TB
    NEL["run_container_eval()"] --> CONFIG["Build config_ef.yaml"]
    CONFIG --> DOCKER["docker run<br/>eval-factory container"]
    DOCKER --> HARNESS["Harness runs<br/>(simple-evals, lm-eval, etc.)"]
    HARNESS --> OUTPUT["results.yml<br/>eval_factory_metrics.json"]
    OUTPUT --> PARSE["Parse & convert<br/>to NEL bundle"]

    subgraph "Container"
        HARNESS
        ADAPTER["AdapterServer<br/>(model proxy)"]
        HARNESS -->|model calls| ADAPTER
        ADAPTER -->|forward| MODEL["Model API"]
    end

    style NEL fill:#e1f5fe
    style HARNESS fill:#fff3e0
```

The container adapter does **not** decompose the harness. The harness owns the model call, which means you get aggregate scores and response stats from the adapter interceptors, but not per-request trajectories. For full observability, use the {doc}`skills-integration` (native mode) instead.

## Available Harnesses

```bash
nel list --source lm-eval
```

| Harness | Container | Example tasks |
|---------|-----------|--------------|
| `simple_evals` | `nvcr.io/.../simple-evals:26.01` | AIME_2025, GPQA_diamond, MMLU |
| `lm-evaluation-harness` | `nvcr.io/.../lm-evaluation-harness:26.01` | ifeval, arc, hellaswag |
| `nemo_skills` | `nvcr.io/.../nemo-skills:26.01` | gsm8k, math, aime24 |
| `mtbench` | `nvcr.io/.../mtbench:26.01` | mt_bench |
| `bfcl` | `nvcr.io/.../bfcl:26.01` | bfcl_v3, bfcl_v4 |
| `hle` | `nvcr.io/.../hle:26.01` | hle |
| `livecodebench` | `nvcr.io/.../livecodebench:26.01` | livecodebench |
| `scicode` | `nvcr.io/.../scicode:26.01` | scicode |
| `vlmevalkit` | `nvcr.io/.../vlmevalkit:26.01` | VLM benchmarks |
| `safety_eval` | `nvcr.io/.../safety-harness:26.01` | Safety evals |
| `helm` | `nvcr.io/.../helm:26.01` | HELM benchmarks |

Plus 10+ more (see `nel list --source lm-eval`).

## Python API

```python
from nemo_evaluator.environments.container import ContainerEnvironment
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.solvers import ChatSolver
from nemo_evaluator.engine.model_client import ModelClient

env = ContainerEnvironment(
    image="nvcr.io/.../simple-evals:26.01",
    task="GPQA_diamond",
)

client = ModelClient(
    base_url="https://inference-api.nvidia.com/v1",
    model="azure/openai/gpt-5.2",
)
solver = ChatSolver(client)
bundle = await run_evaluation(env, solver)
```

## Output Format

The container adapter parses `results.yml` and `eval_factory_metrics.json` from the container output:

```json
{
  "source": "container",
  "image": "nvcr.io/.../simple-evals:26.01",
  "task": "simple_evals.AIME_2025",
  "scores": {
    "AIME_2025/score/micro": {"value": 0.4, "stats": {"stddev": 0.49, "stderr": 0.16}}
  },
  "runtime": {
    "elapsed_seconds": 120.5,
    "inference_time_seconds": 98.2,
    "scoring_time_seconds": 22.3
  },
  "response_stats": {
    "avg_latency_ms": 1250.0,
    "avg_prompt_tokens": 320,
    "avg_completion_tokens": 450,
    "count": 50,
    "successful_count": 48
  }
}
```

## Observability Trade-offs

| Feature | Container mode | Native mode (Skills/BYOB) |
|---------|---------------|--------------------------|
| Aggregate scores | Yes | Yes |
| Response stats (avg latency, tokens) | Yes (from interceptors) | Yes |
| Per-request trajectories | No | Yes |
| Failure categorization | No | Yes |
| n_repeats with pass@k | No (harness controls) | Yes |
| Bootstrap CI | No | Yes |
| Scoring details per sample | No | Yes |

For full observability, see {doc}`skills-integration` which wraps Skills benchmarks natively.
