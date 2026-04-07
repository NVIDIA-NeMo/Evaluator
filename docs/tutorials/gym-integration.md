# Gym Integration

Serve NEL benchmarks for NeMo Gym training and consume remote Gym environments for evaluation.

## Architecture

```{mermaid}
flowchart TB
    subgraph "NeMo Evaluator"
        ENV["EvalEnvironment<br/>(any registered benchmark)"]
        SERVE["nel serve"]
        GYMENV["GymEnvironment"]
    end

    subgraph "NeMo Gym"
        TRAIN["Training Loop"]
        COLLECT["ng_collect_rollouts"]
    end

    ENV --> SERVE
    SERVE -->|"seed_session / verify"| TRAIN
    SERVE -->|"JSONL export"| COLLECT
    GYMENV -->|"HTTP client"| SERVE

    style ENV fill:#e1f5fe
    style TRAIN fill:#fff3e0
```

There are three integration modes:

| Mode | Direction | Use case |
|------|-----------|----------|
| **Serve** | Evaluator -> Gym | Gym training consumes NEL benchmarks live |
| **Export** | Evaluator -> Gym | Batch JSONL for `ng_collect_rollouts` |
| **Consume** | Gym -> Evaluator | Evaluate a model against a remote environment |

## Mode 1: Serve for Gym Training

### Start the environment server

```bash
nel serve -b gsm8k -p 9090
```

The server speaks Gym's native protocol:

- `POST /seed_session` -- returns prompt and expected answer
- `POST /verify` -- accepts response, returns `{reward: float}`
- `GET /health` -- health check
- `GET /dataset_size` -- number of problems

### Point Gym at it

In your Gym training config:

```yaml
resource_servers:
  nemo_evaluator:
    endpoint: http://evaluator-host:9090
    eval_type: gsm8k
```

### Get decision-grade scores after training

The same server also speaks NEL's enriched protocol:

```bash
nel eval run --bench gym://evaluator-host:9090 --repeats 4
```

This produces the full artifact suite (trajectories, CI, failure analysis) from the same environment.

## Mode 2: Export for ng_collect_rollouts

For batch rollout collection without a live server:

```bash
nel serve -b gsm8k --export-data /tmp/evaluator_data
```

Or via Python:

```python
from nemo_evaluator import get_environment
import nemo_evaluator.benchmarks  # noqa: F401

env = get_environment("gsm8k")

# Export seed data for each problem
import json
with open("/tmp/rollout_data.jsonl", "w") as f:
    for idx in range(len(env)):
        seed = await env.seed(idx)
        f.write(json.dumps({
            "responses_create_params": {"input": seed.messages or [{"role": "user", "content": seed.prompt}]},
            "expected_answer": seed.expected_answer,
            "uuid": f"gsm8k-{idx}",
            "metadata": seed.metadata,
        }) + "\n")
```

## Mode 3: Consume a Remote Environment

Evaluate a model against any running `nel serve` endpoint using `GymEnvironment`:

```bash
nel eval run --bench gym://localhost:9090 --repeats 2
nel eval run --bench gym://gym-cluster:8080 --repeats 4 --output-dir ./results/remote
```

```{mermaid}
sequenceDiagram
    participant E as nel eval run
    participant G as GymEnvironment
    participant S as Environment Server
    participant M as Model API

    loop For each problem
        E->>G: seed(idx)
        G->>S: POST /seed_session {idx}
        S-->>G: {prompt, expected_answer}
        G-->>E: SeedResult

        E->>M: solver.solve(task)
        M-->>E: SolveResult

        E->>G: verify(response, expected)
        G->>S: POST /verify {response, expected}
        S-->>G: {reward, scoring_details}
        G-->>E: VerifyResult
    end
```

Because NEL makes the model call (not the environment server), you get full observability: per-request latency, token counts, reasoning tokens, failure categorization.

## Managed Gym Environments

For environments that need a server started and stopped automatically, use `gym://` with a benchmark name (not `host:port`). The registry auto-detects that it's a name and starts a managed server:

```bash
nel eval run --bench gym://gsm8k --repeats 4
```

Or with a custom server command:

```bash
nel eval run --bench "gym://cmd:python my_server.py" --repeats 4
```

The server is started automatically, health-checked, used for evaluation, and torn down on completion.

## Python API

```python
from nemo_evaluator.environments.gym import GymEnvironment
from nemo_evaluator import run_evaluation, ChatSolver, ModelClient

env = GymEnvironment("http://localhost:9090")
client = ModelClient(base_url="https://api.example.com/v1", model="my-model")
solver = ChatSolver(client)

bundle = await run_evaluation(env, solver, n_repeats=4)
```
