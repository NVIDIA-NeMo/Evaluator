# Gym Integration

Serve Evaluator benchmarks for NeMo Gym training and consume Gym environments for evaluation.

## Architecture

```{mermaid}
flowchart TB
    subgraph "Evaluator (this package)"
        ENV["EvalEnvironment<br/>(gsm8k, AIME, custom)"]
        SERVE["nel serve --gym-compat"]
        HARNESS["GymHarness"]
        ADAPTER["GymAdapter"]
    end

    subgraph "NeMo Gym"
        TRAIN["Training Loop"]
        COLLECT["ng_collect_rollouts"]
    end

    ENV --> SERVE
    ENV --> HARNESS
    SERVE -->|"seed_session / verify"| TRAIN
    HARNESS -->|"JSONL export"| COLLECT
    ADAPTER -->|"HTTP client"| SERVE

    style ENV fill:#e1f5fe
    style TRAIN fill:#fff3e0
```

There are three integration modes:

| Mode | Direction | Use case |
|------|-----------|----------|
| **Serve** | Evaluator → Gym | Gym training consumes our benchmarks live |
| **Export** | Evaluator → Gym | Batch JSONL for `ng_collect_rollouts` |
| **Consume** | Gym → Evaluator | Evaluate a model against a remote environment |

## Mode 1: Serve for Gym Training

### Step 1: Start the environment server

```bash
nel serve --benchmark gsm8k --gym-compat --port 9090
```

This speaks Gym's native protocol:
- `POST /seed_session` -- returns `{}`  (Gym doesn't use prompt from seed)
- `POST /verify` -- accepts `NeMoGymResponse`, returns `{reward: float}`
- `GET /health` -- health check
- `GET /dataset_size` -- number of problems

### Step 2: Point Gym at it

In your Gym training config:

```yaml
resource_servers:
  nemo_evaluator:
    endpoint: http://evaluator-host:9090
    eval_type: gsm8k
```

### Step 3: Get decision-grade scores after training

The same server also speaks Evaluator's enriched protocol. After training:

```bash
nel run --adapter gym://evaluator-host:9090 --repeats 4
```

This gives you full artifact suite (trajectories, CI, failure analysis) from the same environment.

## Mode 2: Export for ng_collect_rollouts

For batch rollout collection without a live server:

```python
from nemo_evaluator.adapters.gym_harness import GymHarness
from nemo_evaluator.environments import get_environment
import nemo_evaluator.benchmarks  # noqa: F401

env = get_environment("gsm8k")
harness = GymHarness(env)
path = harness.export_jsonl("/tmp/evaluator_data")
print(f"Exported {len(harness.get_dataset())} rows -> {path}")
```

Or via CLI:

```bash
nel serve --benchmark gsm8k --export-data /tmp/evaluator_data
```

The JSONL format matches `ng_collect_rollouts` input:

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "Solve: ..."}]},
  "expected_answer": "42",
  "uuid": "gsm8k-0",
  "metadata": {"category": "algebra"}
}
```

## Mode 3: Consume a Remote Environment

Evaluate a model against any running `nel serve` endpoint:

```bash
# Against your own server
nel run --adapter gym://localhost:9090 --repeats 2

# Against an external Gym resource server
nel run --adapter gym://gym-cluster:8080 --repeats 4 --output-dir ./results/remote
```

```{mermaid}
sequenceDiagram
    participant E as nel run
    participant S as Environment Server
    participant M as Model API

    loop For each problem
        E->>S: POST /seed_session {idx}
        S-->>E: {prompt, expected_answer}
        E->>M: POST /chat/completions
        M-->>E: {response}
        E->>S: POST /verify {response, expected}
        S-->>E: {reward, scoring_details}
    end

    Note over E: Evaluator owns the model call<br/>→ full trajectory capture
```

Because Evaluator makes the model call (not the environment server), you get full observability: per-request latency, token counts, reasoning tokens, failure categorization.

## Serve on SLURM

For long-running training jobs:

```bash
nel slurm serve --benchmark gsm8k --gym-compat \
    --partition batch --time-limit 24:00:00 --submit
```

The allocated hostname:port is written to `eval_results/endpoint.txt`.

## Serve on Kubernetes

```bash
kubectl apply -f deploy/k8s/serve-deployment.yaml
# Gym connects via: gym://nel-serve.default.svc:9090
```

Includes readiness and liveness probes on `/health`.
