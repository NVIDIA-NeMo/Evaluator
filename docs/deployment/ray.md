# Ray Deployment

Distribute evaluations across a Ray cluster for teams using NeMo Gym's Ray infrastructure.

## Setup

```bash
pip install -e ".[ray]"
```

## CLI

```bash
ray job submit --working-dir . -- python -m nemo_evaluator.engine.ray_launcher \
    --bench gsm8k --shards 8 --repeats 5 \
    --model-url https://integrate.api.nvidia.com/v1 \
    --model-id azure/openai/gpt-5.2 \
    --output-dir ./eval_results/ray
```

## Architecture

```{mermaid}
flowchart TB
    HEAD["Ray Head Node<br/>ray_launcher.py"] --> W0["Worker 0<br/>problems [0, 165)"]
    HEAD --> W1["Worker 1<br/>problems [165, 330)"]
    HEAD --> W7["Worker 7<br/>problems [1155, 1319)"]

    W0 --> FS["Shared Storage"]
    W1 --> FS
    W7 --> FS

    FS --> MERGE["In-process merge"]
    MERGE --> RESULT["Final bundle<br/>eval-*.json"]
```

Each shard runs as a `@ray.remote` task. The head node collects all results and merges them locally.

## Python API

```python
import ray
from nemo_evaluator.engine.ray_launcher import run_shard

ray.init()

shards = 8
futures = [
    run_shard.remote(
        benchmark="gsm8k",
        shard_idx=i,
        total_shards=shards,
        model_url="https://integrate.api.nvidia.com/v1",
        model_id="azure/openai/gpt-5.2",
        n_repeats=5,
    )
    for i in range(shards)
]

results = ray.get(futures)
# results is a list of bundle dicts, one per shard
```

## Resource requirements

Evaluation is CPU+network bound (no GPU needed for the evaluator itself):

```python
@ray.remote(num_cpus=2, memory=2 * 1024 * 1024 * 1024)
def run_shard(...):
    ...
```

Adjust based on dataset size and concurrency. The `ModelClient` default is 8 concurrent requests.
