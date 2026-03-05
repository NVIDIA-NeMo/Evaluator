# External Environment Adapters

Consume environments from external platforms (Gym servers, Prime Intellect) through Evaluator's adapter layer.

## Adapter Architecture

```{mermaid}
flowchart TB
    subgraph "Evaluator"
        LOOP["Eval Loop<br/>run_evaluation()"]
        CLIENT["ModelClient"]
        COLLECT["ArtifactCollector"]
    end

    subgraph "Adapters"
        GYM["GymAdapter<br/>gym://host:port"]
        PI["PIAdapter<br/>pi://env_name"]
    end

    subgraph "External"
        GYMSERV["Gym Resource Server"]
        PIENV["verifiers.SingleTurnEnv"]
    end

    LOOP --> GYM
    LOOP --> PI
    LOOP --> CLIENT
    LOOP --> COLLECT
    GYM -->|HTTP| GYMSERV
    PI -->|Python| PIENV
```

All adapters implement the same interface:

```python
class EnvironmentAdapter(ABC):
    name: str

    async def seed(self, idx: int) -> SeedResult: ...
    async def verify(self, response: str, expected: str, **meta) -> VerifyResult: ...
    async def dataset_size(self) -> int: ...
```

Because the eval loop owns the model call between `seed()` and `verify()`, every adapter gets full observability: per-request latency, token counts, trajectories, failure analysis.

## GymAdapter: Remote HTTP Environments

Consumes any server that implements the `nel serve` protocol.

### CLI

```bash
nel run --adapter gym://localhost:9090 --repeats 4
nel run --adapter gym://gym-cluster:8080 --repeats 2 --max-problems 100
```

### Python

```python
from nemo_evaluator.adapters import GymAdapter
from nemo_evaluator.runner import run_evaluation, ModelClient

adapter = GymAdapter("http://localhost:9090")
client = ModelClient(base_url="https://inference-api.nvidia.com/v1", model="azure/openai/gpt-5.2")
bundle = await run_evaluation(adapter, client, n_repeats=4)
```

## PIAdapter: Prime Intellect Environments

Decomposes `verifiers.SingleTurnEnv` into Evaluator's seed/verify loop.

### Setup

```bash
pip install -e ".[pi]"
```

### CLI

```bash
nel run --adapter pi://simpleqa --repeats 2
nel run --adapter pi://math500 --repeats 4
```

### Python

```python
from nemo_evaluator.adapters.pi import PIAdapter
from nemo_evaluator.runner import run_evaluation, ModelClient

adapter = PIAdapter("simpleqa")
client = ModelClient(base_url="https://inference-api.nvidia.com/v1", model="azure/openai/gpt-5.2")
bundle = await run_evaluation(adapter, client, n_repeats=2, system_prompt=adapter.system_prompt)
```

### How it works

```{mermaid}
sequenceDiagram
    participant E as Evaluator
    participant PI as PIAdapter
    participant V as verifiers.SingleTurnEnv
    participant M as Model

    E->>PI: seed(idx)
    PI->>V: eval_dataset[idx]
    PI-->>E: SeedResult(prompt, expected)

    E->>M: chat(prompt)
    M-->>E: ModelResponse

    E->>PI: verify(response, expected)
    PI->>V: rubric.score_rollout(state)
    PI-->>E: VerifyResult(reward, details)
```

The adapter extracts the prompt and expected answer from PI's `eval_dataset`, lets Evaluator make the model call, then uses PI's `rubric.score_rollout()` for scoring. This gives you PI's native scoring with Evaluator's full artifact suite.

## Writing a Custom Adapter

Implement the `EnvironmentAdapter` ABC:

```python
from nemo_evaluator.adapters.base import EnvironmentAdapter
from nemo_evaluator.environments.base import SeedResult, VerifyResult


class MyAdapter(EnvironmentAdapter):
    name = "my_platform"

    def __init__(self, config: str):
        # Load your environment
        pass

    async def seed(self, idx: int) -> SeedResult:
        # Fetch problem idx from your platform
        return SeedResult(prompt="...", expected_answer="...", metadata={})

    async def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        # Score the response using your platform's logic
        return VerifyResult(reward=1.0, extracted_answer="...", scoring_details={})

    async def dataset_size(self) -> int:
        return 1000
```

Then use it:

```python
adapter = MyAdapter("my_config")
bundle = await run_evaluation(adapter, client, n_repeats=2)
```
