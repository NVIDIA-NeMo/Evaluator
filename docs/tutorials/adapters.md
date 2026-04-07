# External Environments

All external integrations are `EvalEnvironment` subclasses resolved through the unified registry.

## Environment Types

| Environment | URI | Source |
|-------------|-----|--------|
| `GymEnvironment` | `gym://host:port` | Remote HTTP server (Gym protocol) |
| `ManagedGymEnvironment` | `gym://name` | Auto-started Gym server (name auto-detected) |
| `SkillsEnvironment` | `skills://name` | NeMo Skills benchmarks |
| `LMEvalEnvironment` | `lm-eval://task` | lm-evaluation-harness tasks |
| `VLMEvalKitEnvironment` | `vlmevalkit://dataset` | VLMEvalKit VLM benchmarks |
| `ContainerEnvironment` | `container://image#task` | Legacy eval-factory containers |

```bash
nel eval run --bench gym://host:port
nel eval run --bench skills://gpqa
nel eval run --bench lm-eval://aime2025
nel eval run --bench vlmevalkit://MMBench_DEV_EN
```

See the dedicated tutorials for each:

- {doc}`gym-integration` -- Gym environments
- {doc}`skills-integration` -- NeMo Skills benchmarks

## Writing a Custom Environment

Implement `EvalEnvironment` directly:

```python
from nemo_evaluator import EvalEnvironment, SeedResult, VerifyResult, register

@register("my_platform")
class MyPlatformEnvironment(EvalEnvironment):
    def __init__(self, config: str = "default"):
        super().__init__()
        self._dataset = self._load(config)

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        return SeedResult(prompt=row["prompt"], expected_answer=row["answer"])

    async def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        correct = response.strip().lower() == expected.strip().lower()
        return VerifyResult(reward=1.0 if correct else 0.0)

    def _load(self, config):
        ...
```

Or use the `@benchmark` + `@scorer` API for simpler cases (see {doc}`byob`).
