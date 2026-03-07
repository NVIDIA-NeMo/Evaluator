# External Environments

```{deprecated} 0.4.0
The adapter layer (`EnvironmentAdapter`, `GymAdapter`, `PIAdapter`, `SkillsAdapter`) has been removed. All external integrations are now `EvalEnvironment` subclasses resolved through the unified registry.
```

## Migration Guide

| Old (0.3.x) | New (0.4.0) |
|-------------|-------------|
| `nel run --adapter gym://host:port` | `nel run --env gym://host:port` |
| `nel run --adapter skills://gpqa` | `nel run --env skills://gpqa` |
| `nel run --adapter pi://simpleqa` | `nel run --env pi://simpleqa` |
| `GymAdapter("http://...")` | `GymEnvironment("http://...")` |
| `SkillsAdapter("gpqa")` | `SkillsEnvironment("gpqa")` |
| `PIAdapter("simpleqa")` | `PIEnvironment("simpleqa")` |

The `--adapter` CLI flag is no longer recognized. Use `--env` with the same URI.

## Environment Types

All external integrations are standard `EvalEnvironment` subclasses:

| Environment | URI | Source |
|-------------|-----|--------|
| `GymEnvironment` | `gym://host:port` | Remote HTTP server (Gym protocol) |
| `ManagedGymEnvironment` | `gym-managed://...` | Auto-started Gym server |
| `SkillsEnvironment` | `skills://name` | NeMo Skills benchmarks |
| `LMEvalEnvironment` | `lm-eval/task` | lm-evaluation-harness tasks |
| `PIEnvironment` | `pi://name` | Prime Intellect verifiers |

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
