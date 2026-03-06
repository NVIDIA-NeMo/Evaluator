# Contributing to NeMo Evaluator

## Setup

```bash
git clone <repo-url> && cd nemo-evaluator-next
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

```bash
make test          # run tests
make lint          # check style
make format        # auto-fix style
make docs          # build docs locally
make clean         # remove build artifacts
```

## Adding a single-turn benchmark

1. Create `src/nemo_evaluator/benchmarks/your_bench.py`
2. Subclass `EvalEnvironment`, implement `seed()` and `verify()`
3. Decorate the class with `@register("your_bench")`

```python
from nemo_evaluator.environments import EvalEnvironment, SeedResult, VerifyResult, register

@register("my_benchmark")
class MyBenchmark(EvalEnvironment):
    def __init__(self):
        super().__init__()
        self._dataset = [...]  # load your data

    def __len__(self):
        return len(self._dataset)

    def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        return SeedResult(prompt=row["question"], expected_answer=row["answer"])

    def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        correct = response.strip() == expected
        return VerifyResult(reward=1.0 if correct else 0.0, extracted_answer=response.strip())
```

4. Import it in `src/nemo_evaluator/benchmarks/__init__.py`
5. Verify: `nel list-environments` should show your benchmark
6. Test: `nel validate -b my_benchmark --model-url ... --model-id ...`

## Adding a multi-turn benchmark

For interactive / agent benchmarks, subclass `StepEnvironment` instead:

```python
from nemo_evaluator.environments import StepEnvironment, Observation, register

@register("my_agent_bench")
class MyAgentBench(StepEnvironment):
    def __init__(self):
        self._tasks = [...]

    def __len__(self):
        return len(self._tasks)

    def reset(self, idx: int) -> Observation:
        return Observation(content=self._tasks[idx]["instruction"],
                           tools=[{"name": "search", "schema": {...}}])

    def step(self, action: str) -> Observation:
        done = check_solution(action)
        return Observation(content="result...", done=done, reward=1.0 if done else 0.0)

    @property
    def max_steps(self) -> int:
        return 20
```

## Adding a scoring primitive

1. Create `src/nemo_evaluator/scoring/your_scorer.py`
2. Export from `src/nemo_evaluator/scoring/__init__.py`
3. Add tests in `tests/test_your_scorer.py`

Available primitives: `math_equal`, `exact_match`, `extract_answer`, `mcq_score`, `judge_score`, `validate_json_schema`.

## Running tests

```bash
pytest                    # all tests (150+)
pytest tests/test_scoring.py  # single file
pytest -x                 # stop on first failure
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting (config in `pyproject.toml`).

```bash
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

## Pull requests

- Keep changes focused; one logical change per PR
- Add tests for new functionality
- Run `make test && make lint` before pushing
