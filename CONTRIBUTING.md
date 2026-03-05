# Contributing to NeMo Evaluator

## Setup

```bash
git clone <repo-url> && cd nemo-evaluator
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

## Adding a benchmark

1. Create `src/nemo_evaluator/benchmarks/your_bench.py`
2. Subclass `EvalEnvironment`, implement `seed()` and `verify()`
3. Decorate the class with `@register("your_bench")`

```python
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register

@register("my_benchmark")
class MyBenchmark(EvalEnvironment):
    name = "my_benchmark"

    def __init__(self):
        super().__init__()
        self._dataset = [...]  # load your data

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

## Running tests

```bash
pytest                    # all tests
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
