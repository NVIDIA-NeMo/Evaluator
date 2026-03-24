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

## Adding a benchmark (recommended: decorator API)

1. Create `src/nemo_evaluator/benchmarks/your_bench.py`
2. Use `@benchmark` + `@scorer`:

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, answer_line

@benchmark(
    name="my_bench",
    dataset="hf://my-org/my-data?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample)
```

3. Import it in `src/nemo_evaluator/benchmarks/__init__.py`:

```python
import nemo_evaluator.benchmarks.your_bench  # noqa: F401
```

4. Verify: `nel list` should show your benchmark
5. Test: `nel validate -b my_bench --model-url ... --model-id ...`

## Adding a benchmark (advanced: subclass)

For benchmarks that need full control (multi-turn, stateful), subclass `EvalEnvironment`:

```python
from nemo_evaluator.environments import EvalEnvironment, SeedResult, VerifyResult, register

@register("my_complex_bench")
class MyComplexBench(EvalEnvironment):
    def __init__(self):
        super().__init__()
        self._dataset = [...]

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        return SeedResult(prompt=row["question"], expected_answer=row["answer"])

    async def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        correct = response.strip() == expected
        return VerifyResult(reward=1.0 if correct else 0.0, extracted_answer=response.strip())
```

## Adding a scorer

1. Create `src/nemo_evaluator/scoring/your_scorer.py`
2. Export from `src/nemo_evaluator/scoring/__init__.py`
3. Add tests in `tests/test_your_scorer.py`

Available scorers:

| Module | Functions |
|--------|-----------|
| `scoring/text.py` | `exact_match`, `fuzzy_match` |
| `scoring/pattern.py` | `multichoice_regex`, `answer_line`, `numeric_match` |
| `scoring/code_execution.py` | `code_sandbox` |
| `scoring/judge.py` | `needs_judge`, `judge_score` |
| `scoring/json_schema.py` | `extract_json`, `validate_json_schema` |

## Running tests

```bash
pytest                    # all tests
pytest tests/test_benchmark_definitions.py  # single file
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
