---
name: byob-benchmark
description: Use when adding a new Bring-Your-Own-Benchmark (BYOB) to nemo-evaluator — writes a `@benchmark` + `@scorer` module under `src/nemo_evaluator/benchmarks/`, registers it in `__init__.py`, and validates with `nel validate`. Trigger phrases include "add a benchmark", "new BYOB benchmark", "create an evaluator benchmark", "write a `@benchmark` scorer".
---

# BYOB benchmark authoring

Use this skill when the user asks to add a new benchmark to nemo-evaluator. The deeper human-facing tutorial lives at `docs/tutorials/byob.md`; this file is the agent recipe.

## When to use

Use BYOB for **single-turn** benchmarks that fit the shape

```
dataset row -> prompt template (or seed_fn) -> model response -> scorer -> reward
```

Do **not** use BYOB for:

- Multi-turn agentic tasks that need a container workspace, task manifests, or per-task Docker images. Subclass `HarborEnvironment` instead and use `@register(name)` directly. References: `src/nemo_evaluator/benchmarks/nmp_harbor.py`, `src/nemo_evaluator/benchmarks/terminal_bench_v1.py`.
- Benchmarks that delegate scoring/aggregation to an external server. That violates the "evaluator owns the loop" principle in `AGENTS.md`.

## Five-step recipe

1. **Write** `src/nemo_evaluator/benchmarks/<name>.py`. It must define one `@benchmark(...)` + `@scorer` pair (see skeleton below).
2. **Register** it by appending one line to `src/nemo_evaluator/benchmarks/__init__.py`:
   ```python
   import nemo_evaluator.benchmarks.<name>  # noqa: F401
   ```
   Keep the list alphabetically sorted. The decorator fires on import.
3. **Sanity-check** against a live model:
   ```bash
   nel validate -b <name> --samples 10
   ```
4. **Add a unit test** under `tests/test_scoring/` that feeds synthetic `ScorerInput` into your scorer. Follow the pattern in `tests/test_scoring/test_benchmark_definitions.py`. Offline tests only — do not hit HuggingFace.
5. **Pre-commit** (per `AGENTS.md`):
   ```bash
   pytest tests/ -v --tb=short -m "not network" -x
   ```
   Then add a `CHANGELOG.md` entry under the current version. Commit with `feat: [EVAL-XXXX] <description>` (or `chore:`/`fix:` as appropriate).

## Skeleton

```python
# src/nemo_evaluator/benchmarks/my_bench.py
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, exact_match


@benchmark(
    name="my_bench",
    dataset="hf://my-org/my-dataset?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def my_bench_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
```

The top-level package re-exports `benchmark`, `scorer`, `ScorerInput`, and every scoring primitive, so `from nemo_evaluator import benchmark, scorer, ScorerInput, exact_match` also works.

## `@benchmark` parameter reference

Source of truth: `src/nemo_evaluator/environments/custom.py:256-283`.

| Param | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Registry key (lowercased). Used by `nel eval run --bench <name>`. |
| `dataset` | `str \| Callable` | required | See "Dataset specs" below. |
| `prompt` | `str` | `""` | Python format string. `{field}` names come from the loaded row after `field_mapping` / `prepare_row`. Can be a path to a `.txt` / `.md` / `.jinja` file. |
| `target_field` | `str` | `"target"` | Row key holding the expected answer. |
| `endpoint_type` | `str` | `"chat"` | `"chat"` or `"completion"`. Protocol normally set on the service in YAML. |
| `system_prompt` | `str \| None` | `None` | Prepended as a system message. |
| `field_mapping` | `dict[str,str] \| None` | `None` | Rename source keys to template placeholder names. |
| `extra` | `dict` | `{}` | Opaque dict forwarded to the scorer as `sample.config` (e.g. `{"judge": True}`). |
| `requirements` | `list[str] \| None` | `None` | Python package requirements metadata. |
| `prepare_row` | `Callable[[row, idx, rng], row]` | `None` | Transform each row after load. `rng` is a seeded `random.Random(42)` — safe for shuffling. |
| `seed_fn` | `Callable[[row, idx], SeedResult]` | `None` | Overrides the template path entirely. Use for multi-message prompting or per-row sandbox specs. |

Note: `@image_builder` exists (`custom.py:309-328`) for benchmarks that need per-dataset Docker images. Stack it between `@benchmark` and `@scorer`. Rare — reach for it only when the scorer requires a prebuilt image (e.g. SWE-bench-style verifier containers).

## Dataset specs

Resolved in `_load_dataset_from_spec` (`custom.py:85`):

| Form | Example |
|---|---|
| HuggingFace | `"hf://cais/mmlu?config=all&split=test"` — `split` and `config` query params supported; `num_examples` slices via `split=test[:N]` |
| Local JSONL | `"data/my_bench.jsonl"` |
| Local CSV/TSV | `"data/my_bench.csv"` / `".tsv"` (auto-detected by suffix) |
| Callable | `lambda: [...]` or `def load(num_examples=None) -> list[dict]` |

The loader passes `num_examples=N` to callables that declare the kwarg.

## Scorer picker

All importable from the top-level `nemo_evaluator` package.

| Scorer | When to pick it | Example in repo |
|---|---|---|
| `exact_match` | Case-insensitive, article-stripped equality | `xstest.py` |
| `answer_line(sample, pattern=...)` | Pull a line after "Answer:" or custom pattern | `gsm8k.py`, `math500.py` |
| `multichoice_regex(sample, pattern=...)` | Extract A–D/A–J letter | `mmlu.py`, `mmlu_pro.py`, `gpqa.py` |
| `numeric_match` | Last number in the response | math/arithmetic cases |
| `fuzzy_match` | Substring match with alias list (use `extra={"aliases": [...]}`) | `triviaqa.py` |
| `code_sandbox` | Run code in a Docker sandbox (expects `sandbox`) | `humaneval.py` |
| `needs_judge` | LLM-as-judge — pair with `extra={"judge": True}` | `simpleqa.py`, `healthbench.py` |

The scorer must return a dict. The engine reads `correct` (bool) or `reward` (float); `extracted` is optional and surfaces as the extracted-answer string. Any other keys flow into `scoring_details`. Do **not** return both `correct` and `reward`. Async scorers are supported (the engine awaits coroutines — see `custom.py:242`).

## Pattern picker

Decide which hook you need before writing the module:

- **Template only** → row keys already match `{placeholders}`. Examples: `xstest.py`, `simpleqa.py`.
- **`prepare_row`** → row shape needs normalizing before templating (unpack choices, rename fields, strip answer boilerplate, shuffle choices with `rng`). Examples: `mmlu.py` (unpack `choices[]` + map integer `answer` to `A`–`D`), `gpqa.py` (shuffle with the supplied `rng` so runs stay reproducible), `gsm8k.py` (regex `####` to recover the gold answer), `humaneval.py` (rename fields).
- **`seed_fn`** → prompt construction can't be a format string (few-shot messages, multi-turn, per-language, per-row sandbox setup). Examples: `drop.py`, `mgsm.py`, `healthbench.py`, `pinchbench.py`.
- **`@image_builder`** → scorer needs a Docker image that must be built from the dataset first. Check the docstring in `custom.py:309-328`.

## Three canonical examples (from the repo)

Template + `prepare_row` with letter extraction (`src/nemo_evaluator/benchmarks/mmlu.py:42-51`):

```python
@benchmark(
    name="mmlu",
    dataset="hf://cais/mmlu?config=all&split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def mmlu_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample)
```

Template + `prepare_row` with line-level scoring (`src/nemo_evaluator/benchmarks/gsm8k.py:34-43`):

```python
@benchmark(
    name="gsm8k",
    dataset="hf://openai/gsm8k?config=main&split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def gsm8k_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample, pattern=r"(?i)(?:the answer is|answer\s*:)\s*([^\n]+)")
```

Pure template, LLM-as-judge (`src/nemo_evaluator/benchmarks/simpleqa.py:21-30`):

```python
@benchmark(
    name="simpleqa",
    dataset="hf://basicv8vc/SimpleQA?split=test",
    prompt="{problem}",
    target_field="answer",
    extra={"judge": True},
)
@scorer
def simpleqa_scorer(sample: ScorerInput) -> dict:
    return needs_judge(sample)
```

## Out-of-tree benchmarks

For a benchmark that should not live in this repo:

- Point `nel` at a loose `.py` file: `nel eval run --bench /path/to/my_bench.py` or `/path/to/my_bench.py:explicit_name`. See `_resolve_file_bench` in `src/nemo_evaluator/environments/registry.py:199`.
- Or use `load_benchmark_file(path)` at runtime — idempotent, returns the list of newly registered names.
- Containerize with `nel package -m <module_dir> -t my-bench:latest` (`src/nemo_evaluator/cli/package.py`). Use `-o Dockerfile` to get the Dockerfile without building.

## Gotchas

- `{field}` placeholders must match the **final** row keys, i.e. after any `field_mapping` + `prepare_row`. A missing key logs a warning and returns the unformatted template — check logs if the model seems to ignore the data.
- `target_field` that doesn't exist on the row becomes an empty string silently. Verify against a tiny HF load before wiring.
- `@scorer` functions must accept a `ScorerInput`, not raw strings. The input exposes `.response`, `.target`, `.metadata` (all non-target row fields), `.config` (from `extra`), and optionally `.sandbox`.
- `name` is lowercased in the registry. Avoid hyphens in favour of underscores to match existing built-ins.
- Don't skip the `__init__.py` import — if the module isn't imported, `@benchmark` never fires and `nel list` won't show it.
- Never `print()` — use `logging.getLogger(__name__)` per `AGENTS.md`.
- If you need `sandbox` access from the scorer, declare it via `seed_fn` (set `sandbox_spec` / `verify_sandbox_spec` on the returned `SeedResult`) so the lifecycle manager attaches one.

## Before committing

Follow the pre-commit checklist in `AGENTS.md`:

1. Lint the files you touched.
2. `pytest tests/ -v --tb=short -m "not network" -x` must pass.
3. `grep` for any old import paths if you renamed or moved something.
4. Add a `CHANGELOG.md` entry under the current version.

Commit message format: `feat: [EVAL-XXXX] add <name> benchmark` (omit the bracket when there is no Linear ticket).
