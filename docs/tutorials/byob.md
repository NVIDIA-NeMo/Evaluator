# Write Your Own Benchmark (BYOB)

Define a complete benchmark with `@benchmark` + `@scorer` in under 10 lines.

## Minimal Example

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, exact_match

@benchmark(
    name="my-bench",
    dataset="hf://my-org/my-dataset?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
```

Run it:

```bash
nel eval run --bench my-bench
```

That is the entire workflow. The `@benchmark` decorator registers an environment, loads
the dataset, formats prompts, and wires scoring. No subclass boilerplate required.

## How It Works

```{mermaid}
flowchart LR
    A["@benchmark"] --> B["ByobEnvironment"]
    B --> C["seed(idx)"]
    B --> D["verify(response, expected)"]
    C --> E["Prompt + Expected Answer"]
    D --> F["@scorer function"]
    F --> G["Reward + Scoring Details"]
```

1. `@benchmark` creates a `ByobEnvironment` (subclass of `EvalEnvironment`) and registers it by name.
2. On `seed(idx)`, the environment loads the dataset row, formats the prompt template, and returns a `SeedResult`.
3. On `verify(response, expected)`, it calls your `@scorer` function with a `ScorerInput` and converts the result to a `VerifyResult`.

## Step-by-Step

### Step 1: Create the benchmark file

Create `src/nemo_evaluator/benchmarks/my_reasoning.py`:

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, answer_line

@benchmark(
    name="my_reasoning",
    dataset="hf://your-org/reasoning-2026?split=test",
    prompt=(
        "Solve the following problem step by step.\n\n"
        "{question}\n\n"
        "Put your final answer after 'Answer:'."
    ),
    target_field="answer",
)
@scorer
def my_reasoning_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample)
```

### Step 2: Register the import

Add the import to `src/nemo_evaluator/benchmarks/__init__.py`:

```python
from nemo_evaluator.benchmarks.my_reasoning import my_reasoning_scorer
```

### Step 3: Validate

```bash
nel validate -b my_reasoning --samples 10
```

Expected output:

```
my_reasoning: 10 samples
  7/10 correct
  [PASS] p0: expected='42' got='42' (1230ms 156tok)
  [PASS] p1: expected='7/3' got='7/3' (980ms 134tok)
  [FAIL] p2: expected='256' got='512' (1100ms 201tok)
  ...
```

### Step 4: Run full evaluation

```bash
nel eval run --bench my_reasoning --repeats 4 --output-dir ./results/my_reasoning
```

### Step 5: Serve for Gym training

```bash
nel serve -b my_reasoning -p 9090
```

Gym training connects at `http://hostname:9090`.

## Decorator Reference

### `@benchmark` parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Environment name used in `nel eval run --bench <name>` |
| `dataset` | `str \| Callable` | Yes | HuggingFace URI (`hf://...`), local JSONL path, or callable returning `list[dict]` |
| `prompt` | `str` | Yes | Python format string using dataset field names |
| `target_field` | `str` | No | Dataset field containing the expected answer (default: `"target"`) |
| `endpoint_type` | `str` | No | `"chat"` or `"completion"` (default: `"chat"`). In YAML configs, protocol is set on the service instead. |
| `system_prompt` | `str` | No | System message prepended to the conversation |
| `field_mapping` | `dict` | No | Rename dataset fields before prompt formatting |
| `prepare_row` | `Callable` | No | `(row, idx, rng) -> row` -- transform each dataset row after loading |
| `seed_fn` | `Callable` | No | `(row, idx) -> SeedResult` -- fully custom seed (overrides prompt template) |
| `extra` | `dict` | No | Arbitrary config passed to scorer via `sample.config` |
| `requirements` | `list[str]` | No | Python packages required by this benchmark |

### `@scorer` function signature

```python
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    ...
```

The function receives a `ScorerInput`:

| Field | Type | Description |
|-------|------|-------------|
| `sample.response` | `str` | Model's raw response text |
| `sample.target` | `Any` | Expected answer from the dataset |
| `sample.metadata` | `dict` | Row metadata (all non-target fields) |
| `sample.config` | `dict` | The `extra` dict from `@benchmark` |

Return a dict. The key `"correct"` (or `"reward"`) is converted to the numeric reward.
Optionally include `"extracted"` for the extracted answer string.

## Dataset Specs

| Format | Example |
|--------|---------|
| HuggingFace | `"hf://cais/mmlu?config=all&split=test"` |
| Local JSONL | `"data/my_bench.jsonl"` |
| Callable | `lambda: [{"q": "2+2", "a": "4"}]` |

HuggingFace URIs support `split` and `config` query parameters.

## Scoring Primitives

Built-in functions you can call from your scorer:

| Function | Use case |
|----------|----------|
| `exact_match(sample)` | Normalized string equality |
| `multichoice_regex(sample)` | Extract A-D/A-J from "Answer: X" pattern |
| `answer_line(sample)` | Extract answer after "Answer:" line |
| `numeric_match(sample)` | Last number in response |
| `fuzzy_match(sample)` | Substring containment with aliases |
| `code_sandbox(sample)` | Docker-sandboxed code execution |
| `needs_judge(sample)` | Flag for LLM-as-judge post-processing |

All are importable from the top-level package:

```python
from nemo_evaluator import exact_match, multichoice_regex, numeric_match
```

## Extension Hooks

### `prepare_row`: Transform dataset rows

Use `prepare_row` when the raw dataset needs restructuring before prompt formatting.

```python
def shuffle_choices(row, idx, rng):
    """Shuffle GPQA answer choices and track the new correct index."""
    choices = [row["Correct Answer"], row["Incorrect Answer 1"],
               row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng.shuffle(choices)
    correct_idx = choices.index(row["Correct Answer"])
    return {**row, "A": choices[0], "B": choices[1],
            "C": choices[2], "D": choices[3],
            "answer": "ABCD"[correct_idx]}

@benchmark(
    name="gpqa",
    dataset="hf://Idavidrein/gpqa?config=gpqa_diamond&split=train",
    prompt=PROMPT_TEMPLATE,
    target_field="answer",
    prepare_row=shuffle_choices,
)
@scorer
def gpqa_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample)
```

### `seed_fn`: Fully custom seed

When you need complete control over prompt construction:

```python
def custom_seed(row, idx):
    from nemo_evaluator import SeedResult
    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": row["problem"]},
    ]
    return SeedResult(
        prompt=row["problem"],
        expected_answer=row["answer"],
        messages=messages,
        system="You are a math tutor.",
    )

@benchmark(name="custom", dataset="hf://...", prompt="", seed_fn=custom_seed)
@scorer
def custom_scorer(sample: ScorerInput) -> dict:
    return numeric_match(sample)
```

## Real-World Examples

All 15 built-in benchmarks use `@benchmark` + `@scorer`. See `src/nemo_evaluator/benchmarks/` for reference implementations:

| Benchmark | Scorer | Key technique |
|-----------|--------|---------------|
| `mmlu.py` | `multichoice_regex` | `prepare_row` to unpack choices |
| `humaneval.py` | `code_sandbox` | Docker-sandboxed test execution |
| `simpleqa.py` | `needs_judge` | LLM-as-judge post-processing |
| `gpqa.py` | `multichoice_regex` | `prepare_row` to shuffle choices |
| `math500.py` | `answer_line` | Math answer extraction |
| `gsm8k.py` | `numeric_match` | Last-number extraction |

## Advanced: Subclass `EvalEnvironment` Directly

For benchmarks that cannot be expressed with decorators (e.g., multi-turn, stateful),
subclass `EvalEnvironment`:

```python
from nemo_evaluator import EvalEnvironment, SeedResult, VerifyResult, register

@register("my_complex_bench")
class MyComplexBenchmark(EvalEnvironment):
    def __init__(self):
        super().__init__()
        self._dataset = [...]

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        return SeedResult(prompt=row["prompt"], expected_answer=row["answer"])

    async def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        correct = response.strip() == expected.strip()
        return VerifyResult(reward=1.0 if correct else 0.0)
```

The decorator path is preferred for single-turn benchmarks. Reserve subclassing for
cases that genuinely need it.
