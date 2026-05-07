(byob-benchmark-decorator)=

# Benchmark Decorator

The `@benchmark` and `@scorer` decorators are the user-facing API for defining BYOB benchmarks. Stack `@benchmark` (outer) on top of `@scorer` (inner) to register a scoring function with its dataset, prompt, and configuration.

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

@benchmark(name="my-qa", dataset="data.jsonl", prompt="Q: {question}\nA:")
@scorer
def check(sample: ScorerInput) -> dict:
    return {"correct": sample.target.lower() in sample.response.lower()}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Human-readable benchmark name |
| `dataset` | `str` | required | Path to JSONL file or `hf://` URI |
| `prompt` | `str` | required | Format string with `{field}` placeholders, or path to template file |
| `target_field` | `str` | `"target"` | Dataset field containing ground truth |
| `endpoint_type` | `str` | `"chat"` | `"chat"`, `"completions"`, or `"completions_logprob"` |
| `requirements` | `list` or `str` | `None` | Pip deps (list or path to requirements.txt) |
| `field_mapping` | `dict` | `None` | Maps source columns to prompt field names |
| `extra` | `dict` | `None` | Framework-specific params (judge config, etc.) |
| `response_field` | `str` | `None` | JSONL field with pre-generated responses (eval-only mode) |
| `system_prompt` | `str` | `None` | System prompt string or path to template file |
| `choices` | `list[str]` | `None` | Static candidate continuations for `endpoint_type="completions_logprob"` |
| `choices_field` | `str` | `None` | Dataset field containing per-row candidate continuations for `endpoint_type="completions_logprob"`; dotted paths such as `choices.text` are supported |
| `num_fewshot` | `int` | `0` | Number of few-shot examples to prepend to each prompt |
| `fewshot_split` | `str` | `None` | Optional split to sample few-shot examples from |
| `fewshot_template` | `str` | `None` | Optional template for rendering few-shot examples |
| `fewshot_separator` | `str` | `"\n\n"` | Separator between rendered few-shot examples |

## Name Normalization

The `name` parameter is normalized to create a valid Python identifier used in the compiled package and `eval_type` string. The rules are:

1. Lowercase the name
2. Replace non-alphanumeric characters with underscores
3. Collapse consecutive underscores
4. Strip leading and trailing underscores
5. Truncate to 50 characters

For example, `"My QA Benchmark!"` becomes `"my_qa_benchmark"`.

:::{warning}
If a name normalizes to an empty string (for example, `"!!!"`) the decorator raises a `ValueError`. Use a name containing at least one alphanumeric character.
:::

## Prompt Templates

BYOB supports three ways to define prompts.

### Inline format strings

Pass a Python format string directly. Placeholders use `{field}` syntax and are filled from dataset columns (after any `field_mapping` is applied).

```python
@benchmark(
    name="trivia",
    dataset="trivia.jsonl",
    prompt="Q: {question}\nA:",
)
```

### File-based templates

Paths ending in `.txt`, `.md`, `.jinja`, or `.jinja2` are read from disk. Relative paths resolve from the benchmark file's directory.

```python
@benchmark(
    name="trivia",
    dataset="trivia.jsonl",
    prompt="prompts/trivia.txt",
)
```

### Jinja2 templates

Jinja2 rendering is activated when:

- The prompt contains `{%` (block tags) or `{#` (comments)
- The file has a `.jinja` or `.jinja2` extension

Variable-only Jinja2 templates (`{{ var }}` without block tags) require a `.jinja` or `.jinja2` file extension to be detected.

## Decorator Stacking

`@benchmark` must be the outer (top) decorator and `@scorer` must be the inner (bottom) decorator. The `@scorer` decorator validates the function signature and marks it as a scorer. The `@benchmark` decorator then wraps the marked function and registers it in the benchmark registry.

```python
@benchmark(...)   # outer: registers the benchmark
@scorer           # inner: validates and marks the function
def my_scorer(sample: ScorerInput) -> dict:
    ...
```

The `@scorer` decorator accepts functions with one or two parameters:

- **1 parameter** (preferred): `def scorer(sample: ScorerInput)`
- **2 parameters**: `def scorer(sample, config)`

Functions with 0 or 3+ parameters are rejected with a `TypeError`.

## Eval-Only Mode

Set `response_field` to skip model inference and read responses directly from the dataset. This is useful for evaluating pre-generated outputs or comparing models offline.

```python
@benchmark(
    name="eval-only",
    dataset="responses.jsonl",
    prompt="Q: {question}\nA:",
    response_field="model_output",
)
@scorer
def check(sample: ScorerInput) -> dict:
    return {"correct": sample.target.lower() in sample.response.lower()}
```

The dataset must include the field specified by `response_field`:

```json
{"question": "Is the sky blue?", "answer": "yes", "model_output": "Yes, the sky is blue."}
```

## System Prompts

Use `system_prompt` to prepend a system message to model calls. The value can be an inline string or a path to a template file (same resolution rules as `prompt`).

```python
@benchmark(
    name="code-review",
    dataset="code.jsonl",
    prompt="{code_snippet}",
    system_prompt="You are an expert code reviewer. Be concise.",
)
@scorer
def review(sample: ScorerInput) -> dict:
    return {"has_feedback": len(sample.response.strip()) > 0}
```

:::{note}
System prompts support Jinja2 templates with the same detection rules as user prompts.
:::

## Logprob Multiple-Choice Benchmarks

Use `endpoint_type="completions_logprob"` when the benchmark should score
candidate answers by likelihood instead of asking the model to generate a
free-form answer. This mode calls an OpenAI-compatible `/v1/completions`
endpoint with `max_tokens=0`, `echo=true`, and `logprobs=1`.

Static choices:

```python
@benchmark(
    name="mmlu-mini",
    dataset="data.jsonl",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
    endpoint_type="completions_logprob",
    choices=[" A", " B", " C", " D"],
)
```

Per-row choices, including nested HuggingFace fields:

```python
@benchmark(
    name="arc-mini",
    dataset="hf://my-org/arc-hi?split=test",
    prompt="Question: {{question}}\nAnswer:",
    target_field="answerKey",
    endpoint_type="completions_logprob",
    choices_field="choices.text",
)
```

## See Also

- {ref}`byob` -- BYOB overview and quickstart
- {ref}`byob-scorers` -- Built-in scorers and custom scoring functions
- {ref}`byob-datasets` -- Dataset formats, HuggingFace URIs, and field mapping
