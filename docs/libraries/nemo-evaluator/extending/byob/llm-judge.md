(byob-llm-judge)=

# LLM-as-Judge

Use LLM-as-Judge to evaluate subjective qualities like truthfulness, safety, and response quality using a judge model.

## Quick Example

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput
from nemo_evaluator.contrib.byob.judge import judge_score

@benchmark(
    name="qa-judge",
    dataset="qa.jsonl",
    prompt="Answer: {question}",
    extra={
        "judge": {
            "url": "https://integrate.api.nvidia.com/v1",
            "model_id": "meta/llama-3.1-70b-instruct",
            "api_key": "NVIDIA_API_KEY",
        },
    },
)
@scorer
def qa_judge(sample: ScorerInput) -> dict:
    return judge_score(sample, template="binary_qa", criteria="Factual accuracy")
```

The judge configuration is passed through the `extra` parameter of `@benchmark` and becomes available inside the scorer via `sample.config`.

## Judge Configuration

The `extra={"judge": {...}}` dict configures the judge model endpoint. Field names align with the nemo-skills `extra.judge` convention used across NeMo Evaluator containers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | required | Base URL of the judge model endpoint |
| `model_id` | `str` | required | Judge model identifier |
| `api_key` | `str` | `None` | Environment variable name containing the API key |
| `temperature` | `float` | `0.0` | Sampling temperature (0.0 for deterministic output) |
| `top_p` | `float` | `1.0` | Nucleus sampling parameter |
| `max_new_tokens` | `int` | `4096` | Maximum tokens for the judge response |
| `parallelism` | `int` | `None` | Max concurrent judge requests (informational) |
| `max_retries` | `int` | `16` | Maximum retry attempts on transient failures (429, 5xx) |
| `request_timeout` | `int` | `600` | Request timeout in seconds |

:::{note}
The `api_key` field is the **name of an environment variable**, not the key itself. For example, setting `"api_key": "NVIDIA_API_KEY"` causes the framework to read the actual key from `os.environ["NVIDIA_API_KEY"]` at runtime.
:::

## Built-in Templates

BYOB ships with four judge prompt templates. Each template instructs the judge model to provide chain-of-thought reasoning before emitting a structured grade line.

| Template | Grade Pattern | Score Mapping | Use Case |
|----------|---------------|---------------|----------|
| `"binary_qa"` | `GRADE: C` or `I` | `C->1.0, I->0.0` | Binary correct/incorrect |
| `"binary_qa_partial"` | `GRADE: C`, `P`, or `I` | `C->1.0, P->0.5, I->0.0` | Correct/partial/incorrect |
| `"likert_5"` | `GRADE: 1-5` | `1->0.2, 2->0.4, 3->0.6, 4->0.8, 5->1.0` | 5-point quality scale |
| `"safety"` | `GRADE: SAFE` or `UNSAFE` | `SAFE->1.0, UNSAFE->0.0` | Safety evaluation |

All built-in templates use the placeholders `{question}`, `{response}`, `{reference}`, and `{criteria}`. The framework fills these automatically from `sample.metadata`, `sample.response`, `sample.target`, and the `criteria` argument.

## Custom Templates

Pass a raw string as the `template` argument to use your own prompt. Use `grade_pattern` and `score_mapping` to tell the parser how to extract and map the grade. Any extra placeholders are filled via `**template_kwargs`.

```python
CUSTOM_TEMPLATE = """\
Question: {question}
Response: {response}
Reference: {reference}

Custom Criteria: {my_criteria}

Output GRADE: PASS or GRADE: FAIL
"""

@scorer
def custom_judge(sample: ScorerInput) -> dict:
    return judge_score(
        sample,
        template=CUSTOM_TEMPLATE,
        grade_pattern=r"GRADE:\s*(PASS|FAIL)",
        score_mapping={"PASS": 1.0, "FAIL": 0.0},
        my_criteria="Check for factual accuracy and completeness",
    )
```

:::{tip}
The `{question}` placeholder is automatically resolved from `sample.metadata["question"]` (falling back to `sample.metadata["prompt"]`). The `{response}` and `{reference}` placeholders map to `sample.response` and `sample.target` respectively.
:::

## judge_score API

The `judge_score` function is the primary entry point for judge-based scoring. Import it from `nemo_evaluator.contrib.byob.judge`.

```python
def judge_score(
    sample: ScorerInput,
    template: str = "binary_qa",
    criteria: str = "",
    grade_pattern: Optional[str] = None,
    score_mapping: Optional[Dict[str, float]] = None,
    judge_key: str = "judge",
    response_format: Optional[Dict[str, Any]] = None,
    **template_kwargs: Any,
) -> dict:
```

| Parameter | Description |
|-----------|-------------|
| `sample` | The `ScorerInput` instance. Must have `sample.config[judge_key]` containing judge endpoint configuration. |
| `template` | Built-in template name (e.g. `"binary_qa"`) or a custom template string with placeholders. |
| `criteria` | Evaluation criteria injected into the template's `{criteria}` placeholder. |
| `grade_pattern` | Regex with one capture group for the grade. Defaults to the built-in pattern for named templates, or `r"GRADE:\s*(\S+)"` for custom templates. |
| `score_mapping` | Dict mapping grade strings to numeric scores. Defaults to the built-in mapping for named templates. |
| `judge_key` | Key in `sample.config` containing the judge config dict. Use `"judge_1"`, `"judge_2"`, etc. for multi-judge setups. |
| `response_format` | Optional dict for constrained decoding (e.g. `{"type": "json_object"}`). When set, `parse_grade` uses structured JSON parsing. |
| `**template_kwargs` | Extra variables passed to the template. Override default variables or fill custom placeholders. |

**Returns:** `{"judge_score": float, "judge_grade": str}`

**Fallback values on failure:**

| Failure | `judge_grade` | `judge_score` |
|---------|---------------|---------------|
| HTTP or network error | `"CALL_ERROR"` | `0.0` |
| Grade not parseable from response | `"PARSE_ERROR"` | `0.0` |

:::{warning}
If the grade string is not found in `score_mapping` and is not a valid number, the score defaults to `0.0`. Always verify that your `grade_pattern` and `score_mapping` cover all possible judge outputs.
:::

## Multi-Judge Setup

Use multiple judge models by assigning different keys in `extra`. Each key holds an independent judge configuration.

```python
@benchmark(
    name="multi-judge",
    dataset="qa.jsonl",
    prompt="Answer: {question}",
    extra={
        "judge": {"url": "http://judge1:8000/v1", "model_id": "judge-a"},
        "judge_1": {"url": "http://judge2:8000/v1", "model_id": "judge-b"},
    },
)
@scorer
def multi(sample: ScorerInput) -> dict:
    a = judge_score(sample, template="binary_qa")
    b = judge_score(sample, template="likert_5", judge_key="judge_1")
    return {**a, "quality": b["judge_score"]}
```

The `judge_key` parameter in `judge_score` selects which configuration to use. The default key is `"judge"`. Name additional judges `"judge_1"`, `"judge_2"`, and so on.

:::{tip}
Each judge endpoint gets its own HTTP session with retry logic, so transient failures on one judge do not block the other.
:::

## See Also

- {ref}`byob-scorers` -- Built-in scorers and custom scoring functions
- {ref}`byob` -- BYOB overview and quickstart
