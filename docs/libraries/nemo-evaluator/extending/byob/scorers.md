(byob-scorers)=

# Scorers

Scorers evaluate model responses against ground truth. BYOB provides built-in scorers for common patterns and supports custom scorer functions.

## ScorerInput

Every scorer receives a single `ScorerInput` dataclass importable from `nemo_evaluator.contrib.byob`:

```python
@dataclass
class ScorerInput:
    response: str              # Model output (or argmax choice in logprob mode)
    target: Any                # Ground truth from dataset
    metadata: dict             # Dataset row + per-call response metadata
    model_call_fn: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    conversation: Optional[List[dict]] = None
    turn_index: Optional[int] = None
```

| Field | Description |
|-------|-------------|
| `response` | The model output text for the current sample. In `completions_logprob` mode this is set to the choice with the highest sum-logprob (i.e. the argmax). |
| `target` | The ground-truth value read from the field specified by `target_field` in `@benchmark`. |
| `metadata` | Shared bag for **dataset-row fields and per-call response metadata**. Standard scorers use it to access any column on the row (e.g. `sample.metadata["passage"]`). Strategies that produce extra per-call data write namespaced keys (prefixed with `_`) into this dict before invoking the scorer. |
| `model_call_fn` | Reserved for multi-turn evaluation (not yet implemented). |
| `config` | Extra configuration passed through `extra=` in `@benchmark` (e.g. judge settings). |
| `conversation` | Reserved for multi-turn benchmarks (not yet implemented). |
| `turn_index` | Reserved for multi-turn benchmarks (not yet implemented). |

### Reserved metadata keys

`MultipleChoiceStrategy` (selected by `endpoint_type="completions_logprob"`) writes the following keys into `ScorerInput.metadata` before invoking the scorer:

| Key | Type | Description |
|-----|------|-------------|
| `_choices` | `list[str]` | Candidate continuations resolved from `choices=` or `choices_field=` on `@benchmark`. |
| `_choices_logprobs` | `list[float]` | Per-choice sum log-probabilities returned by the loglikelihood call. Same length as `_choices`. |
| `_choices_is_greedy` | `list[bool]` | Per-choice booleans: `True` when every continuation token equals the top-1 prediction (i.e. the choice would have been produced under greedy decoding). Same length as `_choices`. |

`response` is also set to `_choices[argmax(_choices_logprobs)]` so legacy text-based scorers continue to work in logprob mode.

## The @scorer Decorator

The `@scorer` decorator marks a function as a BYOB scorer. It validates the function signature at decoration time and sets an internal `_is_scorer` flag used by the framework.

A scorer must return a `dict` with string keys and `bool` or `float` values. These key-value pairs become the reported metrics.

```python
from nemo_evaluator.contrib.byob import scorer, ScorerInput

@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return {"correct": sample.response.strip().lower() == str(sample.target).strip().lower()}
```

## Built-in Scorers

Import built-in scorers from `nemo_evaluator.contrib.byob.scorers`:

| Scorer | Returns | Description |
|--------|---------|-------------|
| `exact_match` | `{"correct": bool}` | Case-insensitive, whitespace-stripped equality |
| `contains` | `{"correct": bool}` | Case-insensitive substring match |
| `f1_token` | `{"f1": float, "precision": float, "recall": float}` | Token-level F1 using Counter intersection |
| `regex_match` | `{"correct": bool}` | Regex pattern match (target is the pattern) |
| `bleu` | `{"bleu_1": float, "bleu_2": float, "bleu_3": float, "bleu_4": float}` | Sentence-level BLEU-1 through BLEU-4 with add-1 smoothing |
| `rouge` | `{"rouge_1": float, "rouge_2": float, "rouge_l": float}` | ROUGE-1, ROUGE-2, ROUGE-L F1 scores |
| `retrieval_metrics` | `{"precision_at_k": float, "recall_at_k": float, "mrr": float, "ndcg": float}` | Retrieval quality metrics |
| `multiple_choice_acc` | `{"acc": float, "acc_norm": float, "acc_greedy": float}` | Multiple-choice loglikelihood ranking. `acc` matches lm-evaluation-harness MMLU-style raw argmax; `acc_norm` is per-byte length-normalized argmax (ARC/BoolQ style); `acc_greedy` is the highest-loglikelihood greedy choice. Requires `endpoint_type="completions_logprob"` and either `choices=` or `choices_field=` on `@benchmark`. |
| `mcq_letter_extract` | `{"correct": bool, "parsed": bool}` | Extracts an A-J letter from free-form text (handles "A", "A)", "The answer is B", "(C)", "Option D", and `\boxed{E}`). Targets may be letters, integer indices, or verbatim choice text from the metadata `a`/`b`/`c`/`d` keys. Empty or `None` responses are treated as unparsed rather than raising. |
| `gsm8k_answer` | `{"correct": bool, "parsed": bool}` | Canonical GSM8K numeric extractor. Tries the `#### <number>` marker first, then `\boxed{<number>}`, then falls back to the last number in the response. Strips commas and normalizes trailing zeros. |
| `boolean_yesno` | `{"correct": bool, "parsed": bool}` | Extracts English yes/no decisions from free-form text. Recognizes tokens such as yes/no/yep/nope/true/false. |
| `chrf` | `{"chrf": float, "chrf_pp": float}` | Sentence-level chrF and chrF++ in [0, 100]. Pure-Python sacrebleu-style formula (character 1- to 6-gram F2; chrF++ adds word 1- and 2-gram F2). |

### Usage example

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput
from nemo_evaluator.contrib.byob.scorers import exact_match

@benchmark(name="my-qa", dataset="data.jsonl", prompt="Q: {question}\nA:", target_field="answer")
@scorer
def check(sample: ScorerInput) -> dict:
    return exact_match(sample)
```

:::{note}
`retrieval_metrics` expects two lists in `sample.metadata`:

- `retrieved` -- ordered list of retrieved item identifiers.
- `relevant` -- list of relevant (ground-truth) item identifiers.
- `k` (optional) -- cut-off depth; defaults to `len(retrieved)`.

Make sure your JSONL dataset includes these fields for every row.
:::

## Writing Custom Scorers

A custom scorer is any function decorated with `@scorer` that accepts a `ScorerInput` and returns a `dict`:

```python
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    # Access model response
    response = sample.response.strip()
    # Access ground truth
    target = str(sample.target).strip()
    # Access additional dataset fields
    category = sample.metadata.get("category", "unknown")

    is_correct = response.lower() == target.lower()
    return {
        "correct": is_correct,
        f"correct_{category}": is_correct,
    }
```

:::{tip}
Return per-category breakdowns by including dynamic keys in the return dict. The aggregation layer will automatically compute means for every unique key across all samples, giving you fine-grained metric slices at no extra cost.
:::

### Combining built-in and custom logic

You can call built-in scorers inside a custom scorer and merge the results:

```python
from nemo_evaluator.contrib.byob.scorers import f1_token, exact_match

@scorer
def combined(sample: ScorerInput) -> dict:
    em = exact_match(sample)
    f1 = f1_token(sample)
    return {**em, **f1}
```

## Multiple-choice loglikelihood ranking

For MMLU-, ARC-, and BoolQ-style benchmarks, BYOB supports per-choice
loglikelihood ranking with **lm-evaluation-harness parity**:

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput
from nemo_evaluator.contrib.byob.scorers import multiple_choice_acc

@benchmark(
    name="mmlu-mini",
    dataset="hf://my-org/my-mmlu?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",                    # gold letter, e.g. "B"
    endpoint_type="completions_logprob",      # enables loglikelihood scoring
    choices=[" A", " B", " C", " D"],         # static candidates per row
    num_fewshot=5,                            # optional fewshot prefix
)
@scorer
def mmlu_score(sample: ScorerInput) -> dict:
    return multiple_choice_acc(sample)        # {acc, acc_norm, acc_greedy}
```

For datasets with **per-row variable choices** (e.g. ARC), set
`choices_field` instead of `choices`:

```python
@benchmark(
    ...,
    choices_field="choices_text",             # row[choices_text] is a list[str]
)
```

Nested/dotted fields are also supported for HuggingFace datasets that store
choices under a struct-like column:

```python
@benchmark(
    ...,
    choices_field="choices.text",             # row["choices"]["text"]
)
```

### How it works

`MultipleChoiceStrategy` (selected automatically when
`endpoint_type="completions_logprob"`) calls the OpenAI-compatible
`/v1/completions` endpoint once per choice, exactly like lm-eval's
`local-completions` adapter:

```text
POST /v1/completions
{
  "model": "...",
  "prompt": "<context><continuation>",
  "max_tokens": 0,
  "logprobs": 1,
  "echo": true,
  "temperature": 0
}
```

The runner inspects `logprobs.text_offset` to locate the continuation
token span, sums `token_logprobs` over that span, and decides
`is_greedy` by checking whether each continuation token matches the
top-1 entry of `top_logprobs`. The resulting per-choice
`(sum_logprob, is_greedy)` tuples are written into `ScorerInput.metadata`
under the reserved keys `_choices`, `_choices_logprobs`, and
`_choices_is_greedy`. `multiple_choice_acc` then computes:

- `acc` -- 1.0 iff `argmax(metadata["_choices_logprobs"]) == gold_index`
  (MMLU canonical).
- `acc_norm` -- 1.0 iff
  `argmax(metadata["_choices_logprobs"][i] /
  max(len(metadata["_choices"][i].encode("utf-8")), 1)) == gold_index`
  (ARC/BoolQ canonical, per-byte length normalization).
- `acc_greedy` -- 1.0 iff the highest-loglikelihood **greedy** choice
  matches gold (diagnostic).

The gold answer can be a letter (`"A"`..`"J"`), an integer index, or
the verbatim choice string -- `multiple_choice_acc` handles all three.

## See Also

- {ref}`byob` -- BYOB overview and quickstart
- {ref}`byob-llm-judge` -- LLM-as-Judge evaluation for subjective criteria
