(byob-scorers)=

# Scorers

Scorers evaluate model responses against ground truth. BYOB provides built-in scorers for common patterns and supports custom scorer functions.

## ScorerInput

Every scorer receives a single `ScorerInput` dataclass importable from `nemo_evaluator.contrib.byob`:

```python
@dataclass
class ScorerInput:
    response: str              # Model output
    target: Any                # Ground truth from dataset
    metadata: dict             # Full dataset row as a dict
    model_call_fn: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    conversation: Optional[List[dict]] = None
    turn_index: Optional[int] = None
```

| Field | Description |
|-------|-------------|
| `response` | The model output text for the current sample. |
| `target` | The ground-truth value read from the field specified by `target_field` in `@benchmark`. |
| `metadata` | The entire dataset row as a dictionary, useful for accessing additional fields beyond the target. |
| `model_call_fn` | Reserved for multi-turn evaluation (not yet implemented). |
| `config` | Extra configuration passed through `extra=` in `@benchmark` (e.g. judge settings). |
| `conversation` | Reserved for multi-turn benchmarks (not yet implemented). |
| `turn_index` | Reserved for multi-turn benchmarks (not yet implemented). |

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

## See Also

- {ref}`byob` -- BYOB overview and quickstart
- {ref}`byob-llm-judge` -- LLM-as-Judge evaluation for subjective criteria
