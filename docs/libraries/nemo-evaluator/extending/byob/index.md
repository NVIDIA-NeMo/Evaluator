(byob)=

# Bring Your Own Benchmark (BYOB)

Create custom evaluation benchmarks in ~12 lines of Python using decorators, built-in scorers, and one-command containerization.

**New to BYOB?** See the {ref}`quickstart below <byob-quickstart>` to create your first benchmark.

## Prerequisites

- Python 3.10+
- NeMo Evaluator installed (`pip install nemo-evaluator`)
- An OpenAI-compatible model endpoint

(byob-quickstart)=

## Quickstart

### Step 1 -- Write your benchmark

Create a file called `my_benchmark.py`:

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

@benchmark(
    name="my-qa",
    dataset="data.jsonl",
    prompt="Q: {question}\nA:",
    target_field="answer",
)
@scorer
def check(sample: ScorerInput) -> dict:
    return {"correct": sample.target.lower() in sample.response.lower()}
```

### Step 2 -- Compile

```bash
nemo-evaluator-byob my_benchmark.py
```

### Step 3 -- Run

```bash
nemo-evaluator run_eval \
  --eval_type byob_my_qa.my-qa \
  --model_url http://localhost:8000 \
  --model_id my-model \
  --model_type chat \
  --output_dir ./results \
  --api_key_name API_KEY
```

:::{tip}
Use `nemo-evaluator-byob my_benchmark.py --dry-run` to validate your benchmark without installing it.
:::

## Reference Documentation

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Benchmark Decorator
:link: byob-benchmark-decorator
:link-type: ref
Define benchmarks with the `@benchmark` decorator.
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Scorers
:link: byob-scorers
:link-type: ref
Built-in scorers and custom scoring functions.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge
:link: byob-llm-judge
:link-type: ref
Judge-based evaluation with LLM models.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Datasets
:link: byob-datasets
:link-type: ref
Dataset formats, HuggingFace URIs, and field mapping.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: byob-cli
:link-type: ref
Compile, validate, list, and containerize benchmarks.
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containerization
:link: byob-containerization
:link-type: ref
Package benchmarks as Docker images.
:::

::::

## Examples

Complete annotated examples are available in the source repository under `packages/nemo-evaluator/examples/byob/`:

- **MedMCQA** -- HuggingFace dataset with field mapping and custom letter-extraction scorer
- **Global MMLU Lite** -- Multilingual MMLU with per-category scoring breakdowns
- **TruthfulQA** -- LLM-as-Judge with custom template and `**template_kwargs`
- **Math Reasoning** -- Numeric extraction with tolerance comparison

## Related Documentation

- {ref}`extending-evaluator` -- Overview of extending NeMo Evaluator
- {ref}`framework-definition-file` -- Framework Definition File reference

:::{toctree}
:maxdepth: 1
:hidden:

Benchmark Decorator <benchmark-decorator>
Scorers <scorers>
LLM-as-Judge <llm-judge>
Datasets <datasets>
CLI Reference <cli>
Containerization <containerization>
:::
