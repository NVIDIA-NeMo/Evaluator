(byob)=

# Bring Your Own Benchmark (BYOB)

Create custom evaluation benchmarks for NeMo Evaluator with few lines of Python code.

**New to BYOB?** See the {ref}`quickstart below <byob-quickstart>` to create your first benchmark.

## Prerequisites

- Python 3.10+
- Activated virtual environment with NeMo Evaluator installed: `pip install nemo-evaluator`
- An OpenAI-compatible model endpoint

(byob-quickstart)=

## Quickstart (5 minutes)

The following walkthrough uses the [MedMCQA](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator/examples/byob/medmcqa/) example -- a medical multiple-choice QA benchmark sourced from HuggingFace.

### Step 1 -- Write your benchmark

Create a `benchmark.py` file (see [`medmcqa/benchmark.py`](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator/examples/byob/medmcqa/benchmark.py) for the full source). It should define the dataset source, prompt template, and scoring logic for the evaluation.

> **MedMCQA dataset fields:** `question` (exam question) · `opa`..`opd` (answer options A-D) · `cop` (correct option index, 0-3)
>
> The benchmark below uses `field_mapping` to rename `opa`..`opd` to `a`..`d` for cleaner prompt placeholders, and `cop` as the target field.
>
> The `datasets` pip package is listed in `requirements`  in order to enable access to the HuggingFace dataset.

```python
import re

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

# Map HF integer answer codes to letters
_COP_TO_LETTER = {"0": "A", "1": "B", "2": "C", "3": "D"}

@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt=(
        "You are a medical expert taking a licensing examination.\n\n"
        "Question: {question}\n\n"
        "A) {a}\n"
        "B) {b}\n"
        "C) {c}\n"
        "D) {d}\n\n"
        "Answer with just the letter (A, B, C, or D):"
    ),
    target_field="cop",
    endpoint_type="chat",
    requirements=["datasets"],
    field_mapping={"opa": "a", "opb": "b", "opc": "c", "opd": "d"},
)
@scorer
def medmcqa_scorer(sample: ScorerInput) -> dict:
    response_clean = sample.response.strip()

    # Try: first character is A-D
    if response_clean and response_clean[0].upper() in "ABCD":
        predicted = response_clean[0].upper()
    else:
        # Try: find "answer is X" or standalone letter
        match = re.search(
            r"(?:answer\s+is\s+|^\s*\(?)\s*([A-Da-d])\b",
            response_clean,
            re.IGNORECASE,
        )
        if match:
            predicted = match.group(1).upper()
        else:
            # Last resort: find any standalone A-D in first 50 chars
            match = re.search(r"\b([A-Da-d])\b", response_clean[:50])
            predicted = match.group(1).upper() if match else ""

    # Convert HF integer target (0-3) to letter (A-D)
    target_str = str(sample.target).strip()
    target_letter = _COP_TO_LETTER.get(target_str, target_str.upper())

    return {
        "correct": predicted == target_letter,
        "parsed": bool(predicted),
    }
```

### Step 2 -- Compile

From the [`medmcqa/`](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator/examples/byob/medmcqa/) directory:

```bash
nemo-evaluator-byob benchmark.py
```

This compiles and auto-installs the package via `pip install` (no PYTHONPATH setup needed).

### Step 3 -- Run

```bash
nemo-evaluator run_eval \
  --eval_type byob_medmcqa.medmcqa \
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
