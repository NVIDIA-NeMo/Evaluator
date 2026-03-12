# BYOB (Bring Your Own Benchmark)

Create custom evaluation benchmarks for NeMo Evaluator in ~12 lines of Python.

## Prerequisites

- Python 3.10+
- Activated virtual environment with NeMo Evaluator installed: `pip install nemo-evaluator`
- An OpenAI-compatible model endpoint (or use the mock server for testing)

## Quickstart (5 minutes)

The following walkthrough uses the [MedMCQA](./medmcqa/) example — a medical multiple-choice QA benchmark sourced from HuggingFace.

### Step 1: Write your benchmark

Create a `benchmark.py` file (see [`medmcqa/benchmark.py`](./medmcqa/benchmark.py) for the full source). It should define the dataset source, prompt template, and scoring logic for the evaluation.

> **MedMCQA dataset fields:** `question` (exam question) · `opa`..`opd` (answer options A-D) · `cop` (correct option index, 0-3)
>
> The benchmark below uses `field_mapping` to rename `opa`..`opd` to `a`..`d` for cleaner prompt placeholders, and `cop` as the target field.
>
> The `datasets` pip package is listed in `requirements`  in order to enable access to the HuggingFace dataset.

```python
from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt=(
        "You are a medical expert taking a licensing examination.\n\n"
        "Question: {question}\n\n"
        "A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
        "Answer with just the letter (A, B, C, or D):"
    ),
    target_field="cop",
    endpoint_type="chat",
    requirements=["datasets"],
    field_mapping={"opa": "a", "opb": "b", "opc": "c", "opd": "d"},
)
@scorer
def medmcqa_scorer(sample: ScorerInput) -> dict:
    # Extract letter choice from response and compare to target (0-3 -> A-D)
    predicted = sample.response.strip()[0].upper() if sample.response.strip() else ""
    cop_to_letter = {"0": "A", "1": "B", "2": "C", "3": "D"}
    target_letter = cop_to_letter.get(str(sample.target).strip(), "")
    return {"correct": predicted == target_letter}
```

### Step 2: Compile

From the [`medmcqa/`](./medmcqa/) directory:

```bash
nemo-evaluator-byob benchmark.py
```

This compiles and auto-installs the package via `pip install` (no PYTHONPATH setup needed).

### Step 3: Run

```bash
nemo-evaluator run_eval \
  --eval_type byob_medmcqa.medmcqa \
  --model_url http://localhost:8000 \
  --model_id my-model \
  --model_type chat \
  --output_dir ./results \
  --api_key_name API_KEY
```

## CLI Commands

```bash
# Compile a benchmark (auto-installs)
nemo-evaluator-byob my_benchmark.py

# Validate without installing
nemo-evaluator-byob my_benchmark.py --dry-run

# Compile without auto-install (manual PYTHONPATH required)
nemo-evaluator-byob my_benchmark.py --no-install

# List installed benchmarks
nemo-evaluator-byob --list

# Containerize a benchmark
nemo-evaluator-byob my_benchmark.py --containerize

# Containerize and push to registry
nemo-evaluator-byob my_benchmark.py --push registry.example.com/my-bench:latest

# Show version
nemo-evaluator-byob --version
```

## Built-in Scorers

Import from `nemo_evaluator.contrib.byob.scorers`:


| Scorer              | Returns                                                  | Description                                     |
| ------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| `exact_match`       | `{"correct": bool}`                                      | Case-insensitive, whitespace-stripped equality  |
| `contains`          | `{"correct": bool}`                                      | Case-insensitive substring match                |
| `f1_token`          | `{"f1": float, "precision": float, "recall": float}`     | Token-level F1                                  |
| `regex_match`       | `{"correct": bool}`                                      | Regex pattern match (target is the pattern)     |
| `bleu`              | `{"bleu_1"..4: float}`                                   | Sentence-level BLEU-1 through BLEU-4            |
| `rouge`             | `{"rouge_1": float, "rouge_2": float, "rouge_l": float}` | ROUGE-1, ROUGE-2, ROUGE-L F1                    |
| `retrieval_metrics` | `{"precision_at_k": float, ...}`                         | Retrieval quality metrics (P@k, R@k, MRR, NDCG) |


All built-in scorers accept a single `ScorerInput` argument.

### Scorer Composition

```python
from nemo_evaluator.contrib.byob import any_of, all_of
from nemo_evaluator.contrib.byob.scorers import contains, exact_match

# Correct if EITHER scorer matches
lenient = any_of(contains, exact_match)

# Correct only if BOTH scorers match
strict = all_of(contains, exact_match)
```

## LLM-as-Judge

Use `judge_score()` for subjective evaluation powered by a judge model:

```python
from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput
from nemo_evaluator.contrib.byob.judge import judge_score

@benchmark(
    name="qa-judge",
    dataset="qa.jsonl",
    prompt="Answer: {question}",
    judge={"url": "https://integrate.api.nvidia.com/v1", "model_id": "meta/llama-3.1-70b-instruct", "api_key": "NVIDIA_API_KEY"},
)
@scorer
def qa_judge(sample: ScorerInput) -> dict:
    return judge_score(sample, template="binary_qa", criteria="Factual accuracy")
```

Built-in templates: `binary_qa`, `binary_qa_partial`, `likert_5`, `safety`. Custom templates support `**template_kwargs` for arbitrary placeholders.

## Coding Agent Integration

BYOB works with AI coding agents that can read files and run shell commands. Supported agents include Claude Code, Codex, and others.


| Agent            | Instruction file           | How to start                                 |
| ---------------- | -------------------------- | -------------------------------------------- |
| **Claude Code**  | `.claude/commands/byob.md` | Type `/byob` in any session within the repo  |
| **Codex**        | `examples/byob/AGENTS.md`  | Ask "create a BYOB benchmark" in the repo    |
| **Other agents** | `examples/byob/AGENTS.md`  | Point the agent at the file or at a template |


The agent walks you through 5 steps: understand the task, read data, generate prompt, pick scorer, compile and ship. See [SKILL-USAGE.md](./SKILL-USAGE.md) for the full walkthrough.

## Examples

- [MedMCQA](./medmcqa/) - Medical multiple-choice QA with HuggingFace dataset and field mapping
- [Global MMLU Lite](./global_mmlu_lite/) - Multilingual MMLU with per-category scoring breakdown
- [TruthfulQA](./truthfulqa/) - LLM-as-Judge evaluation with custom template and `**template_kwargs`
- Templates: [Math Reasoning](./templates/math_reasoning.py) (numeric extraction + tolerance)

## Dataset Format

BYOB accepts JSONL (JSON Lines) files or HuggingFace dataset URIs (`hf://org/dataset`).

Each JSONL line is a JSON object:

```jsonl
{"question": "Is the sky blue?", "answer": "yes"}
{"question": "Is water dry?", "answer": "no"}
```

Your prompt template uses `{field}` placeholders that match your dataset fields.

Use `field_mapping` to rename dataset columns. For example, the [MedMCQA](./medmcqa/benchmark.py) benchmark renames HuggingFace fields `opa`..`opd` to `a`..`d`:

```python
@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt="Question: {question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:",
    field_mapping={"opa": "a", "opb": "b", "opc": "c", "opd": "d"},
)
```

## Troubleshooting

### "command not found: nemo-evaluator-byob"

Install the package: `cd packages/nemo-evaluator && pip install -e .`

### "Benchmark not found"

Compilation auto-installs via `pip install`. If you used `--no-install`, add the package to PYTHONPATH:

```bash
export PYTHONPATH="~/.nemo-evaluator/byob_packages/byob_<name>:$PYTHONPATH"
```

### "Dataset file not found"

Use absolute paths or `os.path.join(os.path.dirname(__file__), "data.jsonl")` for paths relative to your benchmark file.

### "Sample N missing field X"

Your prompt template references a field not in the dataset. Check that `{field}` placeholders match your JSONL keys. Use `field_mapping` to rename columns if needed.

### Scorer signature error

BYOB scorers accept a single `ScorerInput` argument. Migrate from the old 3-argument pattern:

```python
# Old (no longer supported):
# def my_scorer(response, target, metadata): ...

# New:
def my_scorer(sample: ScorerInput) -> dict:
    return {"correct": sample.target.lower() in sample.response.lower()}
```

### "Connection refused" or "Model timeout"

Check that your model endpoint is running and accessible. Use `curl` to verify:

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "test"}]}'
```

Set a custom timeout with `--timeout-per-sample 300` for slow models.

### All samples skipped (0% accuracy)

If every sample is skipped, check your model endpoint and API key. Use `--fail-on-skip` in CI to catch this:

```bash
python -m nemo_evaluator.contrib.byob.runner ... --fail-on-skip
```

