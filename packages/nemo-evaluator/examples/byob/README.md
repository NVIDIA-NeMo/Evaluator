# BYOB (Bring Your Own Benchmark)

Create custom evaluation benchmarks for NeMo Evaluator in ~12 lines of Python.

## Prerequisites

- Python 3.10+
- NeMo Evaluator installed: `pip install -e .` from `packages/nemo-evaluator`
- An OpenAI-compatible model endpoint (or use the mock server for testing)

## Quickstart (5 minutes)

### Step 1: Write your benchmark

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

### Step 2: Compile

```bash
nemo-evaluator-byob my_benchmark.py
```

This compiles and auto-installs the package via `pip install -e` (no PYTHONPATH setup needed).

### Step 3: Run

```bash
nemo-evaluator run_eval \
  --eval_type byob_my_qa.my-qa \
  --model_url http://localhost:8000 \
  --model_id my-model
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

| Scorer | Returns | Description |
|--------|---------|-------------|
| `exact_match` | `{"correct": bool}` | Case-insensitive, whitespace-stripped equality |
| `contains` | `{"correct": bool}` | Case-insensitive substring match |
| `f1_token` | `{"f1": float, "precision": float, "recall": float}` | Token-level F1 |
| `regex_match` | `{"correct": bool}` | Regex pattern match (target is the pattern) |
| `bleu` | `{"bleu_1"..4: float}` | Sentence-level BLEU-1 through BLEU-4 |
| `rouge` | `{"rouge_1": float, "rouge_2": float, "rouge_l": float}` | ROUGE-1, ROUGE-2, ROUGE-L F1 |
| `retrieval_metrics` | `{"precision_at_k": float, ...}` | Retrieval quality metrics (P@k, R@k, MRR, NDCG) |

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

| Agent | Instruction file | How to start |
|-------|-----------------|--------------|
| **Claude Code** | `.claude/commands/byob.md` | Type `/byob` in any session within the repo |
| **Codex** | `examples/byob/AGENTS.md` | Ask "create a BYOB benchmark" in the repo |
| **Other agents** | `examples/byob/AGENTS.md` | Point the agent at the file or at a template |

The agent walks you through 5 steps: understand the task, read data, generate prompt, pick scorer, compile and ship. See [SKILL_USAGE.md](./SKILL_USAGE.md) for the full walkthrough.

## Examples

- [MedMCQA](./medmcqa/) - Medical multiple-choice QA with HuggingFace dataset and field mapping
- [Global MMLU Lite](./global_mmlu_lite/) - Multilingual MMLU with per-category scoring breakdown
- [TruthfulQA](./truthfulqa/) - LLM-as-Judge evaluation with custom template and `**template_kwargs`
- Templates: [Math Reasoning](./templates/math_reasoning.py) (numeric extraction + tolerance), [Code Generation](./templates/code_generation.py) (syntax check + execution)

## Dataset Format

BYOB accepts JSONL (JSON Lines) files or HuggingFace dataset URIs (`hf://org/dataset`).

Each JSONL line is a JSON object:

```jsonl
{"question": "Is the sky blue?", "answer": "yes"}
{"question": "Is water dry?", "answer": "no"}
```

Your prompt template uses `{field}` placeholders that match your dataset fields.

Use `field_mapping` to rename dataset columns:

```python
@benchmark(
    name="my-bench",
    dataset="data.jsonl",
    prompt="{question}\nA) {a}\nB) {b}",
    field_mapping={"option_a": "a", "option_b": "b"},
)
```

## CI/CD Integration

See [ci-cd-example.yml](./ci-cd-example.yml) for a complete GitHub Actions workflow that compiles and runs BYOB benchmarks in CI. The example shows how to:
- Accept model endpoints as workflow inputs
- Set up the Python environment and install dependencies
- Compile benchmarks (auto-installed, no PYTHONPATH setup needed)
- Run evaluations and upload results as artifacts

## Troubleshooting

### "command not found: nemo-evaluator-byob"
Install the package: `cd packages/nemo-evaluator && pip install -e .`

### "Benchmark not found"
Compilation auto-installs via `pip install -e`. If you used `--no-install`, add the package to PYTHONPATH:
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
