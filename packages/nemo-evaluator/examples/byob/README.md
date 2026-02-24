# BYOB (Bring Your Own Benchmark)

Create custom evaluation benchmarks for NeMo Evaluator in ~12 lines of Python.

## Prerequisites

- Python 3.10+
- NeMo Evaluator installed: `pip install -e .` from `packages/nemo-evaluator`
- An OpenAI-compatible model endpoint (or use the mock server for testing)

## Quickstart (5 minutes)

### Step 1: Write your benchmark

```python
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(
    name="my-qa",
    dataset="data.jsonl",
    prompt="Q: {question}\nA:",
    target_field="answer",
)
@scorer
def check(response: str, target: str, metadata: dict) -> dict:
    return {"correct": target.lower() in response.lower()}
```

### Step 2: Compile

```bash
nemo-evaluator-byob my_benchmark.py
```

### Step 3: Run

```bash
export PYTHONPATH="~/.nemo-evaluator/byob_packages:$PYTHONPATH"
nemo-evaluator run_eval \
  --eval_type byob_my_qa.my-qa \
  --model_url http://localhost:8000 \
  --model_id my-model
```

## CLI Commands

```bash
# Compile a benchmark
nemo-evaluator-byob my_benchmark.py

# Validate without installing
nemo-evaluator-byob my_benchmark.py --dry-run

# List installed benchmarks
nemo-evaluator-byob --list

# Show version
nemo-evaluator-byob --version
```

## Built-in Scorers

Import from `nemo_evaluator.contrib.byob.scorers`:

| Scorer | Returns | Description |
|--------|---------|-------------|
| `contains` | `{"correct": bool}` | Case-insensitive substring match |
| `exact_match` | `{"correct": bool}` | Case-insensitive, whitespace-stripped equality |
| `f1_token` | `{"f1": float, "precision": float, "recall": float}` | Token-level F1 |
| `regex_match` | `{"correct": bool}` | Regex pattern match (target is the pattern) |

### Scorer Composition

```python
from nemo_evaluator.contrib.byob import any_of, all_of
from nemo_evaluator.contrib.byob.scorers import contains, exact_match

# Correct if EITHER scorer matches
lenient = any_of(contains, exact_match)

# Correct only if BOTH scorers match
strict = all_of(contains, exact_match)
```

## Coding Agent Integration

BYOB works with AI coding agents that can read files and run shell commands. Supported agents include Claude Code, Codex, and others.

| Agent | Instruction file | How to start |
|-------|-----------------|--------------|
| **Claude Code** | `.claude/commands/byob.md` | Type `/byob` in any session within the repo |
| **Codex** | `examples/byob/AGENTS.md` | Ask "create a BYOB benchmark" in the repo |
| **Other agents** | `examples/byob/AGENTS.md` | Point the agent at the file or at a template |

The agent walks you through 5 steps: understand the task, read data, generate prompt, pick scorer, compile and ship. See [SKILL_USAGE.md](./SKILL_USAGE.md) for the full walkthrough.

## Examples

- [BoolQ](./boolq/) - Yes/No question answering from SuperGLUE
- [Template Library](./templates/) - 6 ready-to-use templates for math, multichoice, QA, classification, safety, and code generation

## Dataset Format

BYOB uses JSONL (JSON Lines) format. Each line is a JSON object:

```jsonl
{"question": "Is the sky blue?", "answer": "yes"}
{"question": "Is water dry?", "answer": "no"}
```

Your prompt template uses `{field}` placeholders that match your dataset fields.

## CI/CD Integration

See [ci-cd-example.yml](./ci-cd-example.yml) for a complete GitHub Actions workflow that compiles and runs BYOB benchmarks in CI. The example shows how to:
- Accept model endpoints as workflow inputs
- Set up the Python environment and install dependencies
- Compile benchmarks and configure PYTHONPATH
- Run evaluations and upload results as artifacts

## Troubleshooting

### "command not found: nemo-evaluator-byob"
Install the package: `cd packages/nemo-evaluator && pip install -e .`

### "Benchmark not found"
After compiling, add the install directory to PYTHONPATH:
```bash
export PYTHONPATH="~/.nemo-evaluator/byob_packages:$PYTHONPATH"
```

### "Dataset file not found"
Use absolute paths or `os.path.join(os.path.dirname(__file__), "data.jsonl")` for paths relative to your benchmark file.

### "Sample N missing field X"
Your prompt template references a field not in the dataset. Check that `{field}` placeholders match your JSONL keys.

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
