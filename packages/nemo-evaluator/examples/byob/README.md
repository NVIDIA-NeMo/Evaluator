# BYOB (Bring Your Own Benchmark)

Create custom evaluation benchmarks for NeMo Evaluator in ~12 lines of Python.

## Prerequisites

- Python 3.10+
- NeMo Evaluator installed: `pip install -e .` from `packages/nemo-evaluator`
- An OpenAI-compatible model endpoint (or use the mock server for testing)

## Quickstart (5 minutes)

### Step 1: Write your benchmark

```python
from nemo_evaluator.byob import benchmark, scorer

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

# Compile in native mode (faster, better debugging)
nemo-evaluator-byob my_benchmark.py --native

# Validate without installing
nemo-evaluator-byob my_benchmark.py --dry-run

# List installed benchmarks
nemo-evaluator-byob --list

# Show version
nemo-evaluator-byob --version
```

## Built-in Scorers

Import from `nemo_evaluator.byob.scorers`:

| Scorer | Returns | Description |
|--------|---------|-------------|
| `contains` | `{"correct": bool}` | Case-insensitive substring match |
| `exact_match` | `{"correct": bool}` | Case-insensitive, whitespace-stripped equality |
| `f1_token` | `{"f1": float, "precision": float, "recall": float}` | Token-level F1 |
| `regex_match` | `{"correct": bool}` | Regex pattern match (target is the pattern) |

### Scorer Composition

```python
from nemo_evaluator.byob import any_of, all_of
from nemo_evaluator.byob.scorers import contains, exact_match

# Correct if EITHER scorer matches
lenient = any_of(contains, exact_match)

# Correct only if BOTH scorers match
strict = all_of(contains, exact_match)
```

## Execution Modes

### Subprocess Mode (Default)

Runs the benchmark in a separate Python process. Safe, isolated, production-ready.

```bash
nemo-evaluator-byob my_benchmark.py
```

### Native Mode

Runs the benchmark in-process. Faster startup, better error messages, supports Python debugger breakpoints.

```bash
nemo-evaluator-byob my_benchmark.py --native
```

**Use native mode for:** Development, debugging, unit testing with mocked models.
**Use subprocess mode for:** Production evaluations, untrusted code, long-running evals.

> **Security Notice:** Native mode executes benchmark code directly in the evaluator process with full system access. Use subprocess mode (the default) when running untrusted or community-contributed benchmarks. Native mode provides no sandboxing or isolation.

### When to Use Each Mode

| Scenario | Recommended Mode | Why |
|----------|-----------------|-----|
| Production CI/CD pipeline | Subprocess | Process isolation, crash safety |
| Debugging a new scorer | Native | Breakpoints work, faster iteration |
| Running untrusted benchmarks | Subprocess | Security isolation |
| Unit testing with mocked models | Native | No subprocess overhead |
| Long-running evaluations (>1000 samples) | Subprocess | Memory isolation |

## Claude Code Skill

If you use Claude Code, the `/byob` skill provides an interactive wizard that guides you through creating a benchmark. It reads your data, generates the benchmark file, smoke tests the scorer, and compiles -- all in under 5 minutes. The skill is defined in `.claude/commands/byob.md`.

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
python -m nemo_evaluator.byob.runner ... --fail-on-skip
```
