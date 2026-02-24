# BYOB Template Library

Templates are complete, working benchmark files that Claude Code reads and adapts when users invoke the `/byob` skill. Each template can also be compiled and run standalone.

For the full BYOB documentation (CLI commands, built-in scorers, execution modes, troubleshooting), see the [main BYOB README](../README.md).

## Available Templates

| Template | Suitable For | Scorer Type |
|---|---|---|
| `math_reasoning.py` | AIME, GSM8K, MATH, competition math | Number extraction (last number in response) |
| `multichoice.py` | MMLU, ARC, HellaSwag, MedQA | Letter extraction (A/B/C/D) |
| `open_qa.py` | TriviaQA, NaturalQuestions, RAG | Contains (built-in scorer) |
| `classification.py` | Sentiment, topic, intent detection | First-line label extraction |
| `safety.py` | Refusal testing, toxicity evaluation | Refusal phrase detection |
| `code_generation.py` | HumanEval, MBPP, coding problems | Code execution with test assertions |

## Template Structure

Every template follows this pattern:

```
<name>.py          - Benchmark file with @benchmark + @scorer
<name>_data.jsonl  - Sample dataset with exactly 3 rows
```

### Required elements in each `.py` file:

1. **SPDX license header** (Apache 2.0)
2. **Docstring** with: "Suitable for:", "Scoring:", "Dataset fields:", "Target field:"
3. **Import**: `from nemo_evaluator.contrib.byob import benchmark, scorer`
4. **`_SCRIPT_DIR`** pattern for dataset path resolution
5. **One `@benchmark` + `@scorer` pair** (exactly one per file)
6. **Scorer returns dict** with at least key `"correct"` (bool)

### Required elements in each `_data.jsonl` file:

- Exactly 3 rows (one JSON object per line)
- All fields referenced by the template's `{field}` prompt placeholders
- The `target_field` specified in `@benchmark`

## Quick Start: Compile a Template

```bash
cd packages/nemo-evaluator
nemo-evaluator-byob examples/byob/templates/math_reasoning.py
```

## Contributing a New Template

1. Copy an existing template as starting point
2. Follow the structure above
3. Create a `_data.jsonl` with exactly 3 sample rows
4. Test compilation: `nemo-evaluator-byob your_template.py`
5. Verify scorer: `python3 -c "from your_template import ...; print(scorer('test', 'test', {}))"`
