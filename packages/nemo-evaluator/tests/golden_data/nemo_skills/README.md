# Golden Data for NeMo-Skills Integration Tests

This directory contains golden data files used for determinism and reproducibility testing of the nemo-skills integration.

## Files

| File | Purpose | Used By |
|------|---------|---------|
| `gsm8k_math_input.jsonl` | 10-sample GSM8K fixture data for math scoring tests | T-069, T-070, T-071, T-072 |
| `multichoice_input.jsonl` | 5-sample multichoice fixture data | Integration tests |
| `gsm8k_math_golden.json` | Golden `ns_results.json` for 10-sample GSM8K math scoring | T-070 (to be generated) |
| `multichoice_golden.json` | Golden `ns_results.json` for multichoice scoring | Integration tests (to be generated) |

## Regeneration Procedure

To regenerate all golden data from scratch:

```bash
# 1. Navigate to the project root
cd packages/nemo-evaluator

# 2. Run the golden data generation script (when pytest-regenerate-golden is available)
python -m pytest tests/integration/nemo_skills/test_ns_determinism.py \
    --regenerate-golden \
    -k "test_generate_golden_data" \
    -v

# 3. Alternatively, regenerate manually:
#    a. Set up the deterministic mock client (see test_ns_determinism.py conftest)
#    b. Run the pipeline with gsm8k_math_input.jsonl
#    c. Copy ns_results.json to tests/golden_data/nemo_skills/gsm8k_math_golden.json
#    d. Run the pipeline with multichoice_input.jsonl
#    e. Copy ns_results.json to tests/golden_data/nemo_skills/multichoice_golden.json

# 4. Verify the regenerated golden data passes:
python -m pytest tests/integration/nemo_skills/test_ns_determinism.py -v
```

## Mock Model Response Contract

The deterministic mock for golden file tests uses the following rule:

- **Math evaluations**: Returns `"The answer is \\boxed{<expected_answer>}"` where `<expected_answer>` is extracted from the original sample (perfect model).
- **Multichoice evaluations**: Returns `"The answer is \\boxed{<expected_answer>}"`.

This ensures 100% accuracy in golden files, making metric values fully predictable.

## Tolerances

| Comparison | Tolerance |
|-----------|-----------|
| `ns_results.json` byte comparison (T-069, T-070, T-072) | **Exact match** (0 byte tolerance) |
| Aggregate metric comparison (T-071 order independence) | **Exact float equality** |
| Score.value | **Exact float equality** |
| Score.stats fields | **Exact float equality** with deterministic inputs |
| Score.stats.count | **Exact integer equality** |

## File Format Requirements

- **Encoding**: UTF-8
- **Line endings**: LF (Unix-style)
- **Indentation** (for JSON): 2 spaces
- **Unicode**: Preserved as literal characters (not escaped)
- **ensure_ascii**: False

All golden files MUST be committed with these properties to ensure deterministic byte-for-byte comparison across platforms.
