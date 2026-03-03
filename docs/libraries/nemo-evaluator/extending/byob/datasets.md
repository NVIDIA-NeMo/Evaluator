(byob-datasets)=

# Datasets

BYOB benchmarks load evaluation data from local JSONL files, CSV/TSV files, or HuggingFace datasets. This page covers supported formats, remote dataset URIs, field mapping, and path resolution.

## JSONL Format

The default and recommended format is JSONL (JSON Lines). Each line is a self-contained JSON object representing one evaluation sample.

```jsonl
{"question": "Is the sky blue?", "answer": "yes"}
{"question": "Is water dry?", "answer": "no"}
```

Every object must be a JSON dictionary. Arrays or scalar values on a line will raise a `ValueError`.

:::{tip}
Files with `.json` extensions are also treated as JSONL (line-delimited JSON), not as a single JSON array.
:::

## CSV and TSV Files

BYOB also supports CSV and TSV files. The first row is treated as column headers using Python's `csv.DictReader`.

```python
@benchmark(
    name="csv-bench",
    dataset="data.csv",
    prompt="Q: {question}\nA:",
    target_field="answer",
)
```

Format detection is based on file extension: `.csv` for comma-separated, `.tsv` for tab-separated.

## HuggingFace Datasets

Use `hf://` URIs to load datasets directly from the HuggingFace Hub. The dataset is downloaded at compile time and cached locally as JSONL.

### URI Format

```
hf://org/dataset?split=validation
hf://org/dataset/config?split=test
```

### Examples

```python
@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt="Q: {question}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\nAnswer:",
    target_field="cop",
    field_mapping={"opa": "a", "opb": "b", "opc": "c", "opd": "d"},
)
```

```python
@benchmark(
    name="boolq",
    dataset="hf://google/boolq?split=validation",
    prompt="Passage: {passage}\nQuestion: {question}\nAnswer (true/false):",
    target_field="answer",
)
```

:::{note}
HuggingFace dataset fetching requires the `datasets` pip package. Install it with `pip install datasets`.
:::

### Caching

Downloaded datasets are cached at `~/.cache/nemo_evaluator/hf_datasets/` by default. Repeated fetches of the same URI reuse the cached JSONL file.

### Split Selection

When no split is specified, the HuggingFace `datasets` library defaults are used. If the result is a `DatasetDict` (multiple splits), the first available split is selected automatically.

## Field Mapping

Use `field_mapping` to rename dataset columns so they match the `{placeholder}` names in your prompt template. The mapping is applied after loading the dataset and before prompt rendering.

### Syntax

```python
field_mapping={"source_column": "target_name"}
```

Keys present in a record are renamed. Keys absent from a record are silently ignored. Keys not listed in the mapping are passed through unchanged.

### Example

The MedMCQA dataset uses column names like `opa`, `opb`, `opc`, `opd` for answer options. Map them to shorter names for cleaner prompt templates:

```python
@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt="Q: {question}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\nAnswer:",
    target_field="cop",
    field_mapping={"opa": "a", "opb": "b", "opc": "c", "opd": "d"},
)
```

## Path Resolution

Dataset paths are resolved relative to the benchmark file's directory, following the same pattern the `@benchmark` decorator uses internally.

| Path type | Resolution |
|-----------|------------|
| Absolute (`/data/eval/data.jsonl`) | Used as-is |
| Relative (`data.jsonl`) | Resolved from the directory containing the benchmark `.py` file |
| HuggingFace URI (`hf://...`) | Downloaded and cached locally |

The internal resolution is equivalent to:

```python
import os
resolved = os.path.join(os.path.dirname(__file__), "data.jsonl")
```

:::{warning}
When running `nemo-evaluator-byob` from a different directory than the benchmark file, relative paths still resolve from the benchmark file's location, not the current working directory.
:::

## Prompt Placeholder Matching

The `{field}` placeholders in your `prompt` string must match keys in the dataset after any `field_mapping` has been applied. Mismatched placeholders cause a `KeyError` at evaluation time.

Use `--dry-run` to validate that your prompt, dataset, and field mapping are consistent before compiling:

```bash
nemo-evaluator-byob my_benchmark.py --dry-run
```

:::{tip}
The dry-run output shows the dataset path, sample count, and any declared requirements -- a quick way to catch configuration errors early.
:::

## See Also

- {ref}`byob` -- BYOB overview and quickstart
- {ref}`byob-benchmark-decorator` -- Benchmark decorator parameters
- {ref}`byob-cli` -- CLI reference for compilation and validation
