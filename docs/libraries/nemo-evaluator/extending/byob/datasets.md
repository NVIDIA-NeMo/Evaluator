(byob-datasets)=

# Datasets

BYOB benchmarks load evaluation data from local JSONL files, CSV/TSV files, or HuggingFace datasets. This page covers supported formats, remote dataset URIs, field mapping, and path resolution.

## JSONL Format

The default and recommended format is JSONL (JSON Lines). Each line is a self-contained JSON object representing one evaluation sample.

```json
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

BYOB also accepts selected HuggingFace `load_dataset` options as query
parameters:

```
hf://org/dataset?split=test&trust_remote_code=true
hf://org/dataset?split=validation&filter_field=language&filter_value=hi
hf://org/dataset?split=test&data_files=file.json&field=examples
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

Load a gated or custom-code dataset by allowing remote code execution:

```python
@benchmark(
    name="indommlu",
    dataset="hf://indolem/IndoMMLU?split=test&trust_remote_code=true",
    prompt="{question}\n\n{options}\n\nAnswer:",
    target_field="answer",
)
```

Filter a multilingual dataset to one language:

```python
@benchmark(
    name="boolq-hi",
    dataset="hf://sarvamai/boolq-indic?split=validation&filter_field=language&filter_value=hi",
    prompt="Passage: {passage}\nQuestion: {question}\nAnswer:",
    target_field="answer",
)
```

Load a nested JSON field from a specific dataset file:

```python
@benchmark(
    name="flores-hi",
    dataset="hf://google/IndicGenBench_flores_in?split=test&data_files=flores_en_hi_test.json&field=examples",
    prompt="Translate to Hindi: {source}",
    target_field="target",
)
```

:::{note}
HuggingFace dataset fetching requires the `datasets` pip package. Install it with `pip install datasets`.
:::

### Caching

Downloaded datasets are cached at `~/.cache/nemo_evaluator/hf_datasets/` by default. Repeated fetches of the same URI reuse the cached JSONL file.

### Split Selection

When no split is specified, the HuggingFace `datasets` library defaults are used. If the result is a `DatasetDict` (multiple splits), the first available split is selected automatically.

### Query Parameters

| Parameter | Description |
|-----------|-------------|
| `split` | Split passed to `datasets.load_dataset`, for example `test` or `validation`. |
| `trust_remote_code=true` | Passes `trust_remote_code=True` to `datasets.load_dataset`. Required by some datasets with custom loading scripts. |
| `filter_field` / `filter_value` | Filters rows after loading, keeping rows where `str(row[filter_field]) == filter_value`. |
| `filter_field_1` / `filter_value_1`, etc. | Additional row filters applied in order. |
| `data_files` | Passes `data_files` to `datasets.load_dataset`, useful for repositories that store examples in individual JSON files. |
| `field` | Passes `field` to `datasets.load_dataset`, useful for JSON files where examples live under a top-level key such as `examples`. |

::::{warning}
If you put `hf://` URIs with `&` query parameters in shell command
templates, quote the dataset argument:

```bash
--dataset "{{config.params.extra.dataset.path}}"
```

Otherwise the shell treats `&` as a background-command separator.
::::

### `extra.dataset.*` namespace

BYOB groups dataset-related configuration under
`config.params.extra.dataset.*` in the FDF / run_config:

| Key | Description |
|-----|-------------|
| `path` | Dataset file path or `hf://` URI (compile-time default from `@benchmark(dataset=...)`). |
| `num_fewshot` | Optional few-shot example count (lm-eval-harness parity). |
| `field_mapping` | Informational mirror of `@benchmark(field_mapping=...)`. |
| `choices` / `choices_field` | Informational mirror of `@benchmark(choices=...)` / `@benchmark(choices_field=...)`. |

### Overriding the dataset at run time

The `@benchmark` decorator's `dataset=` value is the compile-time default. To
swap it for a single run without rebuilding the benchmark, set
`config.params.extra.dataset.path` via the launcher's run_config or CLI. The
launcher deep-merges via OmegaConf, so sibling keys under `extra.dataset`
(`num_fewshot`, `field_mapping`, etc.) and under `extra` (`benchmark_module`,
`requirements`, …) are preserved.

```bash
nemo-evaluator-launcher run --config my_config.yaml \
  -o 'evaluation.tasks.<task_name>.nemo_evaluator_config.config.params.extra.dataset.path=hf://other/foo?split=test'
```

Or in a run_config YAML:

```yaml
evaluation:
  tasks:
    - name: <task_name>
      nemo_evaluator_config:
        config:
          params:
            extra:
              dataset:
                path: hf://other/foo?split=test
                num_fewshot: 5
```

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
