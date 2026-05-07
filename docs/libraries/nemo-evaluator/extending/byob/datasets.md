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

BYOB exposes two dataset-related keys under `config.params.extra.dataset.*`
that can be overridden at run time without rebuilding the benchmark:

| Key | CLI flag | Description |
|-----|----------|-------------|
| `path` | `--dataset` | Dataset file path or `hf://` URI. Compile-time default from `@benchmark(dataset=...)`. |
| `num_fewshot` | `--num-fewshot` | Few-shot example count (lm-eval-harness parity). Pass `0` to force true 0-shot for a benchmark that declares a non-zero default. |

All other dataset-related options (`field_mapping`, `choices`, `choices_field`,
`fewshot_dataset`, `fewshot_prefix`, `fewshot_split`, etc.) are baked into the
benchmark at compile time from the `@benchmark(...)` decorator and are not
runtime-overridable — change them in your benchmark module and recompile with
`nemo-evaluator-byob compile`.

### Overriding the dataset at run time

To swap `path` or `num_fewshot` for a single run, set the corresponding key
under `config.params.extra.dataset.*` via the launcher's run_config or CLI.
The launcher deep-merges via OmegaConf, so sibling keys (and unrelated keys
under `extra` such as `benchmark_module`, `requirements`, …) are preserved.

```bash
nemo-evaluator-launcher run --config my_config.yaml \
  -o 'evaluation.tasks.<task_name>.nemo_evaluator_config.config.params.extra.dataset.path=hf://other/foo?split=test' \
  -o 'evaluation.tasks.<task_name>.nemo_evaluator_config.config.params.extra.dataset.num_fewshot=0'
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

## Few-shot Examples

BYOB resolves the few-shot example pool with this precedence:

1. **`fewshot_dataset`** — explicit URI/path. Use this when the few-shot
   source needs filters, `data_files`, configs, or any other URI options
   that cannot be expressed by a split name (e.g.
   `hf://my-org/foo?data_files=train.json&filter_field=lang&filter_value=hi`).
2. **`fewshot_split`** — split name reused with the primary `hf://` dataset.
   Used only when `fewshot_dataset` is unset *or* fails to load.
3. **Tail of the primary dataset** — last-resort fallback. Logs a loud
   warning because the few-shot pool overlaps with rows being evaluated,
   risking gold-answer leakage into the prompt.

### Examples

Few-shot from a different split of the same HuggingFace dataset:

```python
@benchmark(
    name="mmlu-mini",
    dataset="hf://my-org/mmlu?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
    num_fewshot=5,
    fewshot_split="dev",
)
```

Few-shot from a completely different dataset URI (filters, data_files, etc.):

```python
@benchmark(
    name="boolq-hi",
    dataset="hf://sarvamai/boolq-indic?split=validation&filter_field=language&filter_value=hi",
    prompt="Passage: {passage}\nQuestion: {question}\nAnswer:",
    target_field="answer",
    num_fewshot=4,
    fewshot_dataset="hf://sarvamai/boolq-indic?split=train&filter_field=language&filter_value=hi",
)
```

Add a static introduction before the few-shot examples:

```python
@benchmark(
    name="indommlu",
    dataset="hf://indolem/IndoMMLU?split=test&trust_remote_code=true",
    prompt="{question}\n\n{options}\n\nAnswer:",
    target_field="answer",
    num_fewshot=5,
    fewshot_split="train",
    fewshot_prefix="The following are multiple-choice questions. Choose the best answer.\n\n",
)
```

The final prompt sent to the model is:

```text
<fewshot_prefix><example_1><fewshot_separator>...<example_N><fewshot_separator><test_prompt>
```

:::{tip}
At run time you can force a true 0-shot evaluation against a benchmark
that declares a non-zero `num_fewshot` by passing `--num-fewshot 0` on
the `nemo-evaluator run_eval` CLI. The flag is `None` by default; an
explicit `0` overrides the benchmark default.
:::

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
