# Local Exporter (`local`)

Exports artifacts and optional summaries to the local filesystem (and from remote executors via staging). It can also produce consolidated JSON or CSV summaries of metrics.

- **Purpose**: Copy artifacts/logs locally; optionally build per-invocation summaries

## Usage

Export evaluation results and artifacts to your local filesystem with optional summary reports.


::::{tab-set}

:::{tab-item} CLI

Export artifacts and generate summary reports locally:

```bash
# Basic export to default directory
nv-eval export 8abcd123 --dest local

# Export with JSON summary to custom directory
nv-eval export 8abcd123 --dest local --format json -o ./evaluation-results/

# Export multiple runs with CSV summary and logs included
nv-eval export 8abcd123 9def4567 --dest local \
  --config '{"format": "csv", "copy_logs": true, "output_dir": "./results"}'

# Export only specific metrics to a custom filename
nv-eval export 8abcd123 --dest local \
  --config '{"format": "json", "log_metrics": ["accuracy", "bleu"], "output_filename": "model_metrics.json"}'
```

:::

:::{tab-item} Python

Export results programmatically with flexible configuration:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic local export with JSON summary
export_results(
    ["8abcd123"], 
    dest="local", 
    config={
        "format": "json", 
        "output_dir": "./results"
    }
)

# Export multiple runs with comprehensive configuration
export_results(
    ["8abcd123", "9def4567"], 
    dest="local", 
    config={
        "output_dir": "./evaluation-outputs",
        "format": "csv",
        "copy_logs": True,
        "only_required": False,  # Include all artifacts
        "log_metrics": ["accuracy", "f1_score", "perplexity"],
        "output_filename": "comprehensive_results.csv"
    }
)

# Export artifacts only (no summary)
export_results(
    ["8abcd123"], 
    dest="local", 
    config={
        "output_dir": "./artifacts-only",
        "format": None,  # No summary file
        "copy_logs": True
    }
)
```

:::

::::

## Key Configuration

```{list-table}
:header-rows: 1
:widths: 25 15 45 15

* - Parameter
  - Type
  - Description
  - Default/Notes
* - `output_dir`
  - str
  - Destination directory
  - `./nemo-evaluator-results`
* - `copy_logs`
  - bool
  - Include logs alongside artifacts
  - `false`
* - `only_required`
  - bool
  - Copy only required/optional artifacts
  - `true`
* - `format`
  - str | null
  - Summary format: `json`, `csv`, or omit for none
  - None
* - `log_metrics`
  - list[str]
  - Filter which metrics to include (exact or substring)
  - All metrics
* - `output_filename`
  - str
  - Override default summary filename
  - Auto-generated
```

