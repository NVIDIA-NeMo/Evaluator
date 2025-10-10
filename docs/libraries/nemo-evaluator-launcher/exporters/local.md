(exporter-local)=

# Local Exporter (`local`)

Exports artifacts and optional summaries to the local filesystem. When used with remote executors, stages artifacts from remote locations. Can produce consolidated JSON or CSV summaries of metrics.

## Usage

Export evaluation results and artifacts to your local filesystem with optional summary reports.

::::{tab-set}

:::{tab-item} CLI

Export artifacts and generate summary reports locally:

```bash
# Basic export to current directory
nv-eval export 8abcd123 --dest local

# Export with JSON summary to custom directory
nv-eval export 8abcd123 --dest local --format json --output-dir ./evaluation-results/

# Export multiple runs with CSV summary and logs included
nv-eval export 8abcd123 9def4567 --dest local --format csv --copy-logs --output-dir ./results

# Export only specific metrics to a custom filename
nv-eval export 8abcd123 --dest local --format json --log-metrics accuracy --log-metrics bleu --output-filename model_metrics.json
```

:::

:::{tab-item} Python

Export results programmatically with flexible configuration:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic local export with JSON summary
export_results(
    invocation_ids=["8abcd123"], 
    dest="local", 
    config={
        "format": "json", 
        "output_dir": "./results"
    }
)

# Export multiple runs with comprehensive configuration
export_results(
    invocation_ids=["8abcd123", "9def4567"], 
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
    invocation_ids=["8abcd123"], 
    dest="local", 
    config={
        "output_dir": "./artifacts-only",
        "format": None,  # No summary file
        "copy_logs": True
    }
)
```

:::

:::{tab-item} YAML Config

Configure local export in your evaluation YAML file for automatic export on completion:

```yaml
execution:
  auto_export:
    destinations: ["local"]

export:
  local:
    format: "json"
    output_dir: "./results"
```

::::

## Key Configuration

```{list-table}
:header-rows: 1
:widths: 25 15 45 15

* - Parameter
  - Type
  - Description
  - Default
* - `output_dir`
  - str
  - Destination directory for exported results
  - `.` (CLI), `./nemo-evaluator-launcher-results` (Python API)
* - `copy_logs`
  - bool
  - Include logs alongside artifacts
  - `false`
* - `only_required`
  - bool
  - Copy only required and optional artifacts; excludes other files
  - `true`
* - `format`
  - str | null
  - Summary format: `json`, `csv`, or `null` for artifacts only
  - `null`
* - `log_metrics`
  - list[str]
  - Filter metrics by name (exact or substring match)
  - All metrics
* - `output_filename`
  - str
  - Override default summary filename (`processed_results.json` or `processed_results.csv`)
  - `processed_results.<format>`
```
