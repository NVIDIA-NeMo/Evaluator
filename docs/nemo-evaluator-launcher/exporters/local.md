# Local Exporter (`local`)

## Overview

Export artifacts and optional metric summaries to the local file system. You can also stage results from remote jobs and create consolidated JSON or CSV summaries.

## Key Features

- Collect artifacts and logs from runs into a single results directory.
- Generate JSON or CSV metric summaries for one or more invocations.
- Stage results from remote jobs to a local or different file system.
- Use with auto-export to collect results after runs finish.

## Configuration

- `output_dir` (alias `-o`): Destination directory for results (default: `./nemo-evaluator-launcher-results`).
- `format`: `json` or `csv` to produce summaries. Omit to export artifacts without summaries.
- `log_metrics`: Metrics to include in summaries.
- `only_required`: Copy the minimal set of relevant files (default: true).
- `copy_logs`: Include logs alongside artifacts.

## Default Layout

- Per-job staging:
  - `output_dir/<invocation>/<job>/artifacts/` and `.../logs/`
  - `output_dir/<invocation>/<job>/job_metadata.json`
- Per-invocation summary (when you set `format`):
  - `output_dir/<invocation>/processed_results.<json|csv>` (override with `output_filename`)
- Multi-invocation consolidated summary (when exporting more than one invocation with `format`):
  - `output_dir/processed_results.<json|csv>`

## Credentials

None.

## Examples

### YAML

```yaml
execution:
  auto_export:
    destinations: ["local"]
    configs:
      local:
        format: "json"
        output_dir: "./results"
```

### CLI

```bash
# Single invocation → artifacts + JSON summary
nemo-evaluator-launcher export 8abcd123 --dest local --format json -o ./results

# Single job → artifacts only
nemo-evaluator-launcher export 8abcd123.0 --dest local -o ./results

# Multiple invocations → one consolidated CSV summary
nemo-evaluator-launcher export 8abcd123 9def4567 0abcdabcd.3 --dest local --format csv -o ./results
```

### Python API

```python
from nemo_evaluator_launcher.api.functional import export_results

export_results("8abcd123", dest="local", config={"format": "json", "output_dir": "./results"})
export_results(["8abcd123", "9def4567"], dest="local", config={"format": "csv", "output_dir": "./results"})
```
