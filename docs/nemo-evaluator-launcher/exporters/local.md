# Local Exporter (`local`)

Exports artifacts and optional summaries to the local filesystem (and stages from remote jobs when needed). Also supports consolidated JSON/CSV summaries.

- What you can do:
  - Move artifacts and logs of multiple runs into one tidy results folder
  - Build single- or multi-invocation JSON/CSV metric summaries
  - Stage results from remote jobs back to your machine (or between filesystems)
  - Use with auto-export to collect results automatically after runs finish

- Options:
  - `output_dir` (alias `-o`): where results go (default: `./nemo-evaluator-launcher-results`)
  - `format`: `json` or `csv` to produce summaries (omit for artifacts-only)
  - `log_metrics`: select which metrics to include
  - `only_required`: copy minimal, relevant files (default: true)
  - `copy_logs`: include logs alongside artifacts

- Default file/dir layout:
  - Per-job staging:
    - `output_dir/<invocation>/<job>/artifacts/` and `.../logs/`
    - `output_dir/<invocation>/<job>/job_metadata.json`
  - Per-invocation summary (when `format` set):
    - `output_dir/<invocation>/processed_results.<json|csv>` (override with `output_filename`)
  - Multi-invocation consolidated summary (when exporting multiple invocations with `format`):
    - `output_dir/processed_results.<json|csv>`

- Examples (CLI):
```bash
# Single invocation → artifacts + JSON summary
nemo-evaluator-launcher export 8abcd123 --dest local --format json -o ./results

# Single job → artifacts only
nemo-evaluator-launcher export 8abcd123.0 --dest local -o ./results

# Multiples → one consolidated CSV summary
nemo-evaluator-launcher export 8abcd123 9def4567 0abcdabcd.3 --dest local --format csv -o ./results
```

- Examples (API):
```python
from nemo_evaluator_launcher.api.functional import export_results
export_results("8abcd123", dest="local", config={"format": "json", "output_dir": "./results"})
export_results(["8abcd123", "9def4567"], dest="local", config={"format": "csv", "output_dir": "./results"})
```

- Auto-export (YAML):
```yaml
execution:
  auto_export:
    destinations: ["local"]
    configs:
      local:
        format: "json"
        output_dir: "./results"
```