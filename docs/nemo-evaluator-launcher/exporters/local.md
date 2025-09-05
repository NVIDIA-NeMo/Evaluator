# Local Exporter (`local`)

Exports artifacts and optional summaries to the local filesystem (and from remote executors via staging). It can also produce consolidated JSON or CSV summaries of metrics.

- **Purpose**: Copy artifacts/logs locally; optionally build per-invocation summaries

See common behavior and usage in [overview](overview.md).

## Key configuration
- `output_dir` (str): Destination directory. Default: `./nemo-evaluator-results`
- `copy_logs` (bool): Include logs alongside artifacts. Default: `false`
- `only_required` (bool): Copy only required/optional artifacts. Default: `true`
- `format` (str | null): Summary format, one of `json`, `csv`, or omit for none
- `log_metrics` (list[str]): Filter which metrics to include (exact or substring)
- `output_filename` (str): Override default summary filename

## Examples

CLI:
```bash
nemo-evaluator-launcher export 8abcd123 --dest local --format json -o results/
```

Python:
```python
from nv_eval.api.functional import export_results
export_results(["8abcd123"], dest="local", config={"format": "json", "output_dir": "./results"})
```
