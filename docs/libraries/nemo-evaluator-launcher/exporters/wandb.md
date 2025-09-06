# Weights & Biases Exporter (`wandb`)

Exports accuracy metrics and artifacts to W&B. Supports either per-task runs or a single multi-task run per invocation, with artifact logging and run metadata.

- **Purpose**: Track runs, metrics, and artifacts in W&B
- **Requirements**: `wandb` installed and credentials configured

See common behavior and usage in [overview](overview.md).

## Key configuration
- `entity` (str), `project` (str): W&B entity and project
- `log_mode` (str): `per_task` (default) or `multi_task` (aggregate by invocation)
- `name` (str, optional), `group` (str, optional): Run display name and group
- `tags` (dict[str,str], optional), `description` (str, optional)
- `log_metrics` (list[str], optional): Filter metrics
- `log_artifacts` (bool): Log artifacts in addition to metrics. Default: `true`
- `extra_metadata` (dict, optional): Additional key/value metadata
- Webhook (optional): `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`

## Examples

CLI:
```bash
nv-eval export 8abcd123 --dest wandb -o .
```

Python:
```python
from nemo_evaluator_launcher.api.functional import export_results
export_results(["8abcd123"], dest="wandb", config={"entity": "myorg", "project": "evals"})
```
