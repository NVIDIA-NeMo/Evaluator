# Google Sheets Exporter (`gsheets`)

Exports accuracy metrics to a Google Sheet. Dynamically creates/extends header columns based on observed metrics and appends one row per job.

- **Purpose**: Centralized spreadsheet for tracking results across runs
- **Requirements**: `gspread` installed and a Google service account with access

See common behavior and usage in [overview](overview.md).

## Key configuration
- `service_account_file` (str, optional): Path to service account JSON; default creds if omitted
- `spreadsheet_name` (str): Target spreadsheet name.
- `log_metrics` (list[str], optional): Filter metrics to log

## Examples

CLI:
```bash
nv-eval export 8abcd123 --dest gsheets --output-dir .
```

Python:
```python
from nemo_evaluator_launcher.api.functional import export_results
export_results(["8abcd123"], dest="gsheets", config={"spreadsheet_name": "Nemo Evaluator Launcher Results"})
```
