# Google Sheets Exporter (`gsheets`)

## Overview

Export metrics to Google Sheets for collaborative sharing and analysis. This exporter inherits core features from the [Local](local.md) exporter—artifact staging, multi-run support, and auto-export—and adds Google Sheets integration.

## Key Features

Use the Google Sheets exporter to:

- Append metrics from jobs to a shared spreadsheet.
- Create or extend header columns based on available metrics.
- Share results with team members using Google Sheets.
- Automatically export after evaluations finish (local or cluster).

## Configuration

Configure the exporter with these options:

- `spreadsheet_name`: Target spreadsheet name. Default: "NeMo Evaluator Launcher Results".
- `service_account_file`: Path to the Google service account JSON file. Optional; refer to the Credentials Setup section.
- `log_metrics`: Filter specific metrics for single-job export. Default: all available metrics.

## Default Behavior

By default, the exporter:

- Creates the spreadsheet if it does not exist and grants public view access.
- Uses the first sheet (`sheet1`) for all data.
- Appends a new row for each export to the same spreadsheet.
- Builds dynamic headers: base columns (Model Name, Task Name, Invocation ID, Job ID, Executor) plus metric columns.
- Includes the full harness.benchmark in the Task Name, for example, `simple_evals.mmlu`.
- Removes the task prefix from metric columns, for example, `simple_evals.mmlu_accuracy` becomes `accuracy`.
- Appends one row per job with metrics.

## Default Spreadsheet Layout

The following table shows the default spreadsheet layout:

| Model Name   | Task Name        | Invocation ID | Job ID     | Executor | accuracy | pass@1 | ... |
|--------------|------------------|---------------|------------|----------|----------|--------|-----|
| Example Model| simple_evals.mmlu| 8abcd123      | 8abcd123.0 | local    | 0.85     | 0.72   | ... |

## Credentials

Complete the following steps to enable Google Sheets access:

1. In Google Cloud Console, create or select a project and enable the "Google Sheets API".
2. Create a service account, generate a JSON key, and download it.
3. Option A—explicit path: Save the file in a secure location and point the exporter to it:
   - `service_account_file: "/path/to/service-account.json"`
4. Option B—implicit path: Place the JSON file at `~/.config/gspread/service_account.json` and omit `service_account_file`.
5. If you use an existing spreadsheet, share it with the service account email. If the exporter creates the spreadsheet, it grants public view access by default. Adjust sharing in Google Sheets if you require stricter access.

## Examples

### YAML

The following YAML enables automatic export to Google Sheets:

```yaml
execution:
  auto_export:
    destinations: ["gsheets"]
    configs:
      gsheets:
        spreadsheet_name: "LLM Evaluation Results"
        service_account_file: "/path/to/service-account.json"
        # For single-job export filtering:
        log_metrics: ["accuracy", "pass@1"]
```

### CLI

Basic export:

```bash
# Basic export
nemo-evaluator-launcher export 8abcd123 --dest gsheets
```

### Python API

Basic export using the default spreadsheet name:

```python
from nemo_evaluator_launcher.api.functional import export_results

export_results("8abcd123", dest="gsheets")

# Custom spreadsheet and metric filter for single-job export
export_results("8abcd123.0", dest="gsheets", config={
    "spreadsheet_name": "My LLM Results",
    "log_metrics": ["accuracy", "f1_score"]
})
```

Note: Use auto-export to share results with your team with minimal setup.
