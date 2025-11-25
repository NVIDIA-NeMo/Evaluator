(exporter-gsheets)=

# Google Sheets Exporter (`gsheets`)

Exports accuracy metrics to a Google Sheet. Dynamically creates/extends header columns based on observed metrics and appends one row per job.

- **Purpose**: Centralized spreadsheet for tracking results across runs
- **Requirements**: `gspread` installed and a Google service account with access

## Usage

Export evaluation results to a Google Sheets spreadsheet for easy sharing and analysis.

::::{tab-set}

:::{tab-item} CLI

Export results from a specific evaluation run to Google Sheets:

```bash
# Export results using default spreadsheet name
nemo-evaluator-launcher export 8abcd123 --dest gsheets

# Export with custom spreadsheet name and ID
nemo-evaluator-launcher export 8abcd123 --dest gsheets \
  -o export.gsheets.spreadsheet_name="My Results" \
  -o export.gsheets.spreadsheet_id=1ABC...XYZ
```

:::

:::{tab-item} Python

Export results programmatically with custom configuration:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic export to Google Sheets
export_results(
    invocation_ids=["8abcd123"], 
    dest="gsheets", 
    config={
        "spreadsheet_name": "NeMo Evaluator Launcher Results"
    }
)

# Export with service account and filtered metrics
export_results(
    invocation_ids=["8abcd123", "9def4567"], 
    dest="gsheets", 
    config={
        "spreadsheet_name": "Model Comparison Results",
        "service_account_file": "/path/to/service-account.json",
        "log_metrics": ["accuracy", "f1_score"]
    }
)
```

:::

:::{tab-item} YAML Config

Configure Google Sheets export in your evaluation YAML file for automatic export on completion:

```yaml
execution:
  auto_export:
    destinations: ["gsheets"]

export:
  gsheets:
    spreadsheet_name: "LLM Evaluation Results"
    spreadsheet_id: "1ABC...XYZ"  # optional: use existing sheet
    service_account_file: "/path/to/service-account.json"
    log_metrics: ["accuracy", "pass@1"]
```

::::

## Key Configuration

```{list-table}
:header-rows: 1
:widths: 25 25 25 25

* - Parameter
  - Type
  - Description
  - Default/Notes
* - `service_account_file`
  - str, optional
  - Path to service account JSON
  - Uses default credentials if omitted
* - `spreadsheet_name`
  - str, optional
  - Target spreadsheet name. Used to open existing sheets or name new ones.
  - Default: "NeMo Evaluator Launcher Results"
* - `spreadsheet_id`
  - str, optional
  - Target spreadsheet ID. Find it in the spreadsheet URL: `https://docs.google.com/spreadsheets/d/<spreadsheet_id>/edit`
  - Required if your service account can't create sheets due to quota limits.
* - `log_metrics`
  - list[str], optional
  - Filter metrics to log
  - All metrics if omitted
```
