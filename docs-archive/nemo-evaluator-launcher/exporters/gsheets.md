# Google Sheets Exporter (`gsheets`)

Export metrics to Google Sheets for collaborative sharing and analysis. Inherits all core features from the [Local](local.md) exporter (artifact staging, multi-run, auto-export), and adds Google Sheets integration.

**What you can do:**
- Append metrics from multiple jobs/invocations to a shared spreadsheet
- Automatically create/extend header columns based on available metrics
- Share results with team members through Google Sheets
- Auto-export after evaluations finish (local or cluster)

**Key configuration:**
- `spreadsheet_name`: target spreadsheet name (default: "NeMo Evaluator Launcher Results"). Used to open existing sheets or name new ones
- `spreadsheet_id`: spreadsheet ID (optional but **required if your service account can't create sheets due to quota limits**). Find it in the spreadsheet URL: `https://docs.google.com/spreadsheets/d/<spreadsheet_id>/edit`
- `service_account_file`: path to Google service account JSON file (optional; see credentials setup below)
- `log_metrics`: filter specific metrics (default: all available metrics for single-job export)

**Default behavior:**
- Creates spreadsheet if it doesn't exist and shares it publicly (read-only)
- Uses first sheet (`sheet1`) for all data
- Multiple exports to the same spreadsheet append new rows
- Dynamic headers: base columns (Model Name, Task Name, Invocation ID, Job ID, Executor) + metric columns
- Task Name includes full harness.benchmark (e.g., `simple_evals.mmlu`)
- Metric columns strip task prefix (e.g., `simple_evals.mmlu_accuracy` becomes `accuracy`)
- Appends one row per job with metrics

**Default spreadsheet layout:**
| Model Name | Task Name        | Invocation ID | Job ID     | Executor | accuracy | pass@1 | ... |
|------------|------------------|---------------|------------|----------|----------|--------|-----|
| Nemotron   | Simple_evals.mmlu| 8abcd123      | 8abcd123.0 | local    | 0.85     | 0.72   | ... |

**Credentials setup (one-time):**
1. In Google Cloud Console, create (or select) a project and enable both the "Google Sheets API" and the "Google Drive API"
2. Create a Service Account; generate a JSON key and download it
3. **Important for quota-limited accounts**: If your service account has limited quota or can't create spreadsheets, manually create a spreadsheet in your Google Drive, then share it with the service account email (found in the JSON key file) in edit mode. Pass the spreadsheet ID via `spreadsheet_id` in your config
4. Option A (explicit path): save the file securely and point the exporter to it via `service_account_file`
5. Option B (implicit path): place the JSON at `~/.config/gspread/service_account.json` and omit `service_account_file`


**Example (YAML):**
```yaml
execution:
  auto_export:
    destinations: ["gsheets"]
  
  # Export-related env vars (optional for GSheets)
  env_vars:
    export:
      PATH: "/path/to/conda/env/bin:$PATH"

export:
  gsheets:
    spreadsheet_name: "LLM Evaluation Results"
    spreadsheet_id: "1ABC...XYZ"  # optional: use existing sheet
    service_account_file: "/path/to/service-account.json"
    log_metrics: ["accuracy", "pass@1"]
```

**Examples (CLI):**
```bash
# Basic export
nemo-evaluator-launcher export 8abcd123 --dest gsheets

# With overrides
nemo-evaluator-launcher export 8abcd123 --dest gsheets \
  -o export.gsheets.spreadsheet_name="My Results" \
  -o export.gsheets.spreadsheet_id=1ABC...XYZ
```

**Examples (API):**
```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic export (uses default spreadsheet name)
export_results("8abcd123", dest="gsheets")

# Custom spreadsheet + metric filter
export_results("8abcd123.0", dest="gsheets", config={
    "spreadsheet_name": "My LLM Results",
    "spreadsheet_id": "1ABC...XYZ",
    "log_metrics": ["accuracy", "f1_score"]
})
```

**Tip:** We recommend `auto-export` for team collaboration and hands-free result sharing.