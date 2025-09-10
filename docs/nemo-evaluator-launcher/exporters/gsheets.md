### Google Sheets Exporter (`gsheets`)

Export metrics to Google Sheets for collaborative sharing and analysis. Inherits all core features from the [Local](local.md) exporter (artifact staging, multi-run, auto-export), and adds Google Sheets integration.

**What you can do:**
- Append metrics from multiple jobs/invocations to a shared spreadsheet
- Automatically create/extend header columns based on available metrics
- Share results with team members through Google Sheets
- Auto-export after evaluations finish (local or cluster)

**Key configuration:**
- `spreadsheet_name`: target spreadsheet name (default: "NV-Eval Results")
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
1. In Google Cloud Console, create (or select) a project and enable the “Google Sheets API”.
2. Create a Service Account; generate a JSON key and download it.
3. Option A (explicit path): save the file securely and point the exporter to it:
   - `service_account_file: "/path/to/service-account.json"`
4. Option B (implicit path): place the JSON at `~/.config/gspread/service_account.json` and omit `service_account_file`.
5. If using an existing spreadsheet, share it with the service account email. If the exporter creates it, it will be shared publicly as read-only by default; adjust sharing in Google Sheets if you need stricter access.

**Example (YAML):**
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

**Examples (CLI):**
```bash
# Basic export
nemo-evaluator-launcher export 8abcd123 --dest gsheets
```

**Examples (API):**
```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic export (uses default spreadsheet name)
export_results("8abcd123", dest="gsheets")

# Custom spreadsheet + metric filter for single-job export
export_results("8abcd123.0", dest="gsheets", config={
    "spreadsheet_name": "My LLM Results",
    "log_metrics": ["accuracy", "f1_score"]
})
```

**Tip:** We recommend `auto-export` for team collaboration and hands-free result sharing.
