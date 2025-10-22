# Post-Evaluation Hooks

Run processing or reporting tasks after evaluations complete.

Post-evaluation hooks execute after the main evaluation finishes. The built-in `PostEvalReportHook` hook generates HTML and JSON reports from cached request-response pairs.

## Report Generation

Generate HTML and JSON reports with evaluation request-response examples.

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      post_eval_hooks:
      - name: "post_eval_report"
        enabled: true
        config:
          report_types: ["html", "json"]
          html_report_size: 10
```

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.generate_html_report=True'
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `report_types` | Types of reports to generate (`html`, `json`) | `["html"]` |
| `html_report_size` | Max number of request-response pairs to include in reports | `None` (includes all) |

## Report Output

The hook generates reports in the evaluation output directory:

- **HTML Report**: `{output_dir}/report.html` - Interactive report with request-response pairs and curl commands
- **JSON Report**: `{output_dir}/report.json` - Machine-readable report with structured data
