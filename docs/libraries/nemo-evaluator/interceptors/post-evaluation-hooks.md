# Post-Evaluation Hooks

Run additional processing, reporting, or cleanup tasks after evaluations complete.

Post-evaluation hooks execute after the main evaluation finishes and can perform various tasks such as generating reports, sending notifications, or cleaning up temporary files.

## HTML Report Generation

Generate HTML reports with evaluation results and visualizations.

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.generate_html_report=True'
```

### YAML Configuration
```yaml
post_eval_hooks:
  - name: "post_eval_report"
    enabled: true
    config:
      report_types: ["html", "json"]
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `report_types` | Types of reports to generate | `["html"]` |
| `output_dir` | Directory for generated reports | Same as evaluation output |

## Use Cases

- **Report generation**: Create comprehensive HTML or JSON reports
- **Result visualization**: Generate charts and graphs from evaluation metrics  
- **Notification systems**: Send evaluation completion notifications
- **Data export**: Export results to external systems or databases
- **Cleanup tasks**: Remove temporary files or intermediate data
