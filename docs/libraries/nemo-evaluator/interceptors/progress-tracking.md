# Track Progress

Track evaluation progress and send status updates to external monitoring systems.

## Configuration

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_progress_tracking=True,target.api_endpoint.adapter_config.progress_tracking_url=http://localhost:3828/progress'
```

### YAML Configuration
```yaml
interceptors:
  - name: "progress_tracking"
    enabled: true
    config:
      progress_tracking_url: "http://localhost:3828/progress"
      progress_tracking_interval: 10
      output_dir: "/tmp/output"
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `progress_tracking_url` | URL to send progress updates | None |
| `progress_tracking_interval` | Interval between updates (seconds) | 10 |
| `output_dir` | Directory for progress output files | "/tmp/output" |

## Use Cases

- **Long-running evaluations**: Monitor progress of extended evaluation runs
- **Dashboard integration**: Send updates to monitoring dashboards
- **Automated workflows**: Trigger downstream processes based on evaluation progress
