# Progress Tracking

## Overview
`ProgressTrackingInterceptor` tracks evaluation progress by counting processed samples and optionally sending updates to a webhook endpoint.

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_progress_tracking=True,target.api_endpoint.adapter_config.progress_tracking_url=http://monitoring:3828/progress'
```

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "progress_tracking"
          enabled: true
          config:
            progress_tracking_url: "http://monitoring:3828/progress"
            progress_tracking_interval: 10
            request_method: "PATCH"
            output_dir: "/tmp/output"
        - name: "endpoint"
          enabled: true
          config: {}
```


## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progress_tracking_url` | `str \| None` | `"http://localhost:8000"` | URL to post progress updates. Supports expansion of shell variables. |
| `progress_tracking_interval` | `int` | `1` | Update every N samples |
| `request_method` | `str` | `"PATCH"` | HTTP method for progress updates |
| `output_dir` | `str \| None` | `None` | Directory to save progress file |

## Behavior

The interceptor tracks the number of responses processed and:

1. **Sends webhook updates**: Posts progress updates to the configured URL at the specified interval
2. **Saves progress to disk**: If `output_dir` is configured, writes progress count to a `progress` file in that directory
3. **Resumes from checkpoint**: If a progress file exists on initialization, resumes counting from that value
