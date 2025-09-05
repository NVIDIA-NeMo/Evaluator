# Request & Response Logging

Log incoming requests and outgoing responses for debugging, analysis, and audit trails.

## Request Logging Interceptor

Logs incoming requests for debugging and analysis.

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.max_saved_requests=1000'
```

### YAML Configuration
```yaml
interceptors:
  - name: "request_logging"
    enabled: true
    config:
      max_requests: 1000
      log_failed_requests: true
      output_dir: "./logs"
```

## Response Logging Interceptor

Logs outgoing responses for analysis.

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_response_logging=True,target.api_endpoint.adapter_config.max_saved_responses=1000'
```

### YAML Configuration
```yaml
interceptors:
  - name: "response_logging"
    enabled: true
    config:
      max_responses: 1000
      output_dir: "./logs"
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_requests` | Maximum number of requests to log | 1000 |
| `max_responses` | Maximum number of responses to log | 1000 |
| `log_failed_requests` | Log requests that result in errors | true |
| `output_dir` | Directory to save log files | "./logs" |
