(interceptor-response-logging)=
# Response Logging Interceptor

## Overview

The `ResponseLoggingInterceptor` captures and logs API responses for analysis and debugging. Use this interceptor to examine model outputs and identify response patterns.


## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_response_logging=True,target.api_endpoint.adapter_config.max_saved_responses=1000'
```

### YAML Configuration


```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "endpoint"
          enabled: true
          config: {}
        - name: "response_logging"
          enabled: true
          config:
            max_responses: 1000
```

## Configuration Options

| Parameter           | Description                                                  | Default   | Type          |
|---------------------|--------------------------------------------------------------|-----------|---------------|
| `log_response_body`   | Whether to log the response body contents.                 | `True`    | `bool`        |
| `log_response_headers`| Whether to log the response HTTP headers.                  | `True`    | `bool`        |
| `max_responses`       | Maximum number of responses to log (None for unlimited).   | `None`    | `int` (optional)|