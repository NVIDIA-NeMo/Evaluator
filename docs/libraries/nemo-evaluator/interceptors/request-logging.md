(interceptor-request-logging)=
# Request Logging Interceptor

## Overview

The `RequestLoggingInterceptor` captures and logs incoming API requests for debugging, analysis, and audit purposes. This interceptor is essential for troubleshooting evaluation issues and understanding request patterns.

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.max_saved_requests=1000'
```

### YAML Configuration


```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "request_logging"
            enabled: true
            config:
              max_requests: 1000
        - name: "endpoint"
          enabled: true
          config: {}
```

## Configuration Options

For detailed configuration options, please refer to the {ref}`interceptor_reference` Python API reference.
