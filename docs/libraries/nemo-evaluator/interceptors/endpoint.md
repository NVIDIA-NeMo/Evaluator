(interceptor-endpoint)=
# Endpoint Interceptor

## Overview

**Required interceptor** that handles the actual API communication. This interceptor must be present in every configuration as it performs the final request to the target API endpoint.

**Important**: This interceptor should always be placed after the last request interceptor and before the first response interceptor.


## Configuration

### CLI Configuration

```bash
# The endpoint interceptor is automatically enabled and requires no additional CLI configuration
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
```

## Configuration Options

The Endpoint Interceptor is configured automatically.
