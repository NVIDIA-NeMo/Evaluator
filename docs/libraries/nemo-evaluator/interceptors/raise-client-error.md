(interceptor-raise-client-error)=
# Raise Client Error Interceptor

## Overview

The Raise `RaiseClientErrorInterceptor` handles non-retryable client errors by raising exceptions instead of continuing the benchmark evaluation. By default, it will raise exceptions on 4xx HTTP status codes (excluding 408 Request Timeout and 429 Too Many Requests, which are typically retryable).

This interceptor is useful when you want to fail fast on client errors that indicate configuration issues, authentication problems, or other non-recoverable errors rather than continuing the evaluation with failed requests.

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_raise_client_errors=True'
```

### YAML Configuration

::::{tab-set}

:::{tab-item} Default Configuration
Raises on 4xx status codes except 408 (Request Timeout) and 429 (Too Many Requests).

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "raise_client_errors"
          enabled: true
          config:
            # Default configuration - raises on 4xx except 408, 429
            exclude_status_codes: [408, 429]
            status_code_range_start: 400
            status_code_range_end: 499
        - name: "endpoint"
          enabled: true
          config: {}
```
:::

:::{tab-item} Specific Status Codes
Raises only on specific status codes rather than a range.

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "raise_client_errors"
          enabled: true
          config:
            # Custom configuration - only specific status codes
            status_codes: [400, 401, 403, 404]
        - name: "endpoint"
          enabled: true
          config: {}
```
:::

:::{tab-item} Custom Exclusions
Uses a status code range with custom exclusions, including 404 Not Found.

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "raise_client_errors"
          enabled: true
          config:
            # Custom range with different exclusions
            status_code_range_start: 400
            status_code_range_end: 499
            exclude_status_codes: [408, 429, 404]  # Also exclude 404 not found
        - name: "endpoint"
          enabled: true
          config: {}
```
:::

::::

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclude_status_codes` | `List[int]` | `[408, 429]` | Status codes to exclude from raising client errors when present in the status code range |
| `status_codes` | `List[int]` | `None` | Specific list of status codes that should raise exceptions. If provided, this takes precedence over range settings |
| `status_code_range_start` | `int` | `400` | Start of the status code range (inclusive) for which to raise exceptions |
| `status_code_range_end` | `int` | `499` | End of the status code range (inclusive) for which to raise exceptions |

## Behavior

### Default Behavior
- Raises exceptions on HTTP status codes 400-499
- Excludes 408 (Request Timeout) and 429 (Too Many Requests) as these are typically retryable
- Logs critical errors before raising the exception

### Configuration Logic
1. If `status_codes` is specified, only those exact status codes will trigger exceptions
2. If `status_codes` is not specified, the range defined by `status_code_range_start` and `status_code_range_end` is used
3. `exclude_status_codes` are always excluded from raising exceptions
4. Cannot have the same status code in both `status_codes` and `exclude_status_codes`

### Error Handling
- Raises `FatalErrorException` when a matching status code is encountered
- Logs critical error messages with status code and URL information
- Stops the evaluation process immediately

## Examples

::::{tab-set}

:::{tab-item} Auth Failures Only
Raises exceptions only on authentication and authorization failures.

```yaml
config:
  status_codes: [401, 403]
```
:::

:::{tab-item} All Client Errors Except Rate Limiting
Raises on all 4xx errors except timeout and rate limit errors.

```yaml
config:
  status_code_range_start: 400
  status_code_range_end: 499
  exclude_status_codes: [408, 429]
```
:::

:::{tab-item} Strict Mode - All Client Errors
Raises exceptions on any 4xx status code without exclusions.

```yaml
config:
  status_code_range_start: 400
  status_code_range_end: 499
  exclude_status_codes: []
```
:::

::::

## Common Use Cases

- **API Configuration Validation**: Fail immediately on authentication errors (401, 403)
- **Input Validation**: Stop evaluation on bad request errors (400)
- **Resource Existence**: Fail on not found errors (404) for critical resources
- **Development/Testing**: Use strict mode to catch all client-side issues
- **Production**: Use default settings to allow retryable errors while catching configuration issues
