(configuring-interceptors)=
# Configuring Interceptors

Interceptors are components that process API requests and responses during evaluation. They provide capabilities for logging, caching, request modification, and response processing. Interceptors execute in a specific order within the evaluation pipeline, allowing you to customize and track the behavior of your evaluations.

## Overview

The NeMo Evaluator supports several types of interceptors:

- **Core Interceptors**: Essential components for logging and caching
- **Specialized Interceptors**: Advanced features for request modification and reasoning
- **Post-Evaluation Hooks**: Processing that occurs after evaluation completion

You can configure interceptors using either CLI overrides or YAML configuration files. The examples below show both approaches for each interceptor type.

## Core Interceptors

### Request Logging Interceptor

Captures and logs incoming API requests for debugging, analysis, and audit purposes. This interceptor is essential for troubleshooting evaluation issues and understanding request patterns.

**Key Features:**

- Logs request payloads and metadata
- Configurable request limits
- Logging of failed requests

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.max_saved_requests=1000'
```

:::

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "request_logging"
    enabled: true
    config:
      max_requests: 1000
      log_failed_requests: true
```

:::
::::

### Response Logging Interceptor

Captures and logs API responses for analysis and debugging. Use this interceptor to examine model outputs and identify response patterns.

**Key Features:**

- Logs response payloads and metadata
- Configurable response limits
- Supports filtering by response status

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_response_logging=True,target.api_endpoint.adapter_config.max_saved_responses=1000'
```

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "response_logging"
    enabled: true
    config:
      max_responses: 1000
```

:::
::::

### Caching Interceptor

Implements intelligent caching of API requests and responses to improve evaluation performance and reduce API costs. The caching interceptor can speed up repeated evaluations.

**Key Features:**

- Persistent disk-based caching
- Configurable cache directory
- Separate limits for cached requests and responses
- Reuse of cached responses

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache,target.api_endpoint.adapter_config.reuse_cached_responses=True'
```

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "caching"
    enabled: true
    config:
      cache_dir: "./cache"
      reuse_cached_responses: true
      save_requests: true
      save_responses: true
      max_saved_requests: 1000
      max_saved_responses: 1000
```

:::
::::

### Endpoint Interceptor

**Required interceptor** that handles the actual API communication. This interceptor must be present in every configuration as it performs the final request to the target API endpoint.

**Important**: This interceptor should always be the last in the interceptor chain.

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
# The endpoint interceptor is automatically enabled and requires no additional CLI configuration
```

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "endpoint"
    enabled: true
    config: {}
```

:::
::::

## Specialized Interceptors

### System Message Interceptor

Modifies or injects system messages into API requests. This interceptor is useful for standardizing prompts across evaluations or adding specific instructions to model requests.

**Key Features:**

- Custom system message injection
- Overrides existing system messages
- Supports dynamic message templates

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_system_prompt=True,target.api_endpoint.adapter_config.custom_system_prompt="You are a helpful assistant."'
```

:::

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "system_message"
    enabled: true
    config:
      system_message: "You are a helpful assistant."
```

:::
::::

### Payload Modifier Interceptor

Provides comprehensive request payload modification capabilities. Use this interceptor to standardize request parameters, add missing fields, or remove unwanted parameters.

**Key Features:**

- Add new parameters to requests
- Remove existing parameters
- Rename parameter keys
- Supports complex parameter transformations

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.params_to_add={"temperature": 0.7},target.api_endpoint.adapter_config.params_to_remove=["max_tokens"]'
```

:::

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "payload_modifier"
    enabled: true
    config:
      params_to_add:
        temperature: 0.7
        max_tokens: 1000
      params_to_remove:
        - "max_tokens"
      params_to_rename:
        "old_param": "new_param"
```

:::
::::

### Reasoning Interceptor

Specialized interceptor for handling reasoning tokens in model responses. This interceptor is useful for evaluating models that support chain-of-thought reasoning or explicit thinking processes.

**Key Features:**

- Configurable reasoning token delimiters
- Reasoning metrics tracking
- Reasoning token injection
- Support for incomplete reasoning sequences

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_reasoning=True,target.api_endpoint.adapter_config.end_reasoning_token="</think>",target.api_endpoint.adapter_config.start_reasoning_token="<think>"'
```

:::

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "reasoning"
    enabled: true
    config:
      end_reasoning_token: "</think>"
      start_reasoning_token: "<think>"
      add_reasoning: true
      enable_reasoning_tracking: true
      include_if_not_finished: true
```

:::
::::

### Progress Tracking Interceptor

Monitors and reports evaluation progress in real-time. This interceptor is valuable for long-running evaluations and integration with monitoring systems.

**Key Features:**

- Real-time progress reporting
- Configurable reporting intervals
- HTTP endpoint integration
- Output directory tracking

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_progress_tracking=True,target.api_endpoint.adapter_config.progress_tracking_url=http://localhost:3828/progress'
```

:::

:::{tab-item} YAML Configuration

```yaml
interceptors:
  - name: "progress_tracking"
    enabled: true
    config:
      progress_tracking_url: "http://localhost:3828/progress"
      progress_tracking_interval: 10
      output_dir: "/tmp/output"
```

:::
::::

## Post-Evaluation Hooks

Post-evaluation hooks execute after the main evaluation process completes. These hooks enable processing, custom reporting, data export, and cleanup operations.

**Common Use Cases:**

- Generate custom reports
- Export results to external systems
- Perform data cleanup
- Send notifications

### HTML Report Generation

Generates comprehensive HTML reports from evaluation results. This hook creates user-friendly, visual reports that you can share with stakeholders or archive for future reference.

**Supported Report Types:**

- HTML: Interactive web-based reports
- JSON: Machine-readable structured data

::::{tab-set}
:::{tab-item} CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.generate_html_report=True'
```

:::

:::{tab-item} YAML Configuration

```yaml
post_eval_hooks:
  - name: "post_eval_report"
    enabled: true
    config:
      report_types: ["html", "json"]
      html_report_size: 15  # Optional: limit number of entries
```

:::
::::
