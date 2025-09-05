# Configuring Interceptors

## Core Interceptors

### 1. Request Logging Interceptor
Logs incoming requests for debugging and analysis.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.max_saved_requests=1000'
```

**YAML Configuration:**
```yaml
interceptors:
  - name: "request_logging"
    enabled: true
    config:
      max_requests: 1000
      log_failed_requests: true
      output_dir: "./logs"
```

### 2. Response Logging Interceptor
Logs outgoing responses for analysis.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.use_response_logging=True,target.api_endpoint.adapter_config.max_saved_responses=1000'
```

**YAML Configuration:**
```yaml
interceptors:
  - name: "response_logging"
    enabled: true
    config:
      max_responses: 1000
      output_dir: "./logs"
```

### 3. Caching Interceptor
Caches requests and responses to improve performance and reduce API calls.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache,target.api_endpoint.adapter_config.reuse_cached_responses=True'
```

**YAML Configuration:**
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

### 4. Endpoint Interceptor
The final interceptor that sends requests to the actual API endpoint.

**YAML Configuration:**
```yaml
interceptors:
  - name: "endpoint"
    enabled: true
    config: {}
```

## Specialized Interceptors

### 5. System Message Interceptor
Modifies the system message in requests.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.use_system_prompt=True,target.api_endpoint.adapter_config.custom_system_prompt="You are a helpful assistant."'
```

**YAML Configuration:**
```yaml
interceptors:
  - name: "system_message"
    enabled: true
    config:
      system_message: "You are a helpful assistant."
```

### 6. Payload Modifier Interceptor
Modifies request parameters.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.params_to_add={"temperature": 0.7},target.api_endpoint.adapter_config.params_to_remove=["max_tokens"]'
```

**YAML Configuration:**
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

### 7. Reasoning Interceptor
Handles reasoning tokens in responses and tracks reasoning metrics.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.use_reasoning=True,target.api_endpoint.adapter_config.end_reasoning_token="</think>",target.api_endpoint.adapter_config.start_reasoning_token="<think>"'
```

**YAML Configuration:**
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

### 8. Progress Tracking Interceptor
Tracks evaluation progress.

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.use_progress_tracking=True,target.api_endpoint.adapter_config.progress_tracking_url=http://localhost:3828/progress'
```

**YAML Configuration:**
```yaml
interceptors:
  - name: "progress_tracking"
    enabled: true
    config:
      progress_tracking_url: "http://localhost:3828/progress"
      progress_tracking_interval: 10
      output_dir: "/tmp/output"
```

## Post-Evaluation Hooks

Post-evaluation hooks run after the evaluation is complete and can perform additional processing, reporting, or cleanup tasks.

### HTML Report Generation

**CLI Configuration:**
```bash
--overrides 'target.api_endpoint.adapter_config.generate_html_report=True'
```

**YAML Configuration:**
```yaml
post_eval_hooks:
  - name: "post_eval_report"
    enabled: true
    config:
      report_types: ["html", "json"]
```
