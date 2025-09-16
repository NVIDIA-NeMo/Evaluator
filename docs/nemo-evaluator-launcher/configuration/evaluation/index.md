# Evaluation Configuration

Evaluation configuration defines which benchmarks to run and their configuration. It is common for all executors and can be reused between them to launch the exact same tasks.

**Important**: Each task has its own default values that you can override. For comprehensive override options, see {doc}`nemo-evaluator Parameter Overrides <../../nemo-evaluator/reference/cli:parameter-overrides>`.

## Configuration Structure

```yaml
evaluation:
  overrides:  # Global overrides for all tasks
    config.params.request_timeout: 3600
    target.api_endpoint.adapter_config.use_system_prompt: true
  tasks:
    - name: task_name  # Use default benchmark configuration
    - name: another_task
      overrides:  # Task-specific overrides
        config.params.temperature: 0.6
        config.params.top_p: 0.95
      env_vars:  # Task-specific environment variables
        HF_TOKEN: MY_HF_TOKEN
```

## Key Components

# Global Overrides
- **`overrides`**: Parameter overrides that apply to all tasks
- **`env_vars`**: Environment variables that apply to all tasks

# Task Configuration
- **`tasks`**: List of evaluation tasks to run
- **`name`**: Name of the benchmark task
- **`overrides`**: Task-specific parameter overrides
- **`env_vars`**: Task-specific environment variables

For a comprehensive list of available tasks, their descriptions, and task-specific parameters, see the {doc}`NeMo Evaluator supported tasks <../../nemo-evaluator/reference/containers>`.

## Advanced Task Configuration

# Parameter Overrides
The overrides system is crucial for leveraging the full flexibility of the common endpoint interceptors and task configuration layer. This is where nemo-evaluator intersects with nemo-evaluator-launcher, providing a unified configuration interface.

## Global Overrides
```yaml
evaluation:
  overrides:
    config.params.request_timeout: 3600
    config.params.temperature: 0.7
    target.api_endpoint.adapter_config.use_system_prompt: true
    target.api_endpoint.adapter_config.custom_system_prompt: "Think step by step."
```

## Task-Specific Overrides
```yaml
evaluation:
  tasks:
    - name: gpqa_diamond
      overrides:
        config.params.temperature: 0.6
        config.params.top_p: 0.95
        config.params.max_new_tokens: 8192
        config.params.parallelism: 32
    - name: mbpp
      overrides:
        config.params.temperature: 0.2
        config.params.top_p: 0.95
        config.params.max_new_tokens: 2048
        config.params.extra.n_samples: 5
        target.api_endpoint.adapter_config.custom_system_prompt: "You must only provide the code implementation"
```

# Environment Variables
```yaml
evaluation:
  overrides:
    # Global environment variables
  tasks:
    - name: task_name
      env_vars:
        HF_TOKEN: MY_HF_TOKEN
        CUSTOM_VAR: CUSTOM_VALUE
```

## When to Use

Use evaluation configuration when you want to:

- **Change Default Sampling Parameters**: Adjust temperature, top_p, max_new_tokens for different tasks
- **Set Custom System Prompts**: Add task-specific instructions or reasoning prompts (see {doc}`System Message Interceptor <../../nemo-evaluator/reference/configuring-interceptors:5-system-message-interceptor>`)
- **Add/Remove/Rename Special Parameters**: Modify payload structure (e.g., add `"reasoning": "thinking"`) (see {doc}`Payload Modifier Interceptor <../../nemo-evaluator/reference/configuring-interceptors:6-payload-modifier-interceptor>`)
- **Change Default Task Values**: Override benchmark-specific default configurations
- **Parametrize the Judge**: Configure evaluation judge models and their parameters for scoring (see {doc}`Parameter Overrides <../../nemo-evaluator/reference/cli:parameter-overrides>`)
- **Debug and Test**: e.g. Launch with limited samples
- **Adjust Endpoint Capabilities**: Configure request timeouts, max retries, and parallel request limits

/// tip | Long String Overrides
For overriding long strings (e.g., system prompts), use YAML multiline syntax with `>-`:

```yaml
target.api_endpoint.adapter_config.params_to_add: >-
  <HERE_WRITE_WHATEVER_YOU_WANT_AND_IT_WILL_BE_PASSED_WITHOUT_ANY_MODIFICATIONS_TO_OVERRIDES>
```

This preserves formatting and allows for complex multi-line configurations.
///


## Reference

- **Parameter Overrides**: {doc}`nemo-evaluator CLI Reference <../../nemo-evaluator/reference/cli:parameter-overrides>` - Complete guide to available parameters and override syntax
- **Troubleshooting**: See [Configuration Troubleshooting](../index.md#troubleshooting) for debug mode, testing, and common issues
- **Interceptors Documentation**: {doc}`Configuring Interceptors <../../nemo-evaluator/reference/configuring-interceptors>` - How to modify request/response payloads and add custom parameters
- **Task Configuration**: {doc}`nemo-evaluator Reference <../../nemo-evaluator/index>` - Complete nemo-evaluator documentation
