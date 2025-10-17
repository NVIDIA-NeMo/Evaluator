(evaluation-configuration)=

# Evaluation Configuration

Evaluation configuration defines which benchmarks to run and their configuration. It is common for all executors and can be reused between them to launch the exact same tasks.

**Important**: Each task has its own default values that you can override. For comprehensive override options, see {ref}`parameter-overrides`.

## Configuration Structure

```yaml
evaluation:
  overrides:  # Global overrides for all tasks
    config.params.request_timeout: 3600
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

### Global Overrides

- **`overrides`**: Parameter overrides that apply to all tasks
- **`env_vars`**: Environment variables that apply to all tasks

### Task Configuration

- **`tasks`**: List of evaluation tasks to run
- **`name`**: Name of the benchmark task
- **`overrides`**: Task-specific parameter overrides
- **`env_vars`**: Task-specific environment variables

For a comprehensive list of available tasks, their descriptions, and task-specific parameters, see {ref}`nemo-evaluator-containers`.

## Advanced Task Configuration

### Parameter Overrides

The overrides system is crucial for leveraging the full flexibility of the common endpoint interceptors and task configuration layer. This is where nemo-evaluator intersects with nemo-evaluator-launcher, providing a unified configuration interface.

#### Global Overrides

Settings applied to all tasks listed in the config.

```yaml
evaluation:
  overrides:
    config.params.request_timeout: 3600
    config.params.temperature: 0.7
```

#### Task-Specific Overrides

Parameters passed to a job for a single task. They take precedence over global evaluation settings.

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
```

### Environment Variables

Task-specifi environment variables. These parameters are set for a single job and don't affect other tasks:

```yaml
evaluation:
  tasks:
    - name: task_name1
      # HF_TOKEN and CUSTOM_VAR are available for task_name1
      env_vars:
        HF_TOKEN: MY_HF_TOKEN
        CUSTOM_VAR: CUSTOM_VALUE
    - name: task_name2    # HF_TOKEN and CUSTOM_VAR are not set for task_name2
```

## When to Use

Use evaluation configuration when you want to:

- **Change Default Sampling Parameters**: Adjust temperature, top_p, max_new_tokens for different tasks
- **Change Default Task Values**: Override benchmark-specific default configurations
- **Configure Task-Specific Parameters**: Set custom parameters for individual benchmarks (e.g., n_samples for code generation tasks)
- **Debug and Test**: Launch with limited samples for validation
- **Adjust Endpoint Capabilities**: Configure request timeouts, max retries, and parallel request limits

:::{tip}
For overriding long strings, use YAML multiline syntax with `>-`:

```yaml
config.params.extra.custom_field: >-
  This is a long string that spans multiple lines
  and will be passed as a single value with spaces
  replacing the newlines.
```

This preserves formatting and allows for complex multi-line configurations.
:::

## Reference

- **Parameter Overrides**: {ref}`parameter-overrides` - Complete guide to available parameters and override syntax
- **Adapter Configuration**: For advanced request/response modification (system prompts, payload modification, reasoning handling), see {ref}`nemo-evaluator-interceptors`
- **Task Configuration**: {ref}`lib-core` - Complete nemo-evaluator documentation
- **Available Tasks**: {ref}`nemo-evaluator-containers` - Browse all available evaluation tasks and benchmarks
