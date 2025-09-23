# Evaluation Configuration

An evaluation configuration defines which benchmarks to run and how to run them. Use the same configuration across all executors to launch the same tasks.

**Important:** Each task has default values that you can override. For a complete list of override options, refer to the NeMo Evaluator CLI reference: [Parameter overrides](nemo-evaluator/reference/cli.md#parameter-overrides).

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

### Global Overrides

- **`overrides`**: Parameter overrides that apply to all tasks
- **`env_vars`**: Environment variables that apply to all tasks

### Task Configuration

- **`tasks`**: List of evaluation tasks to run
- **`name`**: Name of the benchmark task
- **`overrides`**: Task-specific parameter overrides
- **`env_vars`**: Task-specific environment variables
- **`config`**: Custom configuration object for advanced task setup

For a comprehensive list of available tasks, their descriptions, and task-specific parameters, refer to the [NeMo Evaluator supported tasks](nemo-evaluator/reference/containers.md).

## Advanced Task Configuration

### Parameter Overrides

The overrides system enables you to use the full flexibility of the common endpoint interceptors and the task configuration layer. This is where NeMo Evaluator intersects with NeMo Evaluator Launcher, providing a unified configuration interface.

#### Global Override Examples

```yaml
evaluation:
  overrides:
    config.params.request_timeout: 3600
    config.params.temperature: 0.7
    target.api_endpoint.adapter_config.use_system_prompt: true
    target.api_endpoint.adapter_config.custom_system_prompt: "Think step by step."
```

#### Task-Specific Overrides

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

### Environment Variables

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

Use an evaluation configuration when you want to:

- **Change Default Sampling Parameters**: Adjust `temperature`, `top_p`, and `max_new_tokens` for different tasks
- **Set Custom System Prompts**: Add task-specific instructions or reasoning prompts (refer to [System Message Interceptor](nemo-evaluator/reference/configuring-interceptors.md#5-system-message-interceptor))
- **Add, Remove, or Rename Special Parameters**: Change the payload structure (for example, add `"reasoning": "thinking"`) (refer to [Payload Modifier Interceptor](nemo-evaluator/reference/configuring-interceptors.md#6-payload-modifier-interceptor))
- **Change Default Task Values**: Override benchmark-specific default configurations
- **Parameterize the Judge**: Configure evaluation judge models and their parameters for scoring (refer to [Parameter Overrides](nemo-evaluator/reference/cli.md#parameter-overrides))
- **Debug and Test**: For example, launch with a limited number of samples
- **Adjust Endpoint Capabilities**: Configure request timeouts, max retries, and parallel request limits
- **Advanced Configuration**: Use the `config` field for complex interceptor chains, custom adapter configurations, and fine-grained control over the evaluation setup

/// tip | Long String Overrides
To override long strings (for example, system prompts), use YAML multi-line syntax with `>-`:

```yaml
target.api_endpoint.adapter_config.params_to_add: >-
  <HERE_WRITE_WHATEVER_YOU_WANT_AND_IT_WILL_BE_PASSED_WITHOUT_ANY_MODIFICATIONS_TO_OVERRIDES>
```

This preserves formatting and enables complex multi-line configurations.
///

## Custom Configuration

For advanced use cases, you can use the `config` field to provide a complete configuration object that maps directly to the [NeMo Evaluator configuration structure](nemo-evaluator/reference/api.md#evaluationconfig). This enables fine-grained control over:

- Complex configurations that are difficult to pass by command-line arguments (for example, long system prompts)
- Custom interceptor chains and adapter settings

```yaml
evaluation:
  tasks:
    - name: mgsm
      config:
        target:
          api_endpoint:
            adapter_config:
              caching_dir: /results/cache
              endpoint_type: chat
              interceptors:
                - name: system_message
                  enabled: true
                  config:
                    system_message: |
                      You are an expert mathematician and problem solver.
                      Always think step by step and be thorough in your explanations.
                - name: endpoint
                  enabled: true
```

/// note | Important Configuration Notes

- **Results Path**: You can use `/results/` as the output path because this directory is always mounted in containers
- **Endpoint Adapter**: Always include the `endpoint` interceptor with `enabled: true`. Without it, the configuration fails
///

The `config` field provides direct access to the underlying NeMo Evaluator configuration structure, allowing you to:

- Configure complex interceptor chains
- Set up custom adapter configurations
- Define target-specific settings
- Override framework-specific parameters

For more details about the configuration structure, refer to the [NeMo Evaluator API documentation](nemo-evaluator/reference/api.md#evaluationconfig).

**Example**: Refer to [local_custom_config_seed_oss_36b_instruct.yaml](../../../packages/nemo-evaluator-launcher/examples/local_custom_config_seed_oss_36b_instruct.yaml) for a complete example that uses a custom configuration.

## References

- **Parameter overrides**: NeMo Evaluator CLI reference—[Parameter overrides](nemo-evaluator/reference/cli.md#parameter-overrides)
- **Troubleshooting**: Refer to [Configuration troubleshooting](../index.md#troubleshooting) for debug mode, testing, and common issues
- **Interceptors**: Refer to [Configuring interceptors](nemo-evaluator/reference/configuring-interceptors.md) for modifying request and response payloads and adding custom parameters
- **Task configuration**: Refer to the [NeMo Evaluator reference](nemo-evaluator/index.md) for complete documentation
