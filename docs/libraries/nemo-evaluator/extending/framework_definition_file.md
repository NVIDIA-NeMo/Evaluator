# Framework Definition File (FDF)

The Framework Definition File (FDF) is a YAML configuration file that serves as the single source of truth for evaluation frameworks in the NeMo Evaluator ecosystem. It defines how evaluation frameworks are configured, executed, and integrated with the Eval Factory system.

## Overview

An FDF defines:
- **Framework metadata**: Name, description, package information
- **Default configurations**: Parameters, commands, and settings
- **Evaluation types**: Available evaluation tasks and their configurations
- **Execution commands**: Jinja2-templated commands for running evaluations
- **API compatibility**: Supported endpoint types and configurations

## File Structure

The FDF follows a hierarchical structure with three main sections:

```yaml
framework:          # Framework identification and metadata
defaults:           # Default configurations and commands
evaluations:        # Available evaluation types
```

## Section Details

### 1. Framework Section

The `framework` section contains basic identification and metadata for your evaluation framework.

```yaml
framework:
  name: example-evaluation-framework          # Internal framework identifier
  pkg_name: example_evaluation_framework     # Python package name
  full_name: Example Evaluation Framework    # Human-readable display name
  description: A comprehensive example...    # Detailed description
  url: https://github.com/example/...        # Original repository URL
```

**Fields:**
- **`name`**: Unique identifier used internally by the system
- **`pkg_name`**: Python package name (usually matches the name)
- **`full_name`**: Human-readable name displayed in UI and documentation
- **`description`**: Comprehensive description of the framework's purpose
- **`url`**: Link to the original benchmark repository

### 2. Defaults Section

The `defaults` section defines the default configuration and execution command that will be used across all evaluations unless overridden. Overriding is supported either through `--overrides` flag (see [Parameter Overrides](../reference/cli.md#parameter-overrides)) or [Run Configuration file](../reference/cli.md#run-configuration).

#### Command Template

The `command` field uses Jinja2 templating to dynamically generate execution commands based on configuration parameters.

```yaml
defaults:
  command: >-
    {% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
    example_eval --model {{target.api_endpoint.model_id}} 
    --task {{config.params.task}}
    --url {{target.api_endpoint.url}} 
    --temperature {{config.params.temperature}}
    # ... additional parameters
```

**Key Template Variables:**
- **`{{target.api_endpoint.model_id}}`**: Target model identifier
- **`{{config.params.task}}`**: Evaluation task type
- **`{{config.output_dir}}`**: Output directory for results
- **`{{config.params.temperature}}`**: Model temperature setting
- **`{{config.params.limit_samples}}`**: Sample limit for evaluation

#### Configuration Defaults

```yaml
defaults:
  config:
    params:
      limit_samples: null           # No limit on samples by default
      max_new_tokens: 4096         # Maximum tokens to generate
      temperature: 0.0             # Deterministic generation
      top_p: 0.00001              # Nucleus sampling parameter
      parallelism: 10              # Number of parallel requests
      max_retries: 5               # Maximum API retry attempts
      request_timeout: 60          # Request timeout in seconds
      extra:                       # Framework-specific parameters
        n_samples: null            # Number of evaluation samples
        downsampling_ratio: null   # Data downsampling ratio
        add_system_prompt: false   # Include system prompt
        args: null                 # Additional CLI arguments
```

**Parameter Categories:**
- **Core Parameters**: Basic evaluation settings (temperature, max_tokens)
- **Performance Parameters**: Parallelism and timeout settings
- **Framework Parameters**: Task-specific configuration options
- **Extra Parameters**: Custom parameters specific to your framework

#### Target Configuration

```yaml
defaults:
  target:
    api_endpoint:
      type: chat                           # Default endpoint type
      supported_endpoint_types:            # All supported types
        - chat
        - completion
```

**Endpoint Types:**
- **`chat`**: Multi-turn conversation format (OpenAI chat completions)
- **`completion`**: Single-turn text completion format

### 3. Evaluations Section

The `evaluations` section defines the specific evaluation types available in your framework, each with its own configuration defaults.

```yaml
evaluations:
  - name: example_task_1                    # Evaluation identifier
    description: Basic functionality demo   # Human-readable description
    defaults:
      config:
        type: "example_task_1"             # Evaluation type identifier
        supported_endpoint_types:          # Supported endpoints for this task
          - chat
          - completion
        params:
          task: "example_task_1"           # Task-specific identifier
          temperature: 0.0                 # Task-specific temperature
          max_new_tokens: 1024             # Task-specific token limit
          extra:
            custom_key: "custom_value"     #  Task-specific custom param
```

**Evaluation Configuration:**
- **`name`**: Unique identifier for the evaluation type
- **`description`**: Clear description of what the evaluation measures
- **`type`**: Internal type identifier used by the framework
- **`supported_endpoint_types`**: API endpoint types compatible with this evaluation
- **`params`**: Task-specific parameter overrides

## Advanced Features

### Conditional Parameter Handling

Use Jinja2 conditionals to handle optional parameters:

```yaml
command: >-
  example_eval --model {{target.api_endpoint.model_id}}
  {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}
  {% if config.params.extra.add_system_prompt %} --add_system_prompt {% endif %}
  {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}
```

### Parameter Inheritance

Parameters follow a hierarchical override system:
1. **Framework defaults** (4th priority)
2. **Evaluation defaults** (3rd priority)
3. **User configuration** (2nd priority)
4. **CLI overrides** (1st priority)

For more information on how to use these overrides, see the [CLI Reference](../reference/cli.md#parameter-overrides) documentation.

### Dynamic Configuration

Use template variables to reference other configuration sections. For example, re-use `config.output_dir` for `--cache` input argument:

```yaml
command: >-
  example_eval --output {{config.output_dir}} --cache {{config.output_dir}}/cache
```

## Integration with Eval Factory

### File Location

Place your FDF in the `core_evals/<framework_name>/` directory of your framework package:

```
your-framework/
├── core_evals/
│   └── your_framework/
│       ├── framework.yml           # This is your FDF
|       ├── framework_entrypoint.py # This is an entrypoint to execute evaluation (usually pre-defined)
│       ├── output.py               # Output parser (custom)
│       └── __init__.py             # Empty init file
├── setup.py                        # Package configuration
└── README.md                       # Framework documentation
```

## Validation

The FDF is validated by the NeMo Evaluator system to ensure:
- Required fields are present
- Parameter types are correct
- Template syntax is valid
- Configuration consistency

## Troubleshooting

### Common Issues

1. **Template Errors**: Check Jinja2 syntax and variable references
2. **Parameter Conflicts**: Ensure parameter names don't conflict between sections
3. **Type Mismatches**: Verify parameter types match expected values
4. **Missing Fields**: Ensure all required fields are defined

### Debug Mode

Enable debug logging to see how your FDF is processed:

```bash
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
eval-factory run_eval --eval_type your_evaluation
```
