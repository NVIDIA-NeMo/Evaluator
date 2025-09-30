(framework-definition-file)=
# Framework Definition File (FDF)

The Framework Definition File (FDF) is a YAML configuration file that serves as the single source of truth for evaluation harnesses in the NeMo Evaluator ecosystem. It defines how evaluation harnesses configure, execute, and integrate with the Evaluation Factory system.

## Overview

An FDF defines:

- **Framework metadata**: Name, description, package information
- **Default configurations**: Parameters, commands, and settings
- **Evaluation types**: Available evaluation tasks and their configurations
- **Execution commands**: Jinja2-templated commands for running evaluations
- **API compatibility**: Supported endpoint types and configurations

## File Structure

The FDF follows a hierarchical structure with 3 main sections:

```yaml
framework:          # Framework identification and metadata
defaults:           # Default configurations and commands
evaluations:        # Available evaluation types
```

## Section Details

### 1. Framework Section

The `framework` section contains basic identification and metadata for your evaluation harness.

```yaml
framework:
  name: example-evaluation-framework          # Internal framework identifier
  pkg_name: example_evaluation_framework     # Python package name
  full_name: Example Evaluation Harness    # Human-readable display name
  description: A comprehensive example...    # Detailed description
  url: https://github.com/example/...        # Original repository URL
```

**Fields:**

- **`name`**: Unique identifier that the system uses internally
- **`pkg_name`**: Python package name (typically matches the name)
- **`full_name`**: Human-readable name displayed in UI and documentation
- **`description`**: Comprehensive description of the framework's purpose
- **`url`**: Link to the original benchmark repository

### 2. Defaults Section

The `defaults` section defines the default configuration and execution command that all evaluations use unless overridden. You can override it through the `--overrides` flag (refer to {ref}`Parameter Overrides <cli-reference:parameter-overrides>`) or the {ref}`Run Configuration file <cli-reference:run-configuration>`.

### Command Template

The `command` field uses Jinja2 templating to dynamically generate execution commands based on configuration parameters. This is where you define how the system calls your evaluation harness's CLI interface.

**Important Note**: `example_eval` is a placeholder representing your actual CLI command. When integrating your harness, replace this with your real command (e.g., `lm-eval`, `bigcode-eval`, `gorilla-eval`, etc.).

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

**What this does:**

- **`example_eval`**: Your harness's actual CLI command (replace with your command)
- **Template variables**: Dynamically insert values from the evaluation configuration
- **Conditional logic**: Include parameters when they have values
- **Environment setup**: Export API keys before running the command

#### Key Template Variables

**Target API Endpoint Variables:**

- **`{{target.api_endpoint.api_key}}`**: Name of the environment variable storing API key
- **`{{target.api_endpoint.model_id}}`**: Target model identifier
- **`{{target.api_endpoint.stream}}`**: Whether to stream responses
- **`{{target.api_endpoint.type}}`**: Target endpoint type
- **`{{target.api_endpoint.url}}`**: URL of the model
- **`{{target.api_endpoint.adapter_config}}`**: Adapter configuration

**Evaluation Configuration Variables:**

- **`{{config.output_dir}}`**: Output directory for results
- **`{{config.type}}`**: Task type
- **`{{config.supported_endpoint_types}}`**: Supported endpoint types (chat/completions)

**Configuration Parameters:**

- **`{{config.params.task}}`**: Evaluation task type
- **`{{config.params.temperature}}`**: Model temperature setting
- **`{{config.params.limit_samples}}`**: Sample limit for evaluation
- **`{{config.params.max_new_tokens}}`**: Token generation limit
- **`{{config.params.max_retries}}`**: Number of REST request retries
- **`{{config.params.parallelism}}`**: Parallelism level
- **`{{config.params.request_timeout}}`**: REST response timeout
- **`{{config.params.top_p}}`**: Top-p sampling parameter
- **`{{config.params.extra}}`**: Framework-specific parameters

#### Configuration Defaults

```yaml
defaults:
  config:
    params:
      limit_samples: null           # No limit on samples by default
      max_new_tokens: 4096         # Token generation limit
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

- **Core Parameters**: Basic evaluation settings (temperature, token limits)
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

Use Jinja2 conditionals to handle optional parameters. This ensures that your CLI command includes parameters when they have values, preventing errors from undefined or null parameters:

```yaml
command: >-
  example_eval --model {{target.api_endpoint.model_id}}
  {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}
  {% if config.params.extra.add_system_prompt %} --add_system_prompt {% endif %}
  {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}
```

### Parameter Inheritance

Parameters follow a hierarchical override system:

- **Framework defaults** (fourth priority)
- **Evaluation defaults** (third priority)
- **User configuration** (second priority)
- **CLI overrides** (first priority)

For more information about using these overrides, refer to the {ref}`CLI Reference <cli-reference:parameter-overrides>` documentation.

### Dynamic Configuration

Use template variables to reference other configuration sections. For example, reuse `config.output_dir` for `--cache` input argument:

```yaml
command: >-
  example_eval --output {{config.output_dir}} --cache {{config.output_dir}}/cache
```

## Integration with Evaluation Factory

## File Location

Place your FDF in the `core_evals/<framework_name>/` directory of your framework package:

```bash
your-framework/
├── core_evals/
│   └── your_framework/
│       ├── framework.yml           # This is your FDF
│       ├── framework_entrypoint.py # This is an entrypoint to execute evaluation (usually pre-defined)
│       ├── output.py               # Output parser (custom)
│       └── __init__.py             # Empty init file
├── setup.py                        # Package configuration
└── README.md                       # Framework documentation
```

## Validation

The NeMo Evaluator system validates the FDF to ensure:

- Required fields are present
- Parameter types are correct
- Template syntax is valid
- Configuration consistency

## Troubleshooting

### Common Issues

1. **Template Errors**: Check Jinja2 syntax and variable references
2. **Parameter Conflicts**: Ensure parameter names don't conflict between sections
3. **Type Mismatches**: Verify parameter types match expected values
4. **Missing Fields**: Ensure the system defines all required fields

### Debug Mode

Enable debug logging to view how the system processes your FDF:

```bash
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
eval-factory run_eval --eval_type your_evaluation
```
