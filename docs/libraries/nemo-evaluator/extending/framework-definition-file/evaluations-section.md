(evaluations-section)=

# Evaluations Section

The `evaluations` section defines the specific evaluation types available in your framework, each with its own configuration defaults.

## Structure

```yaml
evaluations:
  - name: example_task_1                    # Evaluation identifier
    description: Basic functionality demo   # Human-readable description
    defaults:
      config:
        type: "example_task_1"             # Evaluation type identifier
        supported_endpoint_types:          # Supported endpoints for this task
          - chat
          - completions
        params:
          task: "example_task_1"           # Task-specific identifier
          temperature: 0.0                 # Task-specific temperature
          max_new_tokens: 1024             # Task-specific token limit
          extra:
            custom_key: "custom_value"     #  Task-specific custom param
```

## Fields

### name

**Type**: String  
**Required**: Yes

Unique identifier for the evaluation type. This is used to reference the evaluation in CLI commands and configurations.

**Example**:
```yaml
name: humaneval
```

### description

**Type**: String  
**Required**: Yes

Clear description of what the evaluation measures. This helps users understand the purpose and scope of the evaluation.

**Example**:
```yaml
description: Evaluates code generation capabilities using the HumanEval benchmark dataset
```

### type

**Type**: String  
**Required**: Yes

Internal type identifier used by the framework. This typically matches the `name` field but may differ based on your framework's conventions.

**Example**:
```yaml
type: "humaneval"
```

### supported_endpoint_types

**Type**: List of strings  
**Required**: Yes

API endpoint types compatible with this evaluation. Specify which endpoint types work with this evaluation task:

- `chat` - Conversational format with role-based messages
- `completions` - Single-turn text completion
- `vlm` - Vision-language model with image support
- `embedding` - Embedding generation for retrieval tasks

**Example**:
```yaml
supported_endpoint_types:
  - chat
  - completions
```

### params

**Type**: Object  
**Required**: No

Task-specific parameter overrides that differ from the framework-level defaults. Use this to customize settings for individual evaluation types.

**Example**:
```yaml
params:
  task: "humaneval"
  temperature: 0.0
  max_new_tokens: 1024
  extra:
    custom_key: "custom_value"
```

## Multiple Evaluations

You can define multiple evaluation types in a single FDF:

```yaml
evaluations:
  - name: humaneval
    description: Code generation evaluation
    defaults:
      config:
        type: "humaneval"
        supported_endpoint_types:
          - chat
          - completions
        params:
          task: "humaneval"
          max_new_tokens: 1024

  - name: mbpp
    description: Python programming evaluation
    defaults:
      config:
        type: "mbpp"
        supported_endpoint_types:
          - chat
        params:
          task: "mbpp"
          max_new_tokens: 512
```

## Best Practices

- Use descriptive names that indicate the evaluation purpose
- Provide comprehensive descriptions for each evaluation type
- List endpoint types that are actually supported and tested
- Override parameters when they differ from framework defaults
- Use the `extra` object for framework-specific custom parameters
- Group related evaluations together in the same FDF
- Test each evaluation type with all specified endpoint types

