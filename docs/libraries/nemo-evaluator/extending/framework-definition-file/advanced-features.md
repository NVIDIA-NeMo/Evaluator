(advanced-features)=

# Advanced Features

This section covers advanced FDF features including conditional parameter handling, parameter inheritance, and dynamic configuration.

## Conditional Parameter Handling

Use Jinja2 conditionals to handle optional parameters:

```yaml
command: >-
  example_eval --model {{target.api_endpoint.model_id}}
  {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}
  {% if config.params.extra.add_system_prompt %} --add_system_prompt {% endif %}
  {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}
```

### Common Conditional Patterns

**Check for null/none values**:
```yaml
{% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}
```

**Check for boolean flags**:
```yaml
{% if config.params.extra.add_system_prompt %} --add_system_prompt {% endif %}
```

**Check if variable is defined**:
```yaml
{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}
```

**Check for specific values**:
```yaml
{% if target.api_endpoint.type == "chat" %} --use_chat_format {% endif %}
```

## Parameter Inheritance

Parameters follow a hierarchical override system:

1. **Framework defaults** (4th priority) - Lowest priority
2. **Evaluation defaults** (3rd priority)
3. **User configuration** (2nd priority)
4. **CLI overrides** (1st priority) - Highest priority

### Inheritance Example

**Framework defaults (framework.yml)**:
```yaml
defaults:
  config:
    params:
      temperature: 0.0
      max_new_tokens: 4096
```

**Evaluation defaults (framework.yml)**:
```yaml
evaluations:
  - name: humaneval
    defaults:
      config:
        params:
          max_new_tokens: 1024  # Overrides framework default
```

**User configuration (config.yaml)**:
```yaml
config:
  params:
    max_new_tokens: 512  # Overrides evaluation default
    temperature: 0.7      # Overrides framework default
```

**CLI overrides**:
```bash
eval-factory run_eval --overrides config.params.temperature=1.0
# Overrides all previous values
```

For more information on how to use these overrides, see {ref}`parameter-overrides` documentation.

## Dynamic Configuration

Use template variables to reference other configuration sections. For example, re-use `config.output_dir` for `--cache` input argument:

```yaml
command: >-
  example_eval --output {{config.output_dir}} --cache {{config.output_dir}}/cache
```

### Dynamic Configuration Patterns

**Reference output directory**:
```yaml
--results {{config.output_dir}}/results.json
--logs {{config.output_dir}}/logs
```

**Compose complex paths**:
```yaml
--data_dir {{config.output_dir}}/data/{{config.params.task}}
```

**Use task type in paths**:
```yaml
--cache {{config.output_dir}}/cache/{{config.type}}
```

**Reference model information**:
```yaml
--model_name {{target.api_endpoint.model_id}}
--endpoint {{target.api_endpoint.url}}
```

## Environment Variable Handling

**Export API keys conditionally**:
```yaml
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
```

**Set multiple environment variables**:
```yaml
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
{% if config.params.extra.custom_env is defined %}export CUSTOM_VAR={{config.params.extra.custom_env}} && {% endif %}
```

## Complex Command Templates

**Multi-line commands with conditionals**:
```yaml
command: >-
  {% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
  example_eval 
    --model {{target.api_endpoint.model_id}}
    --task {{config.params.task}}
    --url {{target.api_endpoint.url}}
    {% if config.params.limit_samples is not none %}--first_n {{config.params.limit_samples}}{% endif %}
    {% if config.params.extra.add_system_prompt %}--add_system_prompt{% endif %}
    {% if target.api_endpoint.type == "chat" %}--use_chat_format{% endif %}
    --output {{config.output_dir}}
    {% if config.params.extra.args is defined %}{{ config.params.extra.args }}{% endif %}
```

## Best Practices

- Always check if optional parameters are defined before using them
- Use `is not none` for nullable parameters with default values
- Use `is defined` for truly optional parameters that may not exist
- Keep conditional logic simple and readable
- Document custom parameters in the framework's README
- Test all conditional branches with different configurations
- Use parameter inheritance to avoid duplication
- Leverage dynamic paths to organize output files

