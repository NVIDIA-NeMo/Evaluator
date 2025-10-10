(fdf-troubleshooting)=

# Troubleshooting

This section covers common issues encountered when creating and using Framework Definition Files.

## Common Issues

::::{dropdown} Template Errors
:icon: code-square

**Symptom**: Template rendering fails with syntax errors.

**Causes**:
- Missing closing braces in Jinja2 templates
- Invalid variable references
- Incorrect conditional syntax

**Solutions**:

Check that all template variables use correct syntax:
```yaml
# Correct
{{target.api_endpoint.model_id}}

# Incorrect
{{target.api_endpoint.model_id}
{target.api_endpoint.model_id}}
```

Verify conditional statements are properly formatted:
```jinja
# Correct
{% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}

# Incorrect
{% if config.params.limit_samples != none %} --first_n {{config.params.limit_samples}}{% end %}
```

::::

::::{dropdown} Parameter Conflicts
:icon: code-square

**Symptom**: Parameters are not overriding as expected.

**Causes**:
- Incorrect parameter paths in overrides
- Type mismatches between default and override values
- Missing parameter definitions in defaults section
- Incorrect indentation in the YAML config

**Solutions**:

Ensure parameter paths are correct:
```bash
# Correct
--overrides config.params.temperature=0.7

# Incorrect
--overrides params.temperature=0.7
--overrides config.temperature=0.7
```

Verify parameter types match:
```yaml
# Correct
temperature: 0.7        # Float

# Incorrect
temperature: "0.7"      # String
```

Make sure to use the correct indentation:
```yaml
# Correct
defaults:
  config:
    params:
      limit_samples: null
      max_new_tokens: 4096   # max_new_tokens belongs to params

# Incorrect
defaults:
  config:
    params:
      limit_samples: null
    max_new_tokens: 4096     # max_new_tokens is outside of params
```

::::

::::{dropdown} Type Mismatches
:icon: code-square

**Symptom**: Validation errors about incorrect parameter types.

**Causes**:
- String values used for numeric parameters
- Missing quotes for string values
- Boolean values as strings

**Solutions**:

Use correct types for each parameter:
```yaml
# Correct
temperature: 0.7              # Float
max_new_tokens: 1024          # Integer
add_system_prompt: false      # Boolean
task: "humaneval"             # String

# Incorrect
temperature: "0.7"            # String instead of float
max_new_tokens: "1024"        # String instead of integer
add_system_prompt: "false"    # String instead of boolean
```

::::

::::{dropdown} Missing Fields
:icon: code-square

**Symptom**: Validation fails with "required field missing" errors.

**Causes**:
- Incomplete framework section
- Missing required parameters
- Omitted evaluation configurations

**Solutions**:

Ensure all required framework fields are present:
```yaml
framework:
  name: your-framework          # Required
  pkg_name: your_framework      # Required
  full_name: Your Framework     # Required
  description: Description...   # Required
  url: https://github.com/...   # Required
```

Include all required evaluation fields:
```yaml
evaluations:
  - name: task_name                    # Required
    description: Task description      # Required
    defaults:
      config:
        type: "task_type"              # Required
        supported_endpoint_types:      # Required
          - chat
```

::::

## Debug Mode

Enable debug logging to see how your FDF is processed. Use the `--debug` flag or set the logging level:

```bash
# Using debug flag
nemo-evaluator run_eval --eval_type your_evaluation --debug

# Or set log level environment variable
export LOG_LEVEL=DEBUG
nemo-evaluator run_eval --eval_type your_evaluation
```

### Debug Output

Debug mode provides detailed information about:

- FDF discovery and loading
- Template variable resolution
- Parameter inheritance and overrides
- Command generation
- Validation errors with stack traces

### Interpreting Debug Logs

Debug logs show the FDF loading and processing workflow. Key information includes:

**FDF Loading**: Shows which framework.yml files are discovered and loaded

**Template Rendering**: Displays template variable substitution and final rendered commands

**Parameter Overrides**: Shows how configuration values cascade through the inheritance hierarchy

**Validation Errors**: Provides detailed error messages when FDF structure or templates are invalid

## Validation Tips

**Test incrementally**: Start with a minimal FDF and add sections progressively.

**Validate templates separately**: Test Jinja2 templates in isolation before adding to FDF.

**Check references**: Ensure all template variables reference existing configuration paths.

**Use examples**: Base your FDF on existing, working examples from the NeMo Evaluator repository.

**Verify syntax**: Use a YAML validator to catch formatting errors.

## Getting Help

If you encounter issues not covered here:

1. Check the FDF examples in the NeMo Evaluator repository
2. Review debug logs for specific error messages
3. Verify your framework's CLI works independently
4. Consult the {ref}`extending-evaluator` documentation
5. Search for similar issues in the project's issue tracker

