(config-validation)=

# Validation and Debugging

Learn how to validate configurations, debug issues, and resolve common configuration errors.

## Configuration Validation

```bash
# Dry run to see resolved configuration
nv-eval run --config-dir examples --config-name my_config --dry-run

# Print configuration tree
nv-eval run --config-dir examples --config-name my_config --cfg job
```

## Common Validation Errors

**Missing Required Fields:**
```
Error: Missing required field 'execution.output_dir'
```

**Invalid Task Names:**
```
Error: Unknown task 'invalid_task'. Available tasks: hellaswag, arc_challenge, ...
```

**Configuration Conflicts:**
```
Error: Cannot specify both 'api_key' and 'api_key_name' in target.api_endpoint
```

## See Also

- [Structure & Sections](structure.md) - Core configuration sections
- [Overrides & Composition](overrides.md) - Configuration customization patterns
- [Parameter Reference](parameters.md) - Detailed parameter specifications
