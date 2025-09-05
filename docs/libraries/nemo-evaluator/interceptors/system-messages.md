# System Messages

Modify system messages and prompts in evaluation requests.

## Configuration

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_system_prompt=True,target.api_endpoint.adapter_config.custom_system_prompt="You are a helpful assistant."'
```

### YAML Configuration
```yaml
interceptors:
  - name: "system_message"
    enabled: true
    config:
      system_message: "You are a helpful assistant."
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `system_message` | Custom system message to use | None |

## Use Cases

- **Consistent prompting**: Ensure all evaluation requests use the same system message
- **Role-specific evaluation**: Set specific roles or behaviors for different evaluation scenarios
- **Prompt standardization**: Maintain consistent prompting across different benchmarks
