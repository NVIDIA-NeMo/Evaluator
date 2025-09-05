# Payload Modification

Add, remove, or modify request parameters before sending to the model endpoint.

## Configuration

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.params_to_add={"temperature": 0.7},target.api_endpoint.adapter_config.params_to_remove=["max_tokens"]'
```

### YAML Configuration
```yaml
interceptors:
  - name: "payload_modifier"
    enabled: true
    config:
      params_to_add:
        temperature: 0.7
        max_tokens: 1000
      params_to_remove:
        - "max_tokens"
      params_to_rename:
        "old_param": "new_param"
```

## Configuration Options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `params_to_add` | Parameters to add to requests | `{"temperature": 0.7}` |
| `params_to_remove` | Parameters to remove from requests | `["max_tokens"]` |
| `params_to_rename` | Parameters to rename | `{"old_param": "new_param"}` |

## Use Cases

- **Parameter standardization**: Ensure consistent parameters across evaluations
- **Model-specific tuning**: Add parameters required by specific model endpoints
- **Legacy compatibility**: Rename parameters for compatibility with different API versions
