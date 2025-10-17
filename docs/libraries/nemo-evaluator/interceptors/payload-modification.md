(interceptor-payload-modification)=

# Payload Modification

## Overview

`PayloadParamsModifierInterceptor` adds, removes, or modifies request parameters before sending them to the model endpoint.

:::{tip}
Add {ref}`interceptor-request-logging` to your interceptor chain to verify if your requests are modified correctly.
:::

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.params_to_add={"temperature":0.7},target.api_endpoint.adapter_config.params_to_remove=["max_tokens"]'
```

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "payload_modifier"
          enabled: true
          config:
            params_to_add:
              temperature: 0.7
              top_p: 0.9
            params_to_remove:
              - "top_k"              # top-level field in the payload to remove
              - "reasoning_content"  # field in the message to remove
            params_to_rename:
              old_param: "new_param"
        - name: "endpoint"
          enabled: true
          config: {}
```

:::{note}
In the example above, the `reasoning_content` field will be removed recursively from all messages in the payload.
:::

## Configuration Options

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `params_to_add` | `dict` | Dictionary of parameters to add to requests | `{"temperature": 0.7, "top_p": 0.9}` |
| `params_to_remove` | `list` | List of parameter names to recursively remove from requests | `["top_k", "frequency_penalty"]` |
| `params_to_rename` | `dict` | Dictionary mapping old parameter names to new names | `{"old_param": "new_param"}` |

:::{note}
The interceptor applies operations in the following order: remove → add → rename. This means you can remove a parameter and then add a different value for the same parameter name.
:::

## Use Cases

### Parameter Standardization

Ensure consistent parameters across evaluations by adding or removing parameters:

```yaml
config:
  params_to_add:
    temperature: 0.7
    top_p: 0.9
  params_to_remove:
    - "frequency_penalty"
    - "presence_penalty"
```

### Model-Specific Configuration

Add parameters required by specific model endpoints, such as chat template configuration:

```yaml
config:
  params_to_add:
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
```

### API Compatibility

Rename parameters for compatibility with different API versions or endpoint specifications:

```yaml
config:
  params_to_rename:
    max_new_tokens: "max_tokens"
    num_return_sequences: "n"
```
