# Reasoning

Handle reasoning tokens in responses and track reasoning metrics for models that support chain-of-thought processing.

## Configuration

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_reasoning=True,target.api_endpoint.adapter_config.end_reasoning_token="</think>",target.api_endpoint.adapter_config.start_reasoning_token="<think>"'
```

### YAML Configuration
```yaml
interceptors:
  - name: "reasoning"
    enabled: true
    config:
      end_reasoning_token: "</think>"
      start_reasoning_token: "<think>"
      add_reasoning: true
      enable_reasoning_tracking: true
      include_if_not_finished: true
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_reasoning_token` | Token that marks the start of reasoning | `"<think>"` |
| `end_reasoning_token` | Token that marks the end of reasoning | `"</think>"` |
| `add_reasoning` | Add reasoning tokens to requests | true |
| `enable_reasoning_tracking` | Track reasoning metrics | true |
| `include_if_not_finished` | Include partial reasoning if generation stops | true |

## Use Cases

- **Chain-of-thought evaluation**: Evaluate models that use reasoning tokens
- **Reasoning analysis**: Track and analyze reasoning patterns in model responses
- **Performance optimization**: Measure reasoning overhead and effectiveness
