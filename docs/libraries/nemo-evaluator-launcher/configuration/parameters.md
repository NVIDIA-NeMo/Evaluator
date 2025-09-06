(config-parameters)=

# Parameter Reference

Detailed reference for all task parameters, adapter configuration, and request settings.

## Common Task Parameters

These parameters can be overridden per task or globally:

```yaml
config.params:
  temperature: 0.7              # Sampling temperature (0.0-1.0)
  top_p: 0.95                   # Nucleus sampling parameter
  top_k: 50                     # Top-k sampling parameter
  max_new_tokens: 2048          # Maximum tokens to generate
  parallelism: 16               # Number of parallel requests
  request_timeout: 3600         # Request timeout in seconds
  limit_samples: null           # Limit number of samples (for testing)
  
  # Task-specific parameters
  extra:
    n_samples: 1                # Number of samples per prompt (for code gen)
    pass_at_k: [1, 5, 10]      # Pass@k metrics (for code gen)
```

## Adapter Configuration

Control how requests are formatted and processed:

```yaml
target.api_endpoint.adapter_config:
  use_reasoning: false          # Strip reasoning tokens from responses
  use_system_prompt: true       # Include system prompts
  custom_system_prompt: "Think step by step."
  
  # Request formatting
  add_bos_token: false         # Add beginning-of-sequence token
  add_eos_token: false         # Add end-of-sequence token
  
  # Response processing
  strip_whitespace: true        # Strip leading/trailing whitespace
  normalize_newlines: true      # Normalize newline characters
```

## See Also

- [Structure & Sections](structure.md) - Core configuration sections
- [Examples](examples.md) - Complete configuration examples
- [Advanced Patterns](advanced.md) - Complex configuration patterns
