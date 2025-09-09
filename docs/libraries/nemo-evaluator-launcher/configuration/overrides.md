(config-overrides)=

# Overrides and Composition

Learn how to customize configurations using command-line overrides, environment variables, and configuration composition patterns.

## Command Line Overrides

Use Hydra's override syntax to modify configurations:

```bash
# Single override
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results

# Multiple overrides
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=my-model \
  -o evaluation.tasks[0].name=hellaswag

# Add new configuration (+ prefix)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o +config.params.limit_samples=10
```

## Environment Variable References

Reference environment variables in configurations:

```yaml
# Direct reference
target:
  api_endpoint:
    api_key: ${oc.env:NGC_API_KEY}
    
# With default value
target:
  api_endpoint:
    api_key: ${oc.env:NGC_API_KEY,nvapi-default}
    
# In deployment environment
deployment:
  envs:
    HF_TOKEN: ${oc.env:HF_TOKEN}
    CUSTOM_VAR: ${oc.env:CUSTOM_VAR,default_value}
```

## Configuration Composition

Create modular configurations using Hydra's composition:

```yaml
# configs/base.yaml
target:
  api_endpoint:
    timeout: 60
    max_retries: 3
    
evaluation:
  overrides:
    config.params.temperature: 0.7
```

```yaml
# configs/my_evaluation.yaml
defaults:
  - base                      # Include base configuration
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: my_results

evaluation:
  tasks:
    - name: hellaswag
```

## See Also

- [Structure & Sections](structure.md) - Core configuration sections
- [Parameter Reference](parameters.md) - Detailed parameter specifications
- [Validation & Debugging](validation.md) - Configuration validation and troubleshooting
