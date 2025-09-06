(config-advanced)=

# Advanced Patterns

Real-world configuration examples showing advanced techniques for production deployments, automated workflows, and complex evaluation scenarios.

## Automatic Result Export

```yaml
# examples/local_auto_export_llama_3_1_8b_instruct.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results
  auto_export: true
  exporters:
    - type: mlflow
      tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}
    - type: wandb
      project: llm-evaluation
      entity: ${oc.env:WANDB_ENTITY}

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
```

## Custom Metadata Injection

```yaml  
# examples/local_with_user_provided_metadata.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results
  metadata:
    experiment_name: "baseline_evaluation"
    model_version: "v1.0"
    dataset_version: "2024-01"
    researcher: ${oc.env:USER}
    notes: "Initial baseline run for comparison"

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
      params:
        temperature: 0.0
        max_tokens: 512
    - name: arc_challenge
      params:
        temperature: 0.0
        max_tokens: 512
```

## Best Practices

1. **Use Environment Variables**: Store sensitive information in environment variables
2. **Modular Configs**: Create reusable base configurations  
3. **Descriptive Names**: Use clear, descriptive configuration file names
4. **Test with Dry Run**: Always test configurations with `--dry-run` first
5. **Version Control**: Store configurations in version control
6. **Documentation**: Document custom configurations and their purpose
7. **Sample Limits for Testing**: Use `limit_samples` for development and validation
8. **Metadata for Tracking**: Include experiment metadata for better result organization

## See Also

- [Examples](examples.md) - Complete configuration examples
- [Structure & Sections](structure.md) - Core configuration sections
- [Validation & Debugging](validation.md) - Configuration validation and troubleshooting
