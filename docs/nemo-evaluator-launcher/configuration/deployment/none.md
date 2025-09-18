# None Deployment

The "none" deployment option means no model deployment is performed. Instead, you provide an existing OpenAI-compatible endpoint.

## Configuration

```yaml
deployment:
  type: none

target:
  api_endpoint:
    model_id: your-model-name
    url: https://your-endpoint.com/v1/chat/completions
    api_key: your-api-key
```

## Key Settings

- **`type: none`**: Indicates no deployment will be performed
- **`target.api_endpoint.model_id`**: Name/identifier of your model
- **`target.api_endpoint.url`**: Full URL to your OpenAI-compatible endpoint
- **`target.api_endpoint.api_key`**: API key for authentication

**Note**: 
- Do not provide the exact value of your API key in the config file - use environment variable names instead
- If your model does not require an API key, you can skip the `api_key` field entirely

Examples:
- [Local None Example](../../../../packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml) - Local evaluation with existing endpoint
- [Lepton None Example](../../../../packages/nemo-evaluator-launcher/examples/lepton_none_llama_3_1_8b_instruct.yaml) - Lepton evaluation with existing endpoint
- [Slurm None Example](../../../../packages/nemo-evaluator-launcher/examples/slurm_no_deployment_llama_3_1_8b_instruct.yaml) - Slurm evaluation with existing endpoint
- [Local with Metadata](../../../../packages/nemo-evaluator-launcher/examples/local_with_user_provided_metadata.yaml) - Local evaluation with custom metadata
- [Auto Export Example](../../../../packages/nemo-evaluator-launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml) - Local evaluation with automatic result export

- [Local with advanced configuration](../../../../packages/nemo-evaluator-launcher/examples/local_custom_config_seed_oss_36b_instruct.yaml) - Local evaluation with custom configuration

## Use Cases

This deployment option is useful when:
- You have an existing model endpoint
- You want to evaluate models hosted by third-party services
- You want to separate deployment and evaluation environments
- You're using a custom deployment solution

## Reference

- [None Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/none.yaml)
