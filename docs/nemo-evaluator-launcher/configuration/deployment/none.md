# None Deployment

The `none` deployment option does not deploy a model. Instead, you provide an existing OpenAI-compatible endpoint.

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

- **`type: none`**: Specifies no deployment
- **`target.api_endpoint.model_id`**: Model identifier
- **`target.api_endpoint.url`**: Full URL for your OpenAI-compatible endpoint
- **`target.api_endpoint.api_key`**: API key used for authentication

**Note**:

- Do not include the actual API key in the configuration file—use environment variables instead.
- If your model does not require an API key, omit the `api_key` field.

The following examples show typical configurations:

- [Local None Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml) - Local evaluation with an existing endpoint
- [Lepton None Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/lepton_none_llama_3_1_8b_instruct.yaml) - Lepton evaluation with an existing endpoint
- [Slurm None Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/slurm_no_deployment_llama_3_1_8b_instruct.yaml) - Slurm evaluation with an existing endpoint
- [Local with Metadata](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_with_user_provided_metadata.yaml) - Local evaluation with custom metadata
- [Auto Export Example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml) - Local evaluation with automatic result export

- [Local with Advanced Configuration](../../../../packages/nemo-evaluator-launcher/examples/local_custom_config_seed_oss_36b_instruct.yaml) - Local evaluation with a custom configuration

## Use Cases

Use this deployment option when you:

- Have an existing model endpoint
- Need to test models hosted by third-party services
- Need to separate deployment and evaluation environments
- Use a custom deployment solution

## Reference

- [None Configuration File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/none.yaml)
