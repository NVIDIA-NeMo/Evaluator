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
- [Local None Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_llama_3_1_8b_instruct.yaml?ref_type=heads) - Local evaluation with existing endpoint
- [Lepton None Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/lepton_none_llama_3_1_8b_instruct.yaml?ref_type=heads) - Lepton evaluation with existing endpoint
- [Slurm None Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/slurm_no_deployment_llama_3_1_8b_instruct.yaml?ref_type=heads) - Slurm evaluation with existing endpoint
- [Local with Metadata](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_with_user_provided_metadata.yaml?ref_type=heads) - Local evaluation with custom metadata
- [Auto Export Example](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/examples/local_auto_export_llama_3_1_8b_instruct.yaml?ref_type=heads) - Local evaluation with automatic result export

## Use Cases

This deployment option is useful when:
- You have an existing model endpoint
- You want to evaluate models hosted by third-party services
- You want to separate deployment and evaluation environments
- You're using a custom deployment solution

## Reference

- [None Config File](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/blob/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/deployment/none.yaml?ref_type=heads)
