# SGLang Deployment

SGLang is a high-performance serving framework for large language models (LLMs) and vision-language models.

## Configuration

Refer to the [SGLang deployment configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/sglang.yaml) for the complete configuration structure.

## Key Settings

- **`image`**: Docker image for the SGLang server
- **`checkpoint_path`**: Path to the model checkpoints (required if not using Hugging Face)
- **`hf_model_handle`**: Hugging Face model identifier (required if not using a local checkpoint)
- **`served_model_name`**: Name used when serving the model (required)
- **`port`**: Port for the SGLang server (default: 8000)
- **`tensor_parallel_size`**: Number of GPUs for tensor parallelism (default: 8)
- **`data_parallel_size`**: Number of replicas for data parallelism (default: 1)
- **`extra_args`**: Additional arguments passed to the SGLang server
- **`env_vars`**: Environment variables to set in the container

Use the following tips to optimize your configuration:
- Adjust `tensor_parallel_size` and `data_parallel_size` to match your hardware and model size.
- Use `hf_model_handle` for models hosted on Hugging Face.
- Set `HF_HOME` and configure mounts to speed up model loading.
- **Note**: The `${oc.select:deployment.hf_model_handle,/checkpoint}` expression resolves as follows:
  - If `hf_model_handle` is provided, Evaluator downloads the model from Hugging Face.
  - If `checkpoint_path` is provided instead, use that local path.


## References

- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang deployment configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/sglang.yaml)
