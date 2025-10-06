# SGLang Deployment

SGLang is a fast serving framework for large language models and vision language models.

## Configuration

See the complete configuration structure in the [SGLang Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/sglang.yaml).

## Key Settings

- **`image`**: Docker image for SGLang server
- **`checkpoint_path`**: Path to model checkpoints (required if not using Hugging Face)
- **`hf_model_handle`**: Hugging Face model identifier (required if not using local checkpoint)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the SGLang server (default: 8000)
- **`tensor_parallel_size`**: Number of GPUs for tensor parallelism (default: 8)
- **`data_parallel_size`**: Number of replicas for data parallelism (default: 1)
- **`extra_args`**: Additional arguments passed to SGLang server
- **`env_vars`**: Environment variables for the container

Tips:
- adjust `tensor_parallel_size`, `data_parallel_size` to your hardware and model size
- use `hf_model_handle` for models from Hugging Face
- adjust `HF_HOME` and mounts to speed up model loading
- **Note**: The `${oc.select:deployment.hf_model_handle,/checkpoint}` syntax means:
  - If `hf_model_handle` is provided, use that (downloads from Hugging Face)
  - If `checkpoint_path` is provided instead, use that local path


## Reference

- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/sglang.yaml)
