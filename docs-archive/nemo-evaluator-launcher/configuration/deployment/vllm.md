# vLLM Deployment

vLLM is a fast and easy-to-use library for LLM inference and serving.

## Configuration

See the complete configuration structure in the [vLLM Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/vllm.yaml).

## Key Settings

- **`image`**: Docker image for vLLM server
- **`checkpoint_path`**: Path to model checkpoints (required if not using Hugging Face)
- **`hf_model_handle`**: Hugging Face model identifier (required if not using local checkpoint, e.g., meta-llama/Llama-3.1-8B-Instruct)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the vLLM server (default: 8000)
- **`tensor_parallel_size`**: Number of GPUs for tensor parallelism (default: 8)
- **`pipeline_parallel_size`**: Number of pipeline parallel stages (default: 1)
- **`data_parallel_size`**: Number of replicas for data parallelism (default: 1)
- **`gpu_memory_utilization`**: Fraction of GPU memory to use for the model (default: 0.95)
- **`extra_args`**: Additional arguments passed to vLLM server
- **`env_vars`**: Environment variables for the container

Tips:
- adjust `tensor_parallel_size`, `data_parallel_size` to your hardware and model size
- adjust `HF_HOME` and mounts to speed up already loaded model loading from Hugging Face
- **Note**: The `${oc.select:deployment.hf_model_handle,/checkpoint}` syntax means:
  - If `hf_model_handle` is provided, use that (downloads from Hugging Face)
  - If `checkpoint_path` is provided instead, use that local path

Examples:
- [Lepton vLLM Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/lepton_vllm_llama_3_1_8b_instruct.yaml) - vLLM deployment on Lepton platform
- [Slurm vLLM Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct.yaml) - vLLM deployment on Slurm cluster
- [Slurm vLLM HF Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct_hf.yaml) - vLLM with Hugging Face model
- [Notebook API Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/notebooks/nemo-evaluator-launcher-api.ipynb) - Python API usage with vLLM


## Reference

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/vllm.yaml)
