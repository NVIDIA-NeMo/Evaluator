# vLLM Deployment

vLLM is a fast and easy-to-use library for LLM inference and serving.

## Configuration

Refer to the [vLLM configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/vllm.yaml) for the complete schema.

## Key Settings

- **`image`**: Docker image for vLLM server
- **`checkpoint_path`**: Path to model checkpoints (required if not using Hugging Face)
- **`hf_model_handle`**: Hugging Face model identifier (required if not using a local checkpoint, for example, `meta-llama/Llama-3.1-8B-Instruct`)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the vLLM server (default: 8000)
- **`tensor_parallel_size`**: Number of GPU devices for tensor parallelism (default: 8)
- **`pipeline_parallel_size`**: Number of pipeline parallel stages (default: 1)
- **`data_parallel_size`**: Number of replicas for data parallelism (default: 1)
- **`extra_args`**: Extra arguments to the vLLM server
- **`env_vars`**: Environment variables for the container

### Tips

Use the following tips:

- Adjust `tensor_parallel_size` and `data_parallel_size` to your hardware and model size.
- Adjust `HF_HOME` and mount points to reuse cached Hugging Face models.
- **Note**: The `${oc.select:deployment.hf_model_handle,/checkpoint}` syntax means:
  - If you set `hf_model_handle`, use that to download from Hugging Face.
  - If you set `checkpoint_path` instead, use that local path.

### Examples

The following examples show common deployments:

- [Lepton vLLM Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/lepton_vllm_llama_3_1_8b_instruct.yaml) — vLLM deployment on the Lepton platform.
- [Slurm vLLM Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct.yaml) — vLLM deployment on a Slurm cluster.
- [Slurm vLLM HF Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct_hf.yaml) — vLLM with a Hugging Face model.
- [Notebook API Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/notebooks/nemo-evaluator-launcher-api.ipynb) — Use the Python API with vLLM.

## References

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/vllm.yaml)
