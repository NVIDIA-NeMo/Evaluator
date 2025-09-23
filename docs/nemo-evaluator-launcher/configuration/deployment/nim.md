# NIM Deployment

NIM (NVIDIA Inference Microservices) provides optimized inference microservices with an OpenAI-compatible API.

## Configuration

Refer to the [NIM configuration file](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/nim.yaml) for the complete configuration structure.

## Key Settings

Configure the following fields:

- **`image`**: NIM container image (required)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the NIM server (default: `8000`)

### Tips

- Choose the appropriate NIM image for your model from [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim).
- You do not need to adjust parameters such as tensor or data parallelism. NIM selects an optimal configuration based on your hardware.

### Example

Refer to the [Lepton NIM example](https://github.com/NVIDIA-NeMo/Eval/tree/main/packages/nemo-evaluator-launcher/examples/lepton_nim_llama_3_1_8b_instruct.yaml)—NIM deployment on the Lepton platform.

## References

- [NIM Documentation](https://docs.nvidia.com/nim/)
- [NIM Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html)
- [NIM Configuration File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/nim.yaml)
