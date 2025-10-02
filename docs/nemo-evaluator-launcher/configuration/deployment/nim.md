# NIM Deployment

NIM (NVIDIA Inference Microservices) provides optimized inference microservices with OpenAI-compatible APIs.

## Configuration

See the complete configuration structure in the [NIM Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/nim.yaml).

## Key Settings

- **`image`**: NIM container image (required)
- **`served_model_name`**: Name used for serving the model (required)
- **`port`**: Port for the NIM server (default: 8000)

Tips:
- Choose the appropriate NIM image for your model from [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim)
- You do  not need to adjust params like tensor/data parallelism NIM should pick the best set up based on your hardware

Examples:
- [Lepton NIM Example](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/lepton_nim_llama_3_1_8b_instruct.yaml) - NIM deployment on Lepton platform

## Reference

- [NIM Documentation](https://docs.nvidia.com/nim/)
- [NIM Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html#)
- [NIM Config File](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/nim.yaml)
