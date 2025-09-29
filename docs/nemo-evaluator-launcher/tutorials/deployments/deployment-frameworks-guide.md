# Deploy OpenAI-Compatible Endpoints

<!-- cSpell:ignore SGLang sglang vLLM NeMo TensorRT trtllm NIM Triton API APIs -->
<!-- cspell:words SGLang sglang vLLM NeMo TensorRT trtllm NIM Triton -->

## Evaluation Options

You can do one of the following:

- **Deploy and run evaluations**: Use the default option to deploy and run evaluations locally.
  - Good for quick testing and full control of the deployment environment (endpoint lifecycle)
- **Deploy separately**: Deploy by using any framework below, then run evaluations.
  - Good for existing deployments, third-party services, or separate evaluation and deployment environments

/// note | Framework Compatibility
Models deployed with the frameworks listed below work with `nemo_evaluator_launcher`.
///

## Tutorials

- [Run Evaluations on an Existing Local Endpoint](../local-evaluation-of-existing-endpoint.md)

## Quick Setup Options

### vLLM

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
    --model microsoft/Phi-4-mini-instruct
```

vLLM is a fast, easy-to-use library for LLM inference and serving.

#### vLLM References

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [Deploy vLLM with Docker](https://docs.vllm.ai/en/stable/deployment/docker.html)

<!-- vale off -->
### SGLang

SGLang is a fast serving framework for large language models and vision-language models.

#### SGLang References

- [SGLang Documentation](https://docs.sglang.ai/)
- [Deploy SGLang with Docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
<!-- vale on -->

### NeMo

The NVIDIA NeMo framework provides GPU-accelerated, end-to-end training for large language models, multi-modal models, and speech models. The Export-Deploy library provides tools and API for exporting and deploying NeMo and Hugging Face models to production environments. It supports several deployment paths, including TensorRT, TensorRT-LLM, and vLLM deployment by using NVIDIA Triton Inference Server.

#### NeMo References

- [NVIDIA NeMo GitHub Repository](https://github.com/NVIDIA-NeMo)
- [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)
- [NeMo Export-Deploy Scripts](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts)

### TensorRT-LLM (TRT-LLM)

TensorRT-LLM provides optimized inference with an OpenAI-compatible server by using the `trtllm-serve` command.

#### TensorRT-LLM References

- [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
- [trtllm-serve Command](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html)

### NVIDIA Inference Microservices (NIM)

NVIDIA NIM provides optimized inference microservices with an OpenAI-compatible API.

#### NIM References

- [NIM Documentation](https://docs.nvidia.com/nim/)
- [NIM Deployment Guide for Large Language Models](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html#)

## Next Steps

- [Run Evaluations on an Existing Local Endpoint](../local-evaluation-of-existing-endpoint.md): Learn how to run evaluations.
- [Test Endpoint Compatibility](testing-endpoint-oai-compatibility.md): Verify your deployed endpoint by using `curl` requests.
