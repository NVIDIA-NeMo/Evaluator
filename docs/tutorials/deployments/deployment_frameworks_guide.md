(deployment-frameworks-guide)=

# Deploying OpenAI-Compatible Endpoints


## Evaluation Options

You can either:
1. **Deploy + Evaluate**: Use our default option to deploy and evaluate models locally
   - Good for: Quick testing, full control over deployment environment (endpoint lifecycle)
2. **Deploy on your own**: Deploy using any framework below, then evaluate
   - Good for: Existing deployments, third-party services, separate evaluation/deployment environments

:::{note}
Models deployed with the frameworks listed below should work with nemo_evaluator_launcher.
:::

**Tutorials:**
- {ref}`tutorials-overview`

## Quick Setup Options

### vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving..

**Docker Setup:**
```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
    --model microsoft/Phi-4-mini-instruct
```

**Documentation:** 
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM Docker Deployment](https://docs.vllm.ai/en/stable/deployment/docker.html)

### SGLang

SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language. The core features include:

**Documentation:** 
- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang Docker Deployment](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)

### NeMo

NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. The Export-Deploy library ("NeMo Export-Deploy") provides tools and APIs for exporting and deploying NeMo and 🤗Hugging Face models to production environments. It supports various deployment paths including TensorRT, TensorRT-LLM, and vLLM deployment through NVIDIA Triton Inference Server.

**Documentation:** 
- [NVIDIA NeMo](https://github.com/NVIDIA-NeMo)
- [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)
- [NeMo Export-Deploy Scripts](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts)

### TRT-LLM

TRT-LLM provides optimized inference with OpenAI-compatible server through the `trtllm-serve` command.

**Documentation:** 
- [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
- [TRT-LLM Server](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html)

### NIM (NVIDIA Inference Microservices)

NIM provides optimized inference microservices with OpenAI-compatible APIs.

**Documentation:** 
- [NIM Official Docs](https://docs.nvidia.com/nim/)
- [NIM Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html#)


**Next Steps:**
- {ref}`tutorials-overview` - Learn how to run evaluations
- {ref}`testing-endpoint-compatibility` - Test your deployed endpoint with curl requests
