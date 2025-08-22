# About NeMo Eval

## What is NeMo Eval?

NeMo Eval is NVIDIA's comprehensive evaluation framework for Large Language Models (LLMs), designed to streamline the deployment and evaluation of AI models trained with the NeMo Framework. It bridges the gap between model training and real-world performance assessment through production-ready deployment capabilities and state-of-the-art evaluation harnesses.

## Mission & Purpose

Our mission is to democratize LLM evaluation by providing researchers, developers, and organizations with:

- **Simplified Deployment**: Deploy models effortlessly across different serving backends
- **Comprehensive Benchmarking**: Access to cutting-edge evaluation harnesses for academic, reasoning, coding, and safety assessments
- **Production Readiness**: High-performance inference optimizations for real-world deployment scenarios
- **Open Standards**: OpenAI-compatible APIs for seamless integration with existing workflows

## Key Capabilities

### 🚀 **Multi-Backend Deployment**

- **PyTriton Backend**: High-performance inference through NVIDIA Triton Inference Server with multi-node model parallelism support
- **Ray Serve Backend**: Multi-instance evaluation capabilities with single-node model parallelism and horizontal scaling

### 📊 **Evaluation Ecosystem**

Integration with leading evaluation frameworks:

- **lm-evaluation-harness**: Academic benchmarks and reasoning tasks
- **simple-evals**: Streamlined evaluation workflows
- **BigCode**: Code generation and programming assessment
- **BFCL**: Berkeley Function Calling Leaderboard
- **safety-harness**: AI safety and alignment testing
- **garak**: LLM vulnerability scanning

### 🔧 **Production Features**

- CUDA graphs and flash decoding for optimized inference
- Multi-GPU and multi-node distributed computing
- OpenAI-compatible REST API endpoints
- Flexible adapter system with interceptor pipelines
- Real-time health monitoring and endpoint validation

## Technology Stack

- **Core Framework**: Built on PyTorch and Megatron-Core
- **Inference Backends**: NVIDIA Triton, Ray Serve, vLLM
- **API Layer**: FastAPI with OpenAI compatibility
- **Evaluation**: NVIDIA Eval Factory integration
- **Deployment**: Docker containers and distributed computing support

## Who Should Use NeMo Eval?

### 🎓 **Researchers**

Benchmark your models against academic standards and cutting-edge evaluation tasks with minimal setup overhead.

### 👩‍💻 **ML Engineers**

Deploy and assess models in production environments with enterprise-grade performance and reliability.

### 🏢 **Organizations**

Scale LLM evaluation across teams with standardized benchmarks and reproducible results.

### 🧪 **AI Safety Teams**

Conduct comprehensive safety assessments using integrated vulnerability scanning and alignment testing tools.

## Project Governance

**Maintainer**: NVIDIA Corporation  
**License**: Apache License 2.0  
**Contact**: [nemo-toolkit@nvidia.com](mailto:nemo-toolkit@nvidia.com)  
**Repository**: [GitHub - NVIDIA-NeMo/Eval](https://github.com/NVIDIA-NeMo/Eval)

## Community & Support

- **Documentation**: [docs.nvidia.com/nemo/eval](https://docs.nvidia.com/nemo/eval/latest/index.html)
- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)
- **Contributing**: See our [Contributing Guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md)

---

*NeMo Eval is part of the broader [NeMo Framework](https://github.com/NVIDIA-NeMo/) ecosystem, enabling end-to-end LLM development from training to deployment and evaluation.*
