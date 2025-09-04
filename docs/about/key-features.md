
(about-key-features)=

# Key Features

NeMo Eval delivers enterprise-grade LLM evaluation capabilities through a unified framework that streamlines deployment, benchmarking, and performance optimization.

## Streamlined Model Deployment

### Multi-Backend Flexibility

Deploy your models with the backend that best fits your use case:

- **PyTriton Backend**: High-performance inference through NVIDIA Triton Inference Server with multi-node model parallelism support for production deployments
- **Ray Serve Backend**: Multi-instance evaluation capabilities with single-node model parallelism and horizontal scaling for accelerated benchmarking

### One-Line Deployment

Get your model running with minimal code:

```python
from nemo_eval.api import deploy

deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="pytriton",
    num_gpus=4,
    tensor_parallelism_size=4
)
```

## Comprehensive Evaluation Ecosystem

### Extensive Benchmark Coverage

Evaluate across multiple dimensions of LLM performance:

- **Academic Benchmarks**: MMLU, ARC Challenge, HellaSwag, TruthfulQA
- **Reasoning Tasks**: GSM8K, Big-Bench Hard, CommonsenseQA
- **Code Generation**: HumanEval, MBPP, APPS
- **Function Calling**: Berkeley Function Calling Leaderboard (BFCL)
- **Safety & Alignment**: Toxicity detection, bias evaluation, vulnerability scanning
- **Multilingual**: Multilingual MMLU, ARC, and reasoning tasks

### Intelligent Task Discovery

Automatically discover and configure available evaluation tasks:

```python
from nemo_eval.utils.base import list_available_evaluations

# Get all available tasks across frameworks
available_tasks = list_available_evaluations()
```

### Framework Conflict Resolution

Intelligent handling of task name conflicts across evaluation frameworks:

## Production-Ready Performance

### Hardware Optimization

- **CUDA Graphs**: Optimized inference execution paths
- **Flash Decoding**: Accelerated attention computation
- **Multi-GPU Scaling**: Tensor and pipeline parallelism support
- **Multi-Node Deployment**: Distributed inference across compute clusters

### Intelligent Health Monitoring

Ensure deployment reliability with built-in health checks:

### Scalable Architecture

- **Multi-Instance Evaluation**: Parallel model replicas for accelerated benchmarking
- **Load Balancing**: Automatic request distribution
- **Resource Management**: Configurable CPU and GPU allocation

## OpenAI API Compatibility

### Standard REST Endpoints

Seamless integration with existing workflows through OpenAI-compatible APIs:

- `/v1/completions/` - Direct text completion
- `/v1/chat/completions/` - Conversational interface

### Flexible Request Processing

Support for diverse evaluation methodologies:

- **Text Generation**: Open-ended response generation
- **Log-Probability**: Confidence assessment across multiple choices
- **Multi-Choice**: Standardized academic benchmark format

## Advanced Adapter System

### Modular Request Processing

Customize evaluation workflows with interceptor-based architecture:

- **SystemMessageInterceptor**: Custom system prompts
- **RequestLoggingInterceptor**: Comprehensive request logging
- **ResponseLoggingInterceptor**: Response analysis and storage
- **ResponseReasoningInterceptor**: Chain-of-thought processing
- **EndpointInterceptor**: Model routing and load balancing

### Reasoning Chain Support

Built-in support for chain-of-thought and reasoning evaluation:

```python
from nvidia_eval_commons.api.api_dataclasses import AdapterConfig

adapter_config = AdapterConfig(
    use_reasoning=True,
    end_reasoning_token="</think>",
    custom_system_prompt="Think step by step."
)
```

## Enterprise Scalability

### Multi-Tenancy Support

- **Model Parallelism**: Efficient resource utilization across GPUs
- **Request Batching**: Optimized throughput management
- **Dynamic Scaling**: Automatic replica adjustment based on load

### Monitoring and Observability

- **Health Endpoints**: Real-time service status monitoring
- **Request Logging**: Comprehensive evaluation tracking
- **Performance Metrics**: Throughput and latency optimization

## Security and Reliability

### Comprehensive Safety Testing

Integrated safety evaluation through specialized harnesses:

- **Vulnerability Scanning**: Prompt injection and jailbreaking detection
- **Bias Assessment**: Demographic fairness evaluation
- **Privacy Protection**: Information leakage prevention
- **Content Filtering**: Toxic content detection

### Production Deployment Features

- **Multi-Node Fault Tolerance**: Distributed deployment resilience
- **Resource Isolation**: Secure multi-tenant evaluation
- **Configuration Validation**: Comprehensive parameter checking

---

*Ready to get started? Explore our [Quick Start Guide](../get-started/quickstart.md) or dive into [Deployment Options](../deployment/index.md).*