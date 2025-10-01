(bring-your-own-endpoint)=

# Bring-Your-Own-Endpoint

Deploy and manage model serving yourself, then point NeMo Evaluator to your endpoint. This approach gives you full control over deployment infrastructure while still leveraging NeMo Evaluator's evaluation capabilities.

## Overview

With bring-your-own-endpoint, you:
- Handle model deployment and serving independently
- Provide an OpenAI-compatible API endpoint
- Use either the launcher or core library for evaluations
- Maintain full control over infrastructure and scaling

## When to Use This Approach

**Choose bring-your-own-endpoint when you:**
- Have existing model serving infrastructure
- Need custom deployment configurations
- Want to deploy once and run many evaluations
- Have specific security or compliance requirements
- Use enterprise Kubernetes or MLOps pipelines

## Deployment Approaches

Choose the approach that best fits your infrastructure and requirements:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Manual Deployment
:link: manual-deployment
:link-type: doc
Deploy using vLLM, TensorRT-LLM, or custom serving frameworks for full control.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Hosted Services
:link: hosted-services
:link-type: doc
Use NVIDIA Build, OpenAI API, or other cloud providers for instant availability.
:::

::::

## Quick Examples

### Using Launcher with Existing Endpoint

```bash
# Point launcher to your deployed model
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://your-endpoint:8080/v1/completions \
    -o target.api_endpoint.model_id=your-model-name \
    -o deployment.type=none  # No launcher deployment
```

### Using Core Library

```python
from nemo_evaluator import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, evaluate
)

# Configure your endpoint
api_endpoint = ApiEndpoint(
    url="http://your-endpoint:8080/v1/completions",
    model_id="your-model-name"
)
target = EvaluationTarget(api_endpoint=api_endpoint)

# Run evaluation
config = EvaluationConfig(type="mmlu_pro", output_dir="results")
results = evaluate(eval_cfg=config, target_cfg=target)
```

## Endpoint Requirements

Your endpoint must provide OpenAI-compatible APIs:

### Required Endpoints
- **Completions**: `/v1/completions` (POST) - For text completion tasks
- **Chat Completions**: `/v1/chat/completions` (POST) - For conversational tasks
- **Health Check**: `/health` (GET) - For monitoring (recommended)

### Request/Response Format
Must follow OpenAI API specifications for compatibility with evaluation frameworks.

## Popular Integration Patterns

### Hosted Model APIs
- **NVIDIA Build**: Ready-to-use models with enterprise features
- **OpenAI API**: Direct access to GPT models for comparison
- **Cloud Providers**: Azure OpenAI, AWS Bedrock, Google Vertex AI

### Self-Hosted Frameworks
- **vLLM**: Fast inference with PagedAttention and optimized serving
- **TensorRT-LLM**: High-performance serving with NVIDIA TensorRT
- **Ray Serve**: Multi-instance serving with horizontal scaling
- **Custom FastAPI**: Build your own OpenAI-compatible server

### Enterprise Integration
- **Kubernetes**: Container orchestration for production deployments
- **MLOps Pipelines**: Integration with MLflow, Kubeflow, Airflow
- **Security & Compliance**: OAuth, LDAP, audit logging, DLP

## Configuration Management

### Basic Configuration

```yaml
# config/bring_your_own.yaml
deployment:
  type: none  # No launcher deployment

target:
  api_endpoint:
    url: http://your-endpoint:8080/v1/completions
    model_id: your-model-name
    api_key: ${API_KEY}  # Optional

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
```

### With Adapters

```yaml
target:
  api_endpoint:
    url: http://your-endpoint:8080/v1/completions
    model_id: your-model-name
    
    adapter_config:
      # Caching for efficiency
      use_caching: true
      caching_dir: ./cache
      
      # Request logging for debugging
      use_request_logging: true
      max_logged_requests: 10
      
      # Custom processing
      use_reasoning: true
      start_reasoning_token: "<think>"
      end_reasoning_token: "</think>"
```

## Key Benefits

### Infrastructure Control
- **Custom configurations**: Tailor deployment to your specific needs
- **Resource optimization**: Optimize for your hardware and workloads
- **Security compliance**: Meet your organization's security requirements
- **Cost management**: Control costs through efficient resource usage

### Operational Flexibility
- **Deploy once, evaluate many**: Reuse deployments across multiple evaluations
- **Integration ready**: Works with existing infrastructure and workflows
- **Technology choice**: Use any serving framework or cloud provider
- **Scaling control**: Scale according to your requirements

## Getting Started

1. **Choose your approach**: Select from manual deployment, hosted services, or enterprise integration
2. **Deploy your model**: Set up your OpenAI-compatible endpoint
3. **Configure NeMo Evaluator**: Point to your endpoint with proper configuration
4. **Run evaluations**: Use launcher or core library to run benchmarks
5. **Monitor and optimize**: Track performance and optimize as needed

## Next Steps

- **Manual Deployment**: Learn [Manual Deployment](manual-deployment.md) techniques
- **Hosted Services**: Explore [Hosted Services](hosted-services.md) options
- **Configure Adapters**: Set up [Evaluation Adapters](../adapters/index.md) for custom processing

```{toctree}
:maxdepth: 1
:hidden:

Manual Deployment <manual-deployment>
Hosted Services <hosted-services>
```
