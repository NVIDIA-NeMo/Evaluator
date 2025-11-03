
(about-key-features)=

# Key Features

NeMo Evaluator SDK delivers comprehensive AI model evaluation through a dual-library architecture that scales from local development to enterprise production. Experience container-first reproducibility, multi-backend execution, and comprehensive set of state-of-the-art benchmarks.

##  **Unified Orchestration (NeMo Evaluator Launcher)**

### Multi-Backend Execution
Run evaluations anywhere with unified configuration and monitoring:

- **Local Execution**: Docker-based evaluation on your workstation
- **HPC Clusters**: Slurm integration for large-scale parallel evaluation
- **Cloud Platforms**: Lepton AI and custom cloud backend support
- **Hybrid Workflows**: Mix local development with cloud production

```bash
# Single command, multiple backends
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name slurm_llama_3_1_8b_instruct
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name lepton_vllm_llama_3_1_8b_instruct
```

### Evaluation Benchmarks & Tasks
Access comprehensive benchmark suite with single CLI:

```bash
# Discover available benchmarks
nemo-evaluator-launcher ls tasks

# Run academic benchmarks
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o 'evaluation.tasks=["mmlu_pro", "gsm8k", "arc_challenge"]'

# Run safety evaluation
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o 'evaluation.tasks=["aegis_v2", "garak"]'
```

### Built-in Result Export
First-class integration with MLOps platforms:

```bash
# Export to MLflow
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to Google Sheets
nemo-evaluator-launcher export <invocation_id> --dest gsheets
```

##  **Core Evaluation Engine (NeMo Evaluator Core)**

### Container-First Architecture
Pre-built NGC containers guarantee reproducible results across environments:

```{include} ../_resources/tasks-table.md
```

```bash
# Pull and run any evaluation container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
```

### Advanced Adapter System
Sophisticated request/response processing pipeline with interceptor architecture:

```yaml
# Configure adapter system in YAML configuration
target:
  api_endpoint:
    url: "http://localhost:8080/v1/completions/"
    model_id: "my-model"
    adapter_config:
      interceptors:
        # System message interceptor
        - name: system_message
          config:
            system_message: "You are a helpful AI assistant. Think step by step."

        # Request logging interceptor
        - name: request_logging
          config:
            max_requests: 1000

        # Caching interceptor
        - name: caching
          config:
            cache_dir: "./evaluation_cache"

        # Communication with http://localhost:8080/v1/completions/
        -name: endpoint

        # Reasoning interceptor
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"

        # Response logging interceptor
        - name: response_logging
          config:
            max_responses: 1000

        # Progress tracking interceptor
        - name: progress_tracking
```

### Programmatic API
Full Python API for integration into ML pipelines:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, EvaluationTarget

# Configure and run evaluation programmatically
result = evaluate(
    eval_cfg=EvaluationConfig(type="mmlu_pro", output_dir="./results"),
    target_cfg=EvaluationTarget(api_endpoint=endpoint_config)
)
```

##  **Container Direct Access**

### NGC Container Catalog
Direct access to specialized evaluation containers through [NGC Catalog](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=label%3A%22NSPECT-JL1B-TVGU%22):

```bash
# Academic benchmarks
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}

# Code generation evaluation
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}

# Safety and security testing
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }}

# Vision-language model evaluation
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/vlmevalkit:{{ docker_compose_latest }}
```

### Reproducible Evaluation Environments
Every container provides:
- **Fixed dependencies**: Locked versions for consistent results
- **Pre-configured frameworks**: Ready-to-run evaluation harnesses
- **Isolated execution**: No dependency conflicts between evaluations
- **Version tracking**: Tagged releases for exact reproducibility

##  **Enterprise Features**

### Multi-Backend Scalability
Scale from laptop to datacenter with unified configuration:

- **Local Development**: Quick iteration with Docker
- **HPC Clusters**: Slurm integration for large-scale evaluation
- **Cloud Platforms**: Lepton AI and custom backend support
- **Hybrid Workflows**: Seamless transition between environments

### Advanced Configuration Management
Hydra-based configuration with full reproducibility:

```yaml
# Evaluation configuration with custom parameters
evaluation:
  tasks:
    - name: mmlu_pro
      nemo_evaluator_config:
        config:
          params:
            limit_samples: 1000
    - name: gsm8k
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.0

execution:
  output_dir: results

target:
  api_endpoint:
    url: https://my-model-endpoint.com/v1/chat/completions
    model_id: my-custom-model
```

##  **OpenAI API Compatibility**

### Universal Model Support
NeMo Evaluator supports OpenAI-compatible API endpoints:

- **Hosted Models**: NVIDIA Build, OpenAI, Anthropic, Cohere
- **Self-Hosted**: vLLM, TRT-LLM, NeMo Framework
- **Custom Endpoints**: Any service implementing OpenAI API spec (test compatibility with our {ref}`deployment-testing-compatibility` guide)

The platform supports the following endpoint types:

- **`completions`**: Direct text completion without chat formatting (`/v1/completions`). Used for base models and academic benchmarks.
- **`chat`**: Conversational interface with role-based messages (`/v1/chat/completions`). Used for instruction-tuned and chat models.
- **`vlm`**: Vision-language model endpoints supporting image inputs.
- **`embedding`**: Embedding generation endpoints for retrieval evaluation.

### Endpoint Type Support
Support for diverse evaluation endpoint types through the evaluation configuration:

```yaml
# Text generation evaluation (chat endpoint)
target:
  api_endpoint:
    type: chat
    url: https://api.example.com/v1/chat/completions

# Log-probability evaluation (completions endpoint)
target:
  api_endpoint:
    type: completions
    url: https://api.example.com/v1/completions

# Vision-language evaluation (vlm endpoint)
target:
  api_endpoint:
    type: vlm
    url: https://api.example.com/v1/chat/completions

# Retrival evaluation (embedding endpoint)
target:
  api_endpoint:
    type: embedding
    url: https://api.example.com/v1/embeddings

```

##  **Extensibility and Customization**

### Custom Framework Support
Add your own evaluation frameworks using Framework Definition Files:

```yaml
# custom_framework.yml
framework:
  name: my_custom_eval
  description: Custom evaluation for domain-specific tasks

defaults:
  command: >-
    python custom_eval.py --model {{target.api_endpoint.model_id}}
    --task {{config.params.task}} --output {{config.output_dir}}

evaluations:
  - name: domain_specific_task
    description: Evaluate domain-specific capabilities
    defaults:
      config:
        params:
          task: domain_task
          temperature: 0.0
```

### Advanced Interceptor Configuration
Fine-tune request/response processing with the adapter system through YAML configuration:

```yaml
# Production-ready adapter configuration in framework YAML
target:
  api_endpoint:
    url: "https://production-api.com/v1/completions"
    model_id: "production-model"
    adapter_config:
      log_failed_requests: true
      interceptors:
        # System message interceptor
        - name: system_message
          config:
            system_message: "You are an expert AI assistant specialized in this domain."

        # Request logging interceptor
        - name: request_logging
          config:
            max_requests: 5000

        # Caching interceptor
        - name: caching
          config:
            cache_dir: "./production_cache"

        # Reasoning interceptor
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"

        # Response logging interceptor
        - name: response_logging
          config:
            max_responses: 5000

        # Progress tracking interceptor
        - name: progress_tracking
          config:
            progress_tracking_url: "http://monitoring.internal:3828/progress"
```

##  **Security and Safety**

### Comprehensive Safety Evaluation
Built-in safety assessment through specialized containers:

```bash
# Run safety evaluation suite
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["aegis_v2", "garak"]'
```

**Safety Containers Available:**
- **safety-harness**: Content safety evaluation using NemoGuard judge models
- **garak**: Security vulnerability scanning and prompt injection detection
- **agentic_eval**: Tool usage and planning evaluation for agentic AI systems

##  **Monitoring and Observability**

### Real-Time Progress Tracking
Monitor evaluation progress across all backends:

```bash
# Check evaluation status
nemo-evaluator-launcher status <invocation_id>

# Kill running evaluations
nemo-evaluator-launcher kill <invocation_id>
```

### Result Export and Analysis
Export evaluation results to MLOps platforms for downstream analysis:

```bash
# Export to MLflow for experiment tracking
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases for visualization
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to Google Sheets for sharing
nemo-evaluator-launcher export <invocation_id> --dest gsheets
```
