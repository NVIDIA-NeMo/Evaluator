
(about-key-features)=

# Key Features

NeMo Evaluator delivers comprehensive AI model evaluation through a dual-library architecture that scales from local development to enterprise production. Experience container-first reproducibility, multi-backend execution, and 100+ benchmarks across 18 evaluation harnesses.

##  **Unified Orchestration (NeMo Evaluator Launcher)**

### Multi-Backend Execution
Run evaluations anywhere with unified configuration and monitoring:

- **Local Execution**: Docker-based evaluation on your workstation
- **HPC Clusters**: Slurm integration for large-scale parallel evaluation  
- **Cloud Platforms**: Lepton AI and custom cloud backend support
- **Hybrid Workflows**: Mix local development with cloud production

```bash
# Single command, multiple backends
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
nv-eval run --config-dir examples --config-name slurm_llama_3_1_8b_instruct  
nv-eval run --config-dir examples --config-name lepton_vllm_llama_3_1_8b_instruct
```

### 100+ Benchmarks Across 18 Harnesses
Access comprehensive benchmark suite with single CLI:

```bash
# Discover available benchmarks
nv-eval ls tasks

# Run academic benchmarks
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o evaluation.tasks='["mmlu_pro", "gsm8k", "arc_challenge"]'

# Run safety evaluation
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o evaluation.tasks='["toxicity", "bias_detection", "jailbreak_resistance"]'
```

### Built-in Result Export
First-class integration with MLOps platforms:

```bash
# Export to MLflow
nv-eval export <invocation_id> --dest mlflow

# Export to Weights & Biases
nv-eval export <invocation_id> --dest wandb

# Export to Google Sheets
nv-eval export <invocation_id> --dest gsheets
```

##  **Core Evaluation Engine (NeMo Evaluator Core)**

### Container-First Architecture
Pre-built NGC containers guarantee reproducible results across environments:

| Container | Benchmarks | Use Case |
|-----------|------------|----------|
| **simple-evals** | MMLU Pro, GSM8K, ARC | Academic benchmarks |
| **lm-evaluation-harness** | HellaSwag, TruthfulQA, PIQA | Language model evaluation |
| **bigcode-evaluation-harness** | HumanEval, MBPP, APPS | Code generation |
| **safety-harness** | Toxicity, bias, jailbreaking | Safety assessment |
| **vlmevalkit** | VQA, image captioning | Vision-language models |
| **agentic_eval** | Tool usage, planning | Agentic AI evaluation |

```bash
# Pull and run any evaluation container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
```

### Advanced Adapter System
Sophisticated request/response processing pipeline with interceptor architecture:

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    # Core endpoint configuration
    api_url="http://localhost:8080/v1/completions/",
    
    # Logging interceptors
    use_request_logging=True,      # Log all requests for debugging
    use_response_logging=True,     # Log all responses for analysis
    max_logged_requests=1000,
    
    # Caching interceptor
    use_caching=True,              # Cache responses for performance
    caching_dir="./evaluation_cache",
    reuse_cached_responses=True,
    
    # Reasoning interceptor
    use_reasoning=True,            # Chain-of-thought support
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    
    # System message interceptor
    use_system_prompt=True,
    custom_system_prompt="You are a helpful AI assistant. Think step by step.",
    
    # Progress tracking
    use_progress_tracking=True
)
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
Direct access to specialized evaluation containers:

```bash
# Academic benchmarks
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3

# Code generation evaluation  
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:25.07.3

# Safety and security testing
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/safety-harness:25.07.3

# Vision-language model evaluation
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/vlmevalkit:25.07.1
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
# Evaluation configuration with overrides
evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 1000
    - name: gsm8k
      params:
        temperature: 0.0

execution:
  backend: slurm
  resources:
    nodes: 4
    gpus_per_node: 8

target:
  api_endpoint:
    url: https://my-model-endpoint.com/v1/chat/completions
    model_id: my-custom-model
```

##  **OpenAI API Compatibility**

### Universal Model Support
Evaluate any model that exposes OpenAI-compatible endpoints:

- **Hosted Models**: NVIDIA Build, OpenAI, Anthropic, Cohere
- **Self-Hosted**: vLLM, TRT-LLM, NeMo Framework, Ray Serve
- **Custom Endpoints**: Any service implementing OpenAI API spec

### Flexible Evaluation Modes
Support for diverse evaluation methodologies:

```bash
# Text generation evaluation
eval-factory run_eval --model_type chat --eval_type mmlu_pro

# Log-probability evaluation  
eval-factory run_eval --model_type completions --eval_type multiple_choice

# Vision-language evaluation
eval-factory run_eval --model_type vlm --eval_type vqa_benchmark
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
Fine-tune request/response processing with the adapter system:

```python
# Production-ready adapter configuration
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    # Endpoint configuration
    api_url="https://production-api.com/v1/completions",
    max_retries=3,
    retry_delay=2.0,
    timeout=30.0,
    
    # System message interceptor
    use_system_prompt=True,
    custom_system_prompt="You are an expert AI assistant specialized in this domain.",
    
    # Comprehensive logging
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=5000,
    log_failed_requests=True,
    
    # Performance optimization
    use_caching=True,
    caching_dir="./production_cache",
    reuse_cached_responses=True,
    
    # Advanced reasoning
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    extract_reasoning=True,
    
    # Monitoring
    use_progress_tracking=True,
    progress_tracking_url="http://monitoring.internal:3828/progress"
)
```

##  **Security and Safety**

### Comprehensive Safety Evaluation
Built-in safety assessment through specialized containers:

```bash
# Run safety evaluation suite
nv-eval run \
    --config-dir examples \
    --config-name comprehensive_safety \
    -o evaluation.tasks='["toxicity", "bias_detection", "jailbreak_resistance", "privacy_leakage"]'
```

**Safety Containers Available:**
- **safety-harness**: Toxicity, bias, and alignment testing
- **garak**: Security vulnerability scanning and prompt injection detection  
- **agentic_eval**: Tool misuse and planning safety evaluation

### Production Security Features
- **Request Validation**: Input sanitization and validation
- **Resource Isolation**: Containerized execution environments
- **Audit Logging**: Complete evaluation traceability
- **Access Control**: Role-based permissions and API key management

##  **Monitoring and Observability**

### Real-Time Progress Tracking
Monitor evaluation progress across all backends:

```bash
# Check evaluation status
nv-eval status <invocation_id>

# View live logs
nv-eval logs <invocation_id> --follow
```

### Comprehensive Result Analytics
Built-in analysis and visualization capabilities:
- **Performance Metrics**: Accuracy, latency, throughput analysis
- **Comparative Analysis**: Multi-model benchmarking
- **Trend Analysis**: Performance over time tracking
- **Export Integration**: Seamless data pipeline integration
