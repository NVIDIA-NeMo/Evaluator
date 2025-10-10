# Architecture Overview

NeMo Evaluator provides a **two-tier architecture** for comprehensive model evaluation:

```{mermaid}
graph TB
    subgraph Tier2[" Orchestration Layer"]
        Launcher["nemo-evaluator-launcher<br/>• CLI orchestration<br/>• Multi-backend execution (local, Slurm, Lepton)<br/>• Deployment management (vLLM, NIM, SGLang)<br/>• Result export (MLflow, W&B, Google Sheets)"]
    end
    
    subgraph Tier1[" Evaluation Engine"]
        Evaluator["nemo-evaluator<br/>• Adapter system<br/>• Interceptor pipeline<br/>• Containerized evaluation execution<br/>• Result aggregation"]
    end
    
    subgraph External["NVIDIA Eval Factory Containers"]
        Containers["Evaluation Frameworks<br/>• nvidia-lm-eval (lm-evaluation-harness)<br/>• nvidia-simple-evals<br/>• nvidia-bfcl, nvidia-bigcode-eval<br/>• nvidia-eval-factory-garak<br/>• nvidia-safety-harness"]
    end
    
    Launcher --> Evaluator
    Evaluator --> Containers
    
    style Tier2 fill:#e1f5fe
    style Tier1 fill:#f3e5f5
    style External fill:#fff3e0
```

## Component Overview

### **Orchestration Layer** (`nemo-evaluator-launcher`)

High-level orchestration for complete evaluation workflows.

**Key Features:**

- CLI and YAML configuration management
- Multi-backend execution (local, Slurm, Lepton)
- Deployment management (vLLM, NIM, SGLang, or bring-your-own-endpoint)
- Result export to MLflow, Weights & Biases, and Google Sheets
- Job monitoring and lifecycle management

**Use Cases:**

- Automated evaluation pipelines
- HPC cluster evaluations with Slurm
- Cloud deployments with Lepton AI
- Multi-model comparative studies

### **Evaluation Engine** (`nemo-evaluator`)

Core evaluation capabilities with request/response processing.

**Key Features:**

- **Adapter System**: Request/response processing layer for API endpoints
- **Interceptor Pipeline**: Modular components for logging, caching, and reasoning
- **Containerized Execution**: Evaluation harnesses run in Docker containers
- **Result Aggregation**: Standardized result schemas and metrics

**Use Cases:**

- Programmatic evaluation integration
- Request/response transformation and logging
- Custom interceptor development
- Direct Python API usage

## Interceptor Pipeline

The evaluation engine provides an interceptor system for request/response processing. Interceptors are configurable components that process API requests and responses in a pipeline.

```{mermaid}
graph LR
    A[Request] --> B[System Message]
    B --> C[Payload Modifier]
    C --> D[Request Logging]
    D --> E[Caching]
    E --> F[API Endpoint]
    F --> G[Response Logging]
    G --> H[Reasoning]
    H --> I[Response Stats]
    I --> J[Response]
    
    style E fill:#e1f5fe
    style F fill:#f3e5f5
```

**Available Interceptors:**

- **System Message**: Inject system prompts into chat requests
- **Payload Modifier**: Transform request parameters
- **Request/Response Logging**: Log requests and responses to files
- **Caching**: Cache responses to avoid redundant API calls
- **Reasoning**: Extract chain-of-thought from responses
- **Response Stats**: Track token usage and latency metrics
- **Progress Tracking**: Monitor evaluation progress

## Integration Patterns

### **Pattern 1: Launcher with Deployment**

Use the launcher to handle both model deployment and evaluation:

```bash
nemo-evaluator-launcher run \
  --config-dir examples \
  --config-name local_llama_3_1_8b_instruct \
  -o deployment.checkpoint_path=/path/to/model \
  -o 'evaluation.tasks=["mmlu_pro", "gsm8k"]'
```

### **Pattern 2: Launcher with Existing Endpoint**

Point the launcher to an existing API endpoint:

```bash
nemo-evaluator-launcher run \
  --config-dir examples \
  --config-name local_llama_3_1_8b_instruct \
  -o target.api_endpoint.url=http://localhost:8080/v1/completions \
  -o deployment.type=none
```

### **Pattern 3: Python API**

Use the Python API for programmatic integration:

```python
from nemo_evaluator import evaluate, EvaluationConfig, EvaluationTarget, ApiEndpoint, EndpointType

# Configure target endpoint
api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions",
    type=EndpointType.COMPLETIONS
)
target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results"
)

# Run evaluation
results = evaluate(eval_cfg=eval_config, target_cfg=target)
```
