# Architecture Overview

NeMo Eval provides a **three-tier architecture** for comprehensive model evaluation, from deployment to orchestration:

```{mermaid}
graph TB
    subgraph Tier3[" Tier 3: Orchestration Layer"]
        Launcher["NeMo Evaluator Launcher<br/>• CLI & API orchestration<br/>• Multi-backend execution<br/>• Result export & monitoring"]
    end
    
    subgraph Tier2[" Tier 2: Evaluation Engine"]
        Evaluator["NeMo Evaluator Core<br/>• Adapter system<br/>• Interceptor pipeline<br/>• Advanced evaluation logic"]
    end
    
    subgraph Tier1[" Tier 1: Model Deployment"]
        Deploy["NeMo Eval Deploy<br/>• PyTriton & Ray backends<br/>• Model serving<br/>• OpenAI-compatible APIs"]
    end
    
    subgraph External["External Frameworks"]
        EvalHarnesses["Evaluation Harnesses<br/>• lm-evaluation-harness<br/>• simple-evals<br/>• BFCL, BigCode<br/>• garak, safety-harness"]
        
        EvalFactory["NVIDIA Eval Factory<br/>• Unified interface<br/>• Containerized benchmarks<br/>• Framework packaging"]
    end
    
    Launcher --> Evaluator
    Evaluator --> Deploy
    Evaluator --> EvalFactory
    EvalHarnesses --> EvalFactory
    
    style Tier3 fill:#e1f5fe
    style Tier2 fill:#f3e5f5  
    style Tier1 fill:#e8f5e8
    style External fill:#fff3e0
```

## Component Overview

### **Tier 1: Model Deployment** (`nemo_eval`)

Foundation layer that handles model serving and API endpoints.

**Key Features:**

- PyTriton and Ray Serve backends
- Multi-GPU and multi-node support
- OpenAI-compatible REST APIs
- High-performance inference with CUDA graphs

**Use Cases:**

- Deploying NeMo checkpoints for evaluation
- Setting up model serving infrastructure
- Creating evaluation-ready endpoints

### **Tier 2: Evaluation Engine** (`nemo-evaluator`)

Core evaluation capabilities with advanced request/response processing.

**Key Features:**

- **Adapter System**: Configurable request/response processing
- **Interceptor Pipeline**: Modular components for logging, caching, reasoning
- **Advanced Evaluation Logic**: Containerized benchmark execution
- **Plugin Architecture**: Extensible framework for custom components

**Use Cases:**

- Advanced evaluation workflows with custom processing
- Request/response transformation and logging
- Integration into existing ML pipelines
- Custom benchmark development

### **Tier 3: Orchestration Layer** (`nemo-evaluator-launcher`)

High-level orchestration for complete evaluation workflows.

**Key Features:**

- CLI and configuration management
- Multi-backend execution (local, Slurm, cloud)
- Result aggregation and export
- Workflow monitoring and management

**Use Cases:**

- Production evaluation pipelines
- Research experiments with reproducible configurations
- Multi-model comparative studies
- Automated evaluation workflows

## Detailed Component Architecture

### **Interceptor Pipeline** (New in nemo-evaluator)

The evaluation engine includes a sophisticated interceptor system for request/response processing:

```{mermaid}
graph LR
    A[Request] --> B[System Message]
    B --> C[Payload Modifier]
    C --> D[Request Logging]
    D --> E[Caching Check]
    E --> F[Endpoint]
    F --> G[Response Logging]
    G --> H[Reasoning Processing]
    H --> I[Progress Tracking]
    I --> J[Response]
    
    style E fill:#e1f5fe
    style F fill:#f3e5f5
```

**Available Interceptors:**

- **System Message**: Inject custom system prompts
- **Payload Modifier**: Transform request parameters
- **Request/Response Logging**: Structured logging with configurable storage
- **Caching**: Response caching with configurable backends
- **Reasoning**: Chain-of-thought processing and extraction
- **Progress Tracking**: Real-time evaluation monitoring

### **Integration Patterns**

#### **Pattern 1: Complete Stack** (Recommended)

```python
# 1. Deploy model
from nemo_eval.api import deploy
deploy(nemo_checkpoint="/path/to/checkpoint", serving_backend="pytriton")

# 2. Configure evaluation with adapters
from nemo_evaluator.adapters.adapter_config import AdapterConfig
adapter_config = AdapterConfig(
    use_reasoning=True,
    use_caching=True,
    custom_system_prompt="You are a helpful assistant."
)

# 3. Run evaluation
from nemo_evaluator.core.evaluate import evaluate
results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_config)
```

#### **Pattern 2: Orchestrated Workflow**

```bash
# Single command handles deployment, evaluation, and export
nv-eval run \
  --config-dir examples \
  --config-name local_llama_3_1_8b_instruct \
  -o target.api_endpoint.adapter_config.use_reasoning=true
```

#### **Pattern 3: Programmatic Integration**

```python
# Direct integration into ML pipelines
from nemo_evaluator.api import run
results = run(
    target_cfg=target_config,
    eval_cfg=evaluation_config,
    adapter_cfg=adapter_config
)
```
