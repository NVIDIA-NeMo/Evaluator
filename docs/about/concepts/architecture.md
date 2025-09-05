# Architecture Overview

The following diagram illustrates the architecture of the Eval ecosystem, showing how different evaluation harnesses integrate with NeMo Eval:

```{mermaid}
graph LR
    subgraph EvalHarnesses["Evaluation Harnesses"]
        LMEval["lm-evaluation-harness"]
        SimpleEvals["simple-evals"]
        Garak["garak"]
        BFCL["BFCL"]
        BigCode["BigCode"]
        SafetyHarness["safety-harness"]
        Others["..."]
    end
    
    subgraph EvalFactory["NVIDIA Eval Factory"]
        Factory["unified interface<br/>for LLM evaluation"]
    end
    
    subgraph NeMoEval["NeMo Eval"]
        NEval["LLM evaluation<br/>with server-client<br/>approach"]
    end
    
    subgraph Deployment["NeMo Export-Deploy"]
        Deploy["model deployment<br/>for NeMo and HF"]
    end
    
    LMEval --> Factory
    SimpleEvals --> Factory
    BFCL --> Factory
    BigCode --> Factory
    Others --> Factory
    
    Factory -.->|"packaging<br/>of eval<br/>harnesses"| Factory
    Factory -->|"model<br/>querying"| NEval
    
    Deploy -->|"model<br/>serving"| NEval
    Garak --> Factory
    SafetyHarness --> Factory
    
    style EvalHarnesses fill:#e1f5fe
    style EvalFactory fill:#f3e5f5
    style NeMoEval fill:#e8f5e8
    style Deployment fill:#fff3e0
```

### **🚀 NeMo Evaluator Launcher** (Recommended for most users)

**Best for**: Researchers, ML engineers, and teams who want turnkey evaluation capabilities

**Use when you need**:

- Simple CLI commands to run evaluations
- Multi-backend execution (local, Slurm, cloud)
- Built-in monitoring and result export
- Reproducible configurations with minimal setup

```bash
# Quick start example
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct
```

**Architecture**:

```{mermaid}
graph LR
    A[User CLI/API] --> B[Launcher Orchestration]
    B --> C[Core Engine]
    B --> D[Executor Backend]
    B --> E[Result Exporters]
    
    C --> F[Evaluation Containers]
    D --> G[Local/Slurm/Cloud]
    E --> H[MLflow/W&B/Sheets]
    
    style B fill:#e1f5fe
```

### **⚙️ NeMo Evaluator Core** (For developers and integrations)

**Best for**: Developers building custom evaluation pipelines, integrating into existing systems

**Use when you need**:

- Programmatic control over evaluation workflows
- Custom request/response processing
- Integration into existing ML pipelines
- Framework extensions and custom benchmarks

```python
# Programmatic example
from nemo_evaluator.core.evaluate import evaluate
result = evaluate(eval_cfg=config, target_cfg=target, adapter_cfg=adapter)
```

**Architecture**:

```{mermaid}
graph LR
    A[Python API] --> B[Core Engine]
    B --> C[Adapter System]
    B --> D[Evaluation Containers]
    
    C --> E[Request Interceptors]
    C --> F[Response Processing]
    D --> G[Benchmark Frameworks]
    
    style B fill:#f3e5f5
```
