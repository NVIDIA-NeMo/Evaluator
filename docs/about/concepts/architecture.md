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