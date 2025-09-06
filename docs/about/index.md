# About NeMo Evaluator

NeMo Evaluator is NVIDIA's comprehensive platform for AI model evaluation and benchmarking. It consists of two core libraries that work together to enable consistent, scalable, and reproducible evaluation of GenAI models spanning LLMs, VLMs, agentic AI, and retrieval systems.

![image](../../NeMo_Repo_Overview_Eval.png)

## System Architecture

NeMo Evaluator consists of two main libraries:

```{list-table} NeMo Evaluator Components
:header-rows: 1
:widths: 30 70

* - Component
  - Key Capabilities
* - **nemo-evaluator**  
    (Core Evaluation Engine)
  - • [Adapter/interceptor architecture](concepts/adapters-interceptors.md) for request and response processing  
    • Standardized evaluation workflows and containerized frameworks  
    • Deterministic configuration and reproducible results  
    • Consistent result schemas and artifact layouts
* - **nemo-evaluator-launcher**  
    (Orchestration Layer)
  - • Unified CLI and programmatic entry points  
    • Multi-backend execution (local, Slurm, cloud)  
    • Job monitoring and lifecycle management  
    • Result export to multiple destinations (MLflow, W&B, Google Sheets)
```

## Target Users

```{list-table} Target User Personas
:header-rows: 1
:widths: 30 70

* - User Type
  - Key Benefits
* - **Researchers**
  - Access 100+ benchmarks across 18 evaluation harnesses with containerized reproducibility. Run evaluations locally or on HPC clusters with minimal setup overhead.
* - **ML Engineers**
  - Integrate evaluations into ML pipelines with programmatic APIs. Deploy models and run evaluations across multiple backends with consistent, reproducible results.
* - **Organizations**
  - Scale evaluation across teams with unified CLI, multi-backend execution, and enterprise-grade result tracking. Export results to existing MLOps infrastructure.
* - **AI Safety Teams**
  - Conduct comprehensive safety assessments using specialized containers for security testing, bias evaluation, and alignment verification with detailed logging and audit trails.
* - **Model Developers**
  - Evaluate custom models against standard benchmarks using OpenAI-compatible APIs. Extend the platform with custom frameworks and evaluation tasks.
```
