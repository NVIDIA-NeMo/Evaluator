(nemo-evaluator-index)=
# NeMo Evaluator: The Universal Platform for LLM Evaluation

NeMo Evaluator is an open-source evaluation engine. It provides standardized, reproducible AI model evaluation through a containerized architecture and adapter system. It enables you to run evaluations across 17+ specialized evaluation harnesses (containers including LM-Eval, HELM, MT-Bench, and others) against any OpenAI-compatible model API. The platform's core strength lies in its interceptor-based adapter architecture. This architecture standardizes request/response flow and optional logging/caching layers. It also includes a collection of ready-to-use evaluation containers published through NVIDIA's NGC catalog.

{doc}`./reference/containers` | {doc}`./workflows/using-containers` | {doc}`./reference/cli` | {doc}`./reference/configuring-interceptors` | {doc}`./workflows/python-api`

---

The architecture is as follows:

```{mermaid}
graph LR
    A[NeMo Evaluator harness] --> B[AdapterServer @ localhost:3825]
    B --> C[Chain of RequestInterceptors]
    C --> D[intcptr_1] --> E[intcptr_2] --> F[...] --> G[intcptr_N]
    G --> H[deployed OAI-compatible model endpoint]
    H --> I[intcptr'_1] --> J[intcptr'_2] --> K[...] --> L[intcptr'_M]
    L --> M[Chain of ResponseInterceptors]
    M --> A
    
    subgraph "AdapterServer"
        C
        D
        E
        F
        G
        I
        J
        K
        L
        M
    end
    
    subgraph "Request Flow"
        N[flask.Request passed on the way up]
    end
    
    subgraph "Response Flow"  
        O[requests.Response passed on the way down]
    end
```

Interceptors are independent units of logic designed for easy integration.

NeMo Evaluator is the core, open-source evaluation engine. It powers standardized, reproducible AI model evaluation across benchmarks. It provides the adapter/interceptor architecture, evaluation workflows, and ready-to-use evaluation containers. These components ensure consistent results across environments and over time.

## How It Differs from the Launcher

- **nemo-evaluator**: Core evaluation engine, adapter system, and evaluation containers. This component focuses on correctness, repeatability, and benchmark definitions.
- **nemo-evaluator-launcher**: Orchestration layer on top of the core engine. It adds a unified CLI, multi-backend execution (local/Slurm/hosted), job monitoring, and exporters. Refer to the {doc}`../nemo-evaluator-launcher/index` for more information.

## Key Capabilities

- **Adapter/Interceptor Architecture**: Standardizes how requests and responses flow to your endpoint (OpenAI-compatible) and through optional logging/caching layers
- **Benchmarks and Containers**: Curated evaluation harnesses packaged as reproducible containers
  - Browse available containers: {doc}`./reference/containers`
- **Flexible Configuration**: Fully resolved configurations per run enable exact replays and comparisons
- **Metrics and Artifacts**: Consistent result schemas and artifact layouts for downstream analysis

## Architecture Overview

- Targets an OpenAI-compatible endpoint for the model under test.
- Applies optional interceptors (request/response logging, caching, and others).
- Executes benchmark tasks using the corresponding containerized framework.
- Produces metrics, logs, and artifacts in a standard directory structure.

## Using the Core Library

- **Python API**: Programmatic access to core evaluation functionality
  - API reference: {doc}`./reference/api`
- **Containers**: Run evaluations using the published containers for each framework
  - Container reference: {doc}`./reference/containers`

For end-to-end CLI and multi-backend orchestration, use the {doc}`../nemo-evaluator-launcher/index`.

## Extending

Add your own benchmark or framework by defining its configuration and interfaces:

- Extension guide: {doc}`./extending/framework-definition-file`.

## Next Steps

- Read the architecture details and glossary in the main docs.
- Explore containers and pick the benchmarks you need: {doc}`./reference/containers`.
- If you want a turnkey CLI, start with the {doc}`../nemo-evaluator-launcher/tutorial`.

## NVIDIA NGC Containers

NeMo Evaluator provides pre-built evaluation containers through the NVIDIA NGC catalog:

```{list-table}
:header-rows: 1
:widths: 20 30 25 15 30

* - Container
  - Description  
  - NGC Catalog
  - Latest Tag
  - Key Benchmarks
* - **agentic_eval**
  - Agentic AI evaluation framework
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)
  - `25.08.1`
  - agentic_eval_answer_accuracy, agentic_eval_goal_accuracy_with_reference, agentic_eval_tool_call_accuracy
* - **bfcl**
  - Function calling
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl)
  - `25.08.1`
  - bfclv2, bfclv2_ast, bfclv3, bfclv3_ast
* - **bigcode-evaluation-harness**
  - Code generation evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness)
  - `25.08.1`
  - humaneval, humanevalplus, mbpp, mbppplus, multiple language variants
* - **garak**
  - Safety and vulnerability testing
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)
  - `25.08.1`
  - garak
* - **helm**
  - Holistic evaluation framework
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)
  - `25.08.1`
  - ci_bench, ehr_sql, head_qa, med_dialog variants, pubmed_qa
* - **lm-evaluation-harness**
  - Language model benchmarks
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)
  - `25.08.1`
  - mmlu, gsm8k, arc_challenge, hellaswag, humaneval, truthfulqa, ifeval
* - **mtbench**
  - Multi-turn conversation evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)
  - `25.08.1`
  - mtbench, mtbench-cor1
* - **simple-evals**
  - Common evaluation tasks
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)
  - `25.08.1`
  - mmlu, gpqa_diamond, humaneval, math_test_500, AIME variants
* - **safety-harness**
  - Safety and bias evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)
  - `25.08.1`
  - aegis_v2 variants, bbq_full, wildguard
* - **vlmevalkit**
  - Vision-language model evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)
  - `25.08.1`
  - ai2d_judge, chartqa, ocrbench, slidevqa
```

## Container Usage

```bash
# Pull a container
docker pull nvcr.io/nvidia/eval-factory/<container-name>:<tag>

# Example: Pull agentic evaluation container
docker pull nvcr.io/nvidia/eval-factory/agentic_eval:25.08.1

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/eval-factory/<container-name>:<tag>
```

## Disclaimer

This project will download and install third-party open source software projects. Review the license terms of these open source projects before use.

---

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

extending/framework-definition-file
workflows/index
reference/index
```
