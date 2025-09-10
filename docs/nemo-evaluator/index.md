# NeMo Evaluator: The Universal Platform for LLM Evaluation

NeMo Evaluator is an open-source evaluation engine that provides standardized, reproducible AI model evaluation through a containerized architecture and adapter system. It enables you to run evaluations across multiple specialized evaluation harnesses (17+ containers including LM-Eval, HELM, MT-Bench, and more) against any OpenAI-compatible model API. The platform's core strength lies in its interceptor-based adapter architecture that standardizes request/response flow, optional logging/caching layers, and its collection of ready-to-use evaluation containers published through NVIDIA's NGC catalog.

[Container Reference](./reference/containers.md) | [Using Containers](./workflows/using_containers.md) | [CLI Reference](./reference/cli.md) | [Configuration Guide](./reference/configuring_interceptors.md) | [Python API](./workflows/python-api.md)

---

The arch is as follows:


         ┌─────────────────────┐
         │                     │
         │  NeMo Evaluator     │
         │  harness            │
         └───▲──────┬──────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:3825)                            │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                                                    │  │   │                      │
         │   │ └─────────────────────-──────────────────────────────┘  │   │   deployed           │
         │   │                                                         │   │   OAI-compatible     │
         │   │                                                         │   │   model endpoint     │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │ ┌─┼──────────────────────────────────────────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘

In other words, interceptors are pieces of independent logic which should be
relatively easy to add separately.


The arch is as follows:


         ┌─────────────────────┐
         │                     │
         │  NeMo Evaluator     │
         │  harness            │
         └───▲──────┬──────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:3825)                            │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                                                    │  │   │                      │
         │   │ └─────────────────────-──────────────────────────────┘  │   │   deployed           │
         │   │                                                         │   │   OAI-compatible     │
         │   │                                                         │   │   model endpoint     │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │ ┌─┼──────────────────────────────────────────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘

In other words, interceptors are pieces of independent logic which should be
relatively easy to add separately.


NeMo Evaluator is the core, open-source evaluation engine that powers standardized, reproducible AI model evaluation across benchmarks. It provides the adapter/interceptor architecture, evaluation workflows, and the set of ready-to-use evaluation containers that ensure consistent results across environments and over time.

## How it differs from the Launcher
- **nemo-evaluator**: Core evaluation engine, adapter system, and evaluation containers. Focused on correctness, repeatability, and benchmark definitions.
- **nemo-evaluator-launcher**: Orchestration on top of the core engine. Adds a unified CLI, multi-backend execution (local/Slurm/hosted), job monitoring, and exporters. See the launcher intro: [nemo-evaluator-launcher](../nemo-evaluator-launcher/index.md)

## Key capabilities
- **Adapter/Interceptor architecture**: Standardizes how requests and responses flow to your endpoint (OpenAI-compatible) and through optional logging/caching layers
- **Benchmarks and containers**: Curated evaluation harnesses packaged as reproducible containers
  - Browse available containers: [Container Reference](./reference/containers.md)
- **Flexible configuration**: Fully resolved configs per run enable exact replays and comparisons
- **Metrics and artifacts**: Consistent result schemas and artifact layouts for downstream analysis

## Architecture overview
- Targets an OpenAI-compatible endpoint for the model under test
- Applies optional interceptors (request/response logging, caching, etc.)
- Executes benchmark tasks using the corresponding containerized framework
- Produces metrics, logs, and artifacts in a standard directory structure

## Using the core library
- **Python API**: Programmatic access to core evaluation functionality
  - API reference: [API Reference](./reference/api.md)
- **Containers**: Run evaluations using the published containers for each framework
  - Container reference: [Container Reference](./reference/containers.md)

For end-to-end CLI and multi-backend orchestration, use the Launcher: [nemo-evaluator-launcher](../nemo-evaluator-launcher/index.md)

## Extending
Add your own benchmark or framework by defining its configuration and interfaces:
- Extension guide: [Framework Definition File](./extending/framework_definition_file.md)

## Next steps
- Read the architecture details and glossary in the main docs
- Explore containers and pick the benchmarks you need: [Container Reference](./reference/containers.md)
- If you want a turnkey CLI, start with the Launcher Quickstart: [Quickstart](../nemo-evaluator-launcher/quickstart.md)

## NGC Containers

NeMo Evaluator provides pre-built evaluation containers through the NVIDIA NGC catalog:

| Container | Description | NGC Catalog | Latest Tag |
|-----------|-------------|-------------|------------|
| **agentic_eval** | Agentic AI evaluation harness | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) | `25.07.3` |
| **rag_retriever_eval** | RAG system evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) | `25.07.3` |
| **simple-evals** | Basic evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `25.07.3` |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `25.07.3` |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `25.07.3` |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `25.07.1` |
| **helm** | Holistic evaluation harness | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `25.07.2` |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `25.07.1` |
| **bfcl** | Cognitive load evaluation | [Link](https://catalog.ngc.nvidia.com/teams/eval-factory/containers/bfcl) | `25.07.3` |
| **garak** | Security and robustness testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `25.07.1` |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `25.07.3` |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `25.07.1` |

### Container Usage

```bash
# Pull a container
docker pull nvcr.io/nvidia/eval-factory/<container-name>:<tag>

# Example: Pull agentic evaluation container
docker pull nvcr.io/nvidia/eval-factory/agentic_eval:25.07.3

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/eval-factory/<container-name>:<tag>
```

## Disclaimer

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
