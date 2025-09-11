(executors-overview)=

# Executors

Executors run evaluations by orchestrating containerized benchmarks in different environments. They handle resource management, IO paths, and ensure reproducible results across various execution backends, from local development to large-scale cluster deployments.

**Core concepts**:
- Your model is separate from the evaluation container; communication is via an OpenAI‑compatible API
- Each benchmark runs in an open‑sourced Docker container for reproducibility  
- Execution backends can optionally manage model deployment

## Choosing an Executor

Select the executor that best matches your environment and requirements:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Local Executor
:link: local
:link-type: doc

Run evaluations on your local machine using Docker for rapid iteration and development workflows.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Slurm Executor
:link: slurm
:link-type: doc

Execute large-scale evaluations on Slurm-managed high-performance computing clusters with optional model deployment.
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Lepton Executor
:link: lepton
:link-type: doc

Run evaluations on Lepton AI's hosted infrastructure with automatic model deployment and scaling.
:::

::::

All executors provide the same evaluation guarantees and produce identical, reproducible results using the same containerized benchmarks.

:::{toctree}
:caption: Executors
:hidden:

Local Executor <local>
Slurm Executor <slurm>
Lepton Executor <lepton>
:::
