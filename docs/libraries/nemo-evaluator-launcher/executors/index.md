# Executors

Executors run evaluations by orchestrating containerized benchmarks in different environments. They handle resource management, IO paths, and ensure reproducible results across various execution backends, from local development to large-scale cluster deployments.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` Overview
:link: overview
:link-type: doc

Understand how executors work, their common patterns, and key concepts for orchestrating containerized evaluations.
:::

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

## Choosing an Executor

Select the executor that best matches your environment and requirements:

- **Local**: Perfect for development, testing, and small evaluations on your machine
- **Slurm**: Ideal for HPC clusters and large-scale parallel evaluation workloads  
- **Lepton**: Best for cloud-native workflows with hosted models and automatic scaling

All executors provide the same evaluation guarantees and produce identical, reproducible results using the same containerized benchmarks.

:::{toctree}
:caption: Executors
:hidden:

Overview <overview>
Local Executor <local>
Slurm Executor <slurm>
Lepton Executor <lepton>
:::
