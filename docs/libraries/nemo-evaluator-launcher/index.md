(lib-launcher)=

# NeMo Evaluator Launcher

The *Orchestration Layer* empowers you to run AI model evaluations at scale. Use the unified CLI and programmatic interfaces to discover benchmarks, configure runs, submit jobs, monitor progress, and export results.

:::{tip}
**New to evaluation?** Start with {ref}`gs-quickstart-launcher` for a step-by-step walkthrough.
:::

## Get Started

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Quickstart
:link: ../../get-started/quickstart/launcher
:link-type: doc

Step-by-step guide to install, configure, and run your first evaluation in minutes.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: configuration/index
:link-type: doc

Complete configuration schema, examples, and advanced patterns for all use cases.
:::

::::

## Execution

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Executors
:link: configuration/executors/index
:link-type: doc

Execute evaluations on your local machine, HPC cluster (Slurm), or cloud platform (Lepton AI).
:::

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Executor
:link: configuration/executors/local
:link-type: doc

Docker-based evaluation on your workstation. Perfect for development and testing.
:::

:::{grid-item-card} {octicon}`organization;1.5em;sd-mr-1` Slurm Executor
:link: configuration/executors/slurm
:link-type: doc

HPC cluster execution with automatic resource management and job scheduling.
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Lepton Executor
:link: configuration/executors/lepton
:link-type: doc

Cloud execution with on-demand GPU provisioning and automatic scaling.
:::

::::


## Export
::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`upload;1.5em;sd-mr-1` Exporters
:link: exporters/index
:link-type: doc

Export results to MLflow, Weights & Biases, Google Sheets, or local files with one command.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` MLflow Export
:link: exporters/mlflow
:link-type: doc

Export evaluation results and metrics to MLflow for experiment tracking.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` W&B Export
:link: exporters/wandb
:link-type: doc

Integrate with Weights & Biases for advanced visualization and collaboration.
:::

:::{grid-item-card} {octicon}`table;1.5em;sd-mr-1` Sheets Export
:link: exporters/gsheets
:link-type: doc

Export to Google Sheets for easy sharing and analysis with stakeholders.
:::

::::

## References

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python API
:link: api
:link-type: doc

Programmatic access for notebooks, automation, and custom evaluation workflows.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: cli
:link-type: doc

Complete command-line interface documentation with examples and usage patterns.
:::

::::

## Typical Workflow

1. **Choose execution backend** (local, Slurm, Lepton AI)
2. **Select example configuration** from the examples directory
3. **Point to your model endpoint** (OpenAI-compatible API)
4. **Launch evaluation** via CLI or Python API
5. **Monitor progress** and export results to your preferred platform

## When to Use the Launcher

Use the launcher whenever you want:
- **Unified interface** for running evaluations across different backends
- **Multi-benchmark coordination** with concurrent execution
- **Turnkey reproducibility** with saved configurations
- **Easy result export** to MLOps platforms and dashboards
- **Production-ready orchestration** with monitoring and lifecycle management

:::{toctree}
:caption: NeMo Evaluator Launcher
:hidden:

About NeMo Evaluator Launcher <self>
CLI Reference (nemo-evaluator-launcher) <cli>
Configuration <configuration/index>
Exporters <exporters/index>
Python API <api>
:::
