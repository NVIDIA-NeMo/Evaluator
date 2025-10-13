(workflows-overview)=

# Workflows

Learn how to use NeMo Evaluator through different workflow patterns. Whether you prefer programmatic control through Python APIs or CLI, these guides provide practical examples for integrating evaluations into your ML pipelines.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` CLI
:link: cli
:link-type: doc

Run evaluations using the pre-built NGC containers and command line interface.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python API
:link: python-api
:link-type: doc

Use the NeMo Evaluator Python API to integrate evaluations directly into your existing ML pipelines and applications.
:::

::::

## Choose Your Workflow

- **Python API**: Integrate evaluations directly into your existing Python applications when you need dynamic configuration management or programmatic control
- **CLI**: Use CLI when you work with CI/CD systems, container orchestration platforms, or other non-interactive workflows.

Both approaches use the same underlying evaluation package and produce identical, reproducible results. Choose based on your integration requirements and preferred level of abstraction.

:::{toctree}
:caption: Workflows
:hidden:

CLI <cli>
Python API <python-api>
:::
