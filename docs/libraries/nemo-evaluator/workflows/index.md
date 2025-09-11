(workflows-overview)=

# Container Workflows

Learn how to use NeMo Evaluator through different workflow patterns. Whether you prefer programmatic control through Python APIs or direct container usage, these guides provide practical examples for integrating evaluations into your ML pipelines.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Using Containers
:link: using_containers
:link-type: doc

Run evaluations using the pre-built NGC containers directly with Docker or container orchestration platforms.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python API
:link: python-api
:link-type: doc

Use the NeMo Evaluator Python API to integrate evaluations directly into your existing ML pipelines and applications.
:::

::::

## Choose Your Workflow

- **Python API**: Integrate evaluations directly into your existing Python applications when you need dynamic configuration management or programmatic control
- **Container Usage**: Use pre-built containers when you work with CI/CD systems, container orchestration platforms, or need complete control over the container environment

Both approaches use the same underlying evaluation containers and produce identical, reproducible results. Choose based on your integration requirements and preferred level of abstraction.

:::{toctree}
:caption: Container Workflows
:hidden:

Using Containers <using_containers>
Python API <python-api>
:::
