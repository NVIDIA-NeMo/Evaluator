(lib-core)=

# NeMo Evaluator

The *Core Evaluation Engine* delivers standardized, reproducible AI model evaluation through containerized benchmarks and a flexible adapter architecture.

:::{tip}
**Need orchestration?** For CLI and multi-backend execution, use the [NeMo Evaluator Launcher](../nemo-evaluator-launcher/index.md).
:::

## Get Started

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Workflows
:link: workflows/index
:link-type: doc

Run evaluations using pre-built containers directly or integrate them through the Python API.
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containers
:link: containers/index
:link-type: doc

Ready-to-use evaluation containers with curated benchmarks and frameworks.
:::

::::

## Reference and Customization

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Interceptors
:link: interceptors/index
:link-type: doc

Set up interceptors to handle requests, responses, logging, caching, and custom processing.
:::

:::{grid-item-card} {octicon}`log;1.5em;sd-mr-1` Logging
:link: logging
:link-type: doc

Comprehensive logging setup for evaluation runs, debugging, and audit trails.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Extending
:link: extending/index
:link-type: doc

Add custom benchmarks and frameworks by defining configuration and interfaces.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` API Reference
:link: ../../../apidocs/index
:link-type: doc

Python API documentation for programmatic evaluation control and integration.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: cli
:link-type: doc

Command-line interface for direct container and evaluation execution.
:::

::::

:::{toctree}
:caption: NeMo Evaluator Core
:hidden:
About NeMo Evaluator <self>
Workflows <workflows/index>
Benchmark Containers <containers/index>
Interceptors <interceptors/index>
Logging <logging>
Extending <extending/index>
API Reference <api>
CLI Reference (nemo-evaluator) <cli>
:::
