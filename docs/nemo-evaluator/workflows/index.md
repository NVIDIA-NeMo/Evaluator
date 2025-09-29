(workflows-index)=
# Workflows and Usage Patterns

This section provides practical guides and workflows for using NeMo Evaluator in different scenarios. Whether you're integrating evaluations into existing pipelines or running containerized benchmarks, these guides will help you get started.

## Available Workflows

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} Python API
:link: python-api
:link-type: ref

Programmatic access to NeMo Evaluator through Python code. Learn how to integrate evaluations into ML pipelines, automate workflows, and build custom evaluation applications.
:::

::: {grid-item-card} Using Containers
:link: using-containers
:link-type: ref

Complete guide to using evaluation containers for running benchmarks. Covers container selection, execution patterns, and advanced configuration options.
:::

::::

Both approaches provide access to the full NeMo Evaluator feature set including interceptors, caching, logging, and comprehensive result artifacts.

---

```{toctree}
:maxdepth: 2
:caption: Workflows
:hidden:

python-api
using-containers
```
