(about-concepts)=
# Concepts

Use this section to understand how {{ product_name_short }} works at a high level. Start with the evaluation model, then read about adapters and deployment choices.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Evaluation Model
:link: evaluation-model
:link-type: ref
Core evaluation types, OpenAI-compatible endpoints, and metrics.
:::

:::{grid-item-card} Framework Definition Files
:link: fdf-concept
:link-type: ref
YAML configuration files that integrate evaluation frameworks into NeMo Evaluator.
:::

:::{grid-item-card} Adapters & Interceptors
:link: adapters-interceptors
:link-type: doc
Advanced request/response processing with configurable interceptor pipelines.
:::

::::

```{toctree}
:hidden:

Architecture <architecture>
Evaluation Model <evaluation-model>
Framework Definition Files <framework-definition-file>
Adapters & Interceptors <adapters-interceptors>
```
