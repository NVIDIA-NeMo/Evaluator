(about-concepts)=
# Concepts

Use this section to understand how NeMo Eval works at a high level. Start with the evaluation model, then read about adapters, deployment choices, and configuration.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Evaluation Model
:link: evaluation-model
:link-type: ref
Core evaluation types, OpenAI-compatible endpoints, and metrics.
:::

:::{grid-item-card} Adapters
:link: adapters
:link-type: ref
Reverse proxy with interceptor chains for request and response processing.
:::

:::{grid-item-card} Deployment Concepts
:link: deployment-concepts
:link-type: ref
Backend selection, scaling modes, and where adapters fit.
:::

:::{grid-item-card} Configuration Model
:link: configuration-model
:link-type: ref
Parameter layers, overrides, and authentication concepts.
:::

::::

```{toctree}
:hidden:

Architecture <architecture>
Evaluation Model <evaluation-model>
Adapters <adapters>
Deployment Concepts <deployment-concepts>
Configuration Model <configuration-model>
```
