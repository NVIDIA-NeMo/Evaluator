---
orphan: true
---

<!-- markdownlint-disable MD041 -->

(adapters)=

# Evaluation Adapters

Evaluation adapters provide a flexible mechanism for intercepting and modifying requests/responses between the evaluation harness and the model endpoint. This allows for custom processing, logging, and transformation of data during the evaluation process.

## Concepts

For a conceptual overview and architecture diagram of adapters and interceptor chains, refer to {ref}`adapters-concepts`.

## Topics

Explore the following pages to use and configure adapters.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Usage
:link: adapters-usage
:link-type: ref
Learn how to enable adapters and pass `AdapterConfig` to `evaluate`.
:::

:::{grid-item-card} Recipes
:link: deployment-adapters-recipes
:link-type: ref
Reasoning cleanup, system prompt override, response shaping, logging caps.
:::

:::{grid-item-card} Configuration
:link: adapters-configuration
:link-type: ref
View available `AdapterConfig` options and defaults.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Usage <usage>
Recipes <recipes/index>
Configuration <configuration>
```
