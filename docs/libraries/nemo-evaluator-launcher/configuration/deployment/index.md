# Deployment Configuration

Deployment configurations define how to provision and host model endpoints for evaluation.

:::{note}
For an overview of all deployment strategies and when to use launcher-orchestrated vs. bring-your-own-endpoint approaches, see {ref}`deployment-overview`.
:::

## Deployment Types

Choose the right deployment strategy for your evaluation needs:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` None (External)
:link: none
:link-type: doc

Use existing API endpoints like NVIDIA API Catalog, OpenAI, or custom deployments. No model deployment needed.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` vLLM
:link: vllm
:link-type: doc

High-performance LLM serving with advanced parallelism strategies. Best for production workloads and large models.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` SGLang
:link: sglang
:link-type: doc

Fast serving framework optimized for structured generation and high-throughput inference with efficient memory usage.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` NIM
:link: nim
:link-type: doc

NVIDIA-optimized inference microservices with automatic scaling, optimization, and enterprise-grade features.
:::

::::

## Quick Reference

```yaml
deployment:
  type: vllm  # or sglang, nim, none
  # ... deployment-specific settings
```

```{toctree}
:caption: Deployment Types
:hidden:

vLLM <vllm>
SGLang <sglang>
NIM <nim>
None (External) <none>
```
