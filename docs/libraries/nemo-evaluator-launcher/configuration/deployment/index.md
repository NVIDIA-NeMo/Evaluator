# Deployment Configuration

Deployment configurations define how to provision and host model endpoints for evaluation.

<!-- :::{note}
For an overview of all deployment strategies and when to use launcher-orchestrated vs. bring-your-own-endpoint approaches, see {ref}`deployment-overview`.
::: -->

## Deployment Types

Choose the deployment type for your evaluation:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` None (External)
:link: none
:link-type: doc

Use existing API endpoints. No model deployment needed.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` vLLM
:link: vllm
:link-type: doc

Deploy models using the vLLM serving framework.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` SGLang
:link: sglang
:link-type: doc

Deploy models using the SGLang serving framework.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` NIM
:link: nim
:link-type: doc

Deploy models using NVIDIA Inference Microservices.
:::


:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` TRT-LLM
:link: trtllm
:link-type: doc


Deploy models using NVIDIA TensorRT LLM.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Generic
:link: generic
:link-type: doc


Deploy models using a fully custom setup.
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
TensorRT-LLM <trtllm>
Generic <generic>
None (External) <none>
```
