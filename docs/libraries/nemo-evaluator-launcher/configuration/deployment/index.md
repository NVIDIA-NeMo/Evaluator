# Deployment Configuration

Deployment configurations define how to provision and host model endpoints for evaluation.

:::{note}
For an overview of all deployment strategies and when to use launcher-orchestrated vs. bring-your-own-endpoint approaches, see [Deployment Overview](../../../deployment/index.md).
:::

## Supported Types

- **[vLLM](vllm.md)**: High-performance LLM serving with optimized attention
- **[SGLang](sglang.md)**: Structured generation with efficient memory usage  
- **[NIM](nim.md)**: NVIDIA-optimized inference microservices
- **[None](none.md)**: Use existing endpoints (no deployment)

## Quick Reference

```yaml
deployment:
  type: vllm  # or sglang, nim, none
  # ... deployment-specific settings
```

**When to use:**
- **vLLM**: General-purpose LLM serving
- **SGLang**: General-purpose LLM serving
- **NIM**: NVIDIA hardware optimized deployments
- **None**: Existing endpoints

## Configuration Files

See all available deployment configurations: [Deployment Configs](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/tree/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/deployment?ref_type=heads)

```{toctree}
:caption: Deployment Types
:hidden:

vLLM <vllm>
SGLang <sglang>
NIM <nim>
None (External) <none>
```
