# Deployment Configuration
<!-- cSpell:ignore SGLang vLLM NIM -->

Deployment configurations define how to provision and host model endpoints for evaluation.

## Supported Types

The launcher supports the following deployment back ends:

<!-- vale off -->
- **[vLLM](vllm.md)**: High-performance LLM serving with optimized attention
- **[SGLang](sglang.md)**: Structured generation with efficient memory usage
- **[NIM](nim.md)**: NVIDIA-optimized inference microservices
- **[None](none.md)**: Use existing endpoints (no deployment)
<!-- vale on -->

## Quick Reference

Use the following minimal YAML structure:

```yaml
deployment:
  type: vllm  # or sglang, nim, none
  # ... deployment-specific settings
```

## When to Use

Choose a deployment type based on your environment:

<!-- vale off -->
- **vLLM**: General-purpose LLM serving
- **SGLang**: General-purpose LLM serving
- **NIM**: Deployments optimized for NVIDIA hardware
- **None**: Existing endpoints
<!-- vale on -->

## Configuration Files

Refer to the [source deployment configuration files](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/).
